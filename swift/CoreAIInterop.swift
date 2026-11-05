// Swift interop layer for Apple's Core AI framework (AIModel / InferenceFunction /
// InferenceValue) and the C entry point `coreAIInfer` consumed by the C++ Core AI
// backend (Detection_CoreAI).
//
// The entry point is a synchronous C function (required by the C++ backend) that
// bridges onto the Swift concurrency runtime: the Core AI inference runs on a
// background task and the caller blocks until it finishes.

import CoreAI
import Foundation
import Dispatch

final class CoreAIState {
    static let shared = CoreAIState()
    private let lock = NSLock()
    private var function: InferenceFunction?

    func loadFunction(modelPath: String) throws -> InferenceFunction {
        lock.lock()
        defer { lock.unlock() }
        if let f = function { return f }
        let f = try Self.loadBlocking(modelPath: modelPath)
        function = f
        return f
    }

    private static func loadBlocking(modelPath: String) throws -> InferenceFunction {
        let sem = DispatchSemaphore(value: 0)
        var result: InferenceFunction? = nil
        var loadErr: Error? = nil
        Task {
            do {
                let model = try await AIModel(contentsOf: URL(fileURLWithPath: modelPath), options: .default)
                result = try model.loadFunction(named: "main")
            } catch {
                loadErr = error
            }
            sem.signal()
        }
        sem.wait()
        if let loadErr = loadErr { throw loadErr }
        guard let f = result else { throw NSError(domain: "CoreAI", code: 1, userInfo: [NSLocalizedDescriptionKey: "Core AI model function not found"]) }
        return f
    }
}

@_cdecl("coreAIInfer")
public func coreAIInfer(_ inputData: UnsafePointer<Float>,
                        _ inputNumel: Int64,
                        _ inputShape: UnsafePointer<Int64>,
                        _ inputRank: Int64,
                        _ modelPath: UnsafePointer<CChar>,
                        _ outputData: UnsafeMutablePointer<Float>,
                        _ outputCapacity: Int64,
                        _ outputNumel: UnsafeMutablePointer<Int64>,
                        _ outputShape: UnsafeMutablePointer<Int64>,
                        _ outputShapeCapacity: Int64) -> Int {
    guard inputNumel > 0, inputRank > 0, outputCapacity > 0 else {
        return -1
    }

    // Input tensor shape (rank entries, e.g. [1, 3, H, W] for a CHW [0,1] frame blob).
    var shapeInts: [Int] = []
    shapeInts.reserveCapacity(Int(inputRank))
    for i in 0..<Int(inputRank) {
        shapeInts.append(Int(inputShape[i]))
    }

    let inputNDArray = NDArray(shape: shapeInts, scalarType: .float32)

    // The C entry point is synchronous, so the (async) Core AI inference runs on a
    // background task and the caller blocks until it completes.
    let sem = DispatchSemaphore(value: 0)
    var outcome: [Float]? = nil
    var taskErr: Error? = nil
    var outShape: [Int64] = []
    Task {
        do {
            let function = try CoreAIState.shared.loadFunction(modelPath: String(cString: modelPath))
            let inputs: [String: NDArray] = ["images": inputNDArray]
            let outputs = try await function.run(inputs: inputs)
            var o = outputs
            let name = o.names.first
            guard let value = o.remove(name!) else { throw NSError(domain: "CoreAI", code: 2, userInfo: [NSLocalizedDescriptionKey: "Core AI inference returned no value"]) }
            guard let arr = value.ndArray else { throw NSError(domain: "CoreAI", code: 3, userInfo: [NSLocalizedDescriptionKey: "Core AI inference output is not a tensor"]) }
            let shape = arr.shape
            var numel: Int64 = 1
            for v in shape { numel = numel * Int64(v) }
            var floats: [Float] = []
            if numel > 0, numel <= Int64(outputCapacity) {
                let raw = arr.rawView()
                raw.withUnsafeBytes { rawPtr, shp, str in
                    var acc: [Float] = []
                    acc.reserveCapacity(Int(numel))
                    var buf = rawPtr.assumingMemoryBound(to: Float.self)
                    for _ in 0..<Int(numel) {
                        acc.append(buf.pointee)
                        buf = buf.advanced(by: 1)
                    }
                    floats = acc
                    return
                }
            }
            outcome = floats
            outShape = shape.map { Int64($0) }
        } catch {
            taskErr = error
        }
        sem.signal()
    }
    sem.wait()
    if taskErr != nil { return -1 }
    guard let floats = outcome else { return -1 }
    guard floats.count <= Int(outputCapacity) else { return -1 }
    for i in 0..<Int(floats.count) {
        outputData[i] = floats[i]
    }
    outputNumel[0] = Int64(floats.count)
    for i in 0..<Int(outShape.count) {
        outputShape[i] = outShape[i]
    }
    return 0
}
