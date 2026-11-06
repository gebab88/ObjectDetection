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

// Copies an output tensor into a dense row-major float32 array. Core AI may hand
// back float16 tensors and non-contiguous layouts (strides are in elements), so
// both the scalar type and the strides are honoured instead of assuming packed
// float32 data.
private func denseFloats(from arr: NDArray) throws -> [Float] {
    let shape = arr.shape
    let rank = shape.count
    let scalarType = arr.scalarType
    guard scalarType == .float32 || scalarType == .float16 else {
        throw NSError(domain: "CoreAI", code: 4, userInfo: [NSLocalizedDescriptionKey: "Core AI output scalar type \(scalarType) is not supported (expected float32 or float16)"])
    }

    var numel = 1
    for v in shape { numel *= v }
    guard numel > 0 else { return [] }

    // Contiguous row-major strides, used when the tensor reports none.
    var contiguous = [Int](repeating: 1, count: rank)
    if rank > 1 {
        for d in stride(from: rank - 2, through: 0, by: -1) {
            contiguous[d] = contiguous[d + 1] * shape[d + 1]
        }
    }
    let reported = arr.strides
    guard reported.isEmpty || reported.count == rank else {
        throw NSError(domain: "CoreAI", code: 5, userInfo: [NSLocalizedDescriptionKey: "Core AI output strides do not match the tensor rank"])
    }
    let strides = reported.isEmpty ? contiguous : reported

    var floats = [Float](repeating: 0, count: numel)
    let raw = arr.rawView()
    raw.withUnsafeBytes { rawPtr, _, _ in
        if scalarType == .float32 && strides == contiguous {
            // Fast path: packed float32, copy in one go.
            floats.withUnsafeMutableBytes { dst in
                dst.baseAddress!.copyMemory(from: rawPtr, byteCount: numel * MemoryLayout<Float>.stride)
            }
            return
        }
        // General path: walk the logical indices in row-major order and gather
        // each element through the strides.
        var index = [Int](repeating: 0, count: rank)
        for linear in 0..<numel {
            var offset = 0
            for d in 0..<rank { offset += index[d] * strides[d] }
            if scalarType == .float32 {
                floats[linear] = rawPtr.load(fromByteOffset: offset * MemoryLayout<Float>.stride, as: Float.self)
            } else {
                floats[linear] = Float(rawPtr.load(fromByteOffset: offset * MemoryLayout<Float16>.stride, as: Float16.self))
            }
            var d = rank - 1
            while d >= 0 {
                index[d] += 1
                if index[d] < shape[d] { break }
                index[d] = 0
                d -= 1
            }
        }
    }
    return floats
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
                        _ outputShapeCapacity: Int64,
                        _ outputRank: UnsafeMutablePointer<Int64>) -> Int32 {
    guard inputNumel > 0, inputRank > 0, outputCapacity > 0, outputShapeCapacity > 0 else {
        return -1
    }

    // Input tensor shape (rank entries, e.g. [1, 3, H, W] for a CHW [0,1] frame blob).
    var shapeInts: [Int] = []
    shapeInts.reserveCapacity(Int(inputRank))
    for i in 0..<Int(inputRank) {
        shapeInts.append(Int(inputShape[i]))
    }

    var inputNDArray = NDArray(shape: shapeInts, scalarType: .float32)
    // Populate the input tensor with the preprocessed frame data passed in by the
    // C++ backend; the NDArray above is zero-filled by default, so without this
    // copy the model would run on an all-zero image.
    inputNDArray.mutableRawView().withUnsafeMutableBytes { rawPtr, _, _ in
        let buf = rawPtr.assumingMemoryBound(to: Float.self)
        for i in 0..<Int(inputNumel) {
            buf[i] = inputData[i]
        }
    }

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
            guard let name = o.names.first, let value = o.remove(name) else { throw NSError(domain: "CoreAI", code: 2, userInfo: [NSLocalizedDescriptionKey: "Core AI inference returned no value"]) }
            guard let arr = value.ndArray else { throw NSError(domain: "CoreAI", code: 3, userInfo: [NSLocalizedDescriptionKey: "Core AI inference output is not a tensor"]) }
            outcome = try denseFloats(from: arr)
            outShape = arr.shape.map { Int64($0) }
        } catch {
            taskErr = error
        }
        sem.signal()
    }
    sem.wait()
    if taskErr != nil { return -1 }
    guard let floats = outcome else { return -1 }
    // Fail closed instead of writing past the caller's buffers: the output values
    // must fit into outputData and the dimensions into outputShape.
    guard !floats.isEmpty, floats.count <= Int(outputCapacity) else { return -1 }
    guard outShape.count <= Int(outputShapeCapacity) else { return -1 }
    floats.withUnsafeBufferPointer { src in
        outputData.update(from: src.baseAddress!, count: src.count)
    }
    outputNumel[0] = Int64(floats.count)
    for i in 0..<outShape.count {
        outputShape[i] = outShape[i]
    }
    outputRank[0] = Int64(outShape.count)
    return 0
}
