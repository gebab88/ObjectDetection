// Swift interop layer for Apple's Core AI framework (AIModel / InferenceFunction /
// InferenceValue) and the C entry point `coreAIInfer` consumed by the C++ Core AI
// backend (Detection_CoreAI).
//
// The entry point is a synchronous C function (required by the C++ backend) that
// bridges onto the Swift concurrency runtime: model loading and inference run in a
// single background task that never blocks, and only the calling C++ thread (which
// is not part of Swift's cooperative thread pool) waits for that task to finish.
// Blocking inside a task on another task would occupy a cooperative pool thread and
// can deadlock once the pool is exhausted (reproducible with
// LIBDISPATCH_COOPERATIVE_POOL_STRICT=1, which limits the pool to one thread).

import CoreAI
import Foundation
import Dispatch

// Caches the loaded model function per model path. Loading is async and awaited
// rather than blocked on; concurrent first calls share one in-flight load task.
actor CoreAIState {
    static let shared = CoreAIState()
    private var modelPath: String?
    private var loadTask: Task<InferenceFunction, Error>?

    func function(modelPath path: String) async throws -> InferenceFunction {
        if let task = loadTask, modelPath == path {
            return try await task.value
        }
        let task = Task<InferenceFunction, Error> {
            let model = try await AIModel(contentsOf: URL(fileURLWithPath: path), options: specializationOptions())
            guard let function = try model.loadFunction(named: "main") else {
                throw NSError(domain: "CoreAI", code: 1, userInfo: [NSLocalizedDescriptionKey: "Core AI model function not found"])
            }
            return function
        }
        modelPath = path
        loadTask = task
        do {
            return try await task.value
        } catch {
            // Do not cache a failed load, so a later call can retry.
            if modelPath == path {
                modelPath = nil
                loadTask = nil
            }
            throw error
        }
    }
}

// Compute unit the model is specialized for, from COREAI_COMPUTE_UNIT:
//   gpu (default), neural_engine, cpu, or all (let Core AI pick).
// The GPU is the default because it was the fastest unit for the YOLO26 models
// measured (yolo26x on an M5 Pro: 28 ms per frame on the GPU, 30 ms with Core
// AI's own choice, with two frames in flight); smaller models may run better on
// the Neural Engine. Unsupported operations still fall back to the other units.
private func specializationOptions() -> SpecializationOptions {
    let value = ProcessInfo.processInfo.environment["COREAI_COMPUTE_UNIT"]?.lowercased() ?? "gpu"
    let options: SpecializationOptions
    let label: String
    switch value {
    case "", "gpu":
        options = SpecializationOptions(preferredComputeUnitKind: .gpu)
        label = "GPU"
    case "neural_engine", "ane":
        options = SpecializationOptions(preferredComputeUnitKind: .neuralEngine)
        label = "Neural Engine"
    case "cpu":
        options = .cpuOnly
        label = "CPU only"
    case "all":
        options = .default
        label = "Core AI's choice"
    default:
        options = SpecializationOptions(preferredComputeUnitKind: .gpu)
        label = "GPU (unknown COREAI_COMPUTE_UNIT '\(value)'; use gpu, neural_engine, cpu or all)"
    }
    FileHandle.standardError.write(Data("[CoreAI] Compute unit: \(label)\n".utf8))
    return options
}

private func coreAIError(_ code: Int, _ message: String) -> NSError {
    NSError(domain: "CoreAI", code: code, userInfo: [NSLocalizedDescriptionKey: message])
}

// Reports a Core AI failure on stderr, once per distinct message: the C entry point
// only returns a status code, so without this the reason (e.g. an input mismatch)
// would never reach the user, and once-only keeps the frame loop from spamming.
final class CoreAIErrorLog {
    static let shared = CoreAIErrorLog()
    private let lock = NSLock()
    private var reported = Set<String>()

    func report(_ error: Error) {
        let message = "[CoreAI] Inference failed: \(error.localizedDescription)\n"
        lock.lock()
        let isNew = reported.insert(message).inserted
        lock.unlock()
        if isNew {
            FileHandle.standardError.write(Data(message.utf8))
        }
    }
}

// Resolves the model's image input from its descriptor instead of assuming the
// Ultralytics export name "images", and checks that the preprocessed frame (float32
// tensor of `shape`) matches what the model expects, so a mismatch fails with a
// clear message rather than an opaque inference error.
private func imageInputName(of function: InferenceFunction, shape: [Int]) throws -> String {
    let descriptor = function.descriptor
    let names = descriptor.inputNames
    let name: String
    if names.count == 1 {
        name = names[0]
    } else if names.contains("images") {
        name = "images"
    } else if names.isEmpty {
        throw coreAIError(6, "Core AI model has no inputs")
    } else {
        throw coreAIError(6, "Core AI model has \(names.count) inputs (\(names.joined(separator: ", "))); cannot tell which one takes the image")
    }

    switch descriptor.inputDescriptor(of: name) {
    case .ndArray(let input)?:
        guard input.scalarType == .float32 else {
            throw coreAIError(7, "Core AI model input '\(name)' expects \(input.scalarType) data; only float32 is supported")
        }
        if !input.hasDynamicShape && input.shape != shape {
            throw coreAIError(8, "Core AI model input '\(name)' expects shape \(input.shape), got \(shape); check model_width/model_height in config.yaml")
        }
    case .image?:
        throw coreAIError(7, "Core AI model input '\(name)' expects an image (pixel buffer); only tensor inputs are supported")
    case nil:
        throw coreAIError(6, "Core AI model input '\(name)' has no descriptor")
    @unknown default:
        throw coreAIError(7, "Core AI model input '\(name)' has an unsupported kind")
    }
    return name
}

// Copies an output tensor as dense row-major float32 straight into the caller's
// buffer `dst` (no intermediate array) and returns the number of values. If they
// do not fit into `capacity`, nothing is written and the required count is
// returned with `fits == false`. Core AI may hand back float16 tensors and
// non-contiguous layouts (strides are in elements), so both the scalar type and
// the strides are honoured instead of assuming packed float32 data.
private func copyDenseFloats(from arr: NDArray, into dst: UnsafeMutablePointer<Float>,
                             capacity: Int) throws -> (count: Int, fits: Bool) {
    let shape = arr.shape
    let rank = shape.count
    let scalarType = arr.scalarType
    guard scalarType == .float32 || scalarType == .float16 else {
        throw NSError(domain: "CoreAI", code: 4, userInfo: [NSLocalizedDescriptionKey: "Core AI output scalar type \(scalarType) is not supported (expected float32 or float16)"])
    }

    var numel = 1
    for v in shape { numel *= v }
    guard numel > 0 else { return (0, true) }
    guard numel <= capacity else { return (numel, false) }

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

    let raw = arr.rawView()
    raw.withUnsafeBytes { rawPtr, _, _ in
        if scalarType == .float32 && strides == contiguous {
            // Fast path: packed float32, copy in one go.
            UnsafeMutableRawPointer(dst).copyMemory(from: rawPtr, byteCount: numel * MemoryLayout<Float>.stride)
            return
        }
        // General path: walk the logical indices in row-major order and gather
        // each element through the strides.
        var index = [Int](repeating: 0, count: rank)
        for linear in 0..<numel {
            var offset = 0
            for d in 0..<rank { offset += index[d] * strides[d] }
            if scalarType == .float32 {
                dst[linear] = rawPtr.load(fromByteOffset: offset * MemoryLayout<Float>.stride, as: Float.self)
            } else {
                dst[linear] = Float(rawPtr.load(fromByteOffset: offset * MemoryLayout<Float16>.stride, as: Float16.self))
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
    return (numel, true)
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

    // The NDArray holds exactly product(shape) floats; copying a different count
    // would read past the caller's data or write past the tensor.
    var shapeNumel = 1
    for v in shapeInts { shapeNumel *= v }
    guard shapeNumel == Int(inputNumel) else {
        CoreAIErrorLog.shared.report(coreAIError(10, "Core AI input has \(inputNumel) values but its shape \(shapeInts) needs \(shapeNumel)"))
        return -1
    }

    var inputNDArray = NDArray(shape: shapeInts, scalarType: .float32)
    var contiguousStrides = [Int](repeating: 1, count: shapeInts.count)
    if shapeInts.count > 1 {
        for d in stride(from: shapeInts.count - 2, through: 0, by: -1) {
            contiguousStrides[d] = contiguousStrides[d + 1] * shapeInts[d + 1]
        }
    }
    guard inputNDArray.strides.isEmpty || inputNDArray.strides == contiguousStrides else {
        CoreAIErrorLog.shared.report(coreAIError(10, "Core AI input tensor has non-contiguous strides \(inputNDArray.strides)"))
        return -1
    }
    // Populate the input tensor with the preprocessed frame data passed in by the
    // C++ backend (the NDArray is zero-filled by default, so without this copy the
    // model would run on an all-zero image). A freshly created NDArray uses the
    // contiguous row-major layout of the CHW blob, so one bulk copy suffices.
    inputNDArray.mutableRawView().withUnsafeMutableBytes { rawPtr, _, _ in
        rawPtr.copyMemory(from: UnsafeRawPointer(inputData), byteCount: shapeNumel * MemoryLayout<Float>.stride)
    }

    // The C entry point is synchronous, so the (async) Core AI inference runs on a
    // background task and the caller blocks until it completes.
    let path = String(cString: modelPath)
    let sem = DispatchSemaphore(value: 0)
    let capacity = Int(outputCapacity)
    let shapeCapacity = Int(outputShapeCapacity)
    var valueCount = 0
    var fits = false
    var taskErr: Error? = nil
    var outShape: [Int64] = []
    Task {
        do {
            let function = try await CoreAIState.shared.function(modelPath: path)
            let inputName = try imageInputName(of: function, shape: shapeInts)
            let inputs: [String: NDArray] = [inputName: inputNDArray]
            let outputs = try await function.run(inputs: inputs)
            var o = outputs
            // Take the model's first declared output (deterministic order from the
            // descriptor); fall back to whatever the result holds.
            guard let name = function.descriptor.outputNames.first ?? o.names.first,
                  let value = o.remove(name) else { throw coreAIError(2, "Core AI inference returned no value") }
            guard let arr = value.ndArray else { throw coreAIError(3, "Core AI inference output '\(name)' is not a tensor") }
            // Never write past the caller's buffers. Too many dimensions is an
            // error; too many values is not: the required count is reported with
            // status 2 so the caller can grow its buffer and call again (the output
            // size depends on the model's input size and class count, which the
            // caller cannot know up front).
            guard arr.shape.count <= shapeCapacity else {
                throw coreAIError(9, "Core AI output has rank \(arr.shape.count); at most \(shapeCapacity) dimensions are supported")
            }
            (valueCount, fits) = try copyDenseFloats(from: arr, into: outputData, capacity: capacity)
            outShape = arr.shape.map { Int64($0) }
        } catch {
            taskErr = error
        }
        sem.signal()
    }
    sem.wait()
    if let taskErr = taskErr {
        CoreAIErrorLog.shared.report(taskErr)
        return -1
    }
    outputNumel[0] = Int64(valueCount)
    if !fits {
        return 2
    }
    // An empty output (e.g. an end-to-end head with zero detections) is a valid
    // result, not a failure.
    for i in 0..<outShape.count {
        outputShape[i] = outShape[i]
    }
    outputRank[0] = Int64(outShape.count)
    return 0
}
