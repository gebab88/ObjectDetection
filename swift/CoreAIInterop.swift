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

import Accelerate
import CoreAI
import Foundation
import Dispatch

// Caches the loaded model function per model path. Loading is async and awaited
// rather than blocked on; concurrent first calls share one in-flight load task.
actor CoreAIState {
    static let shared = CoreAIState()
    private var modelPath: String?
    private var loadTasks: [ComputeUnit: Task<InferenceFunction, Error>] = [:]
    // Units whose cached specialization was already rebuilt for this model.
    private var rebuilt = Set<ComputeUnit>()

    // Returns the model function specialized for `unit`. The first call loads
    // the specializations of all configured lanes at once and waits for all of
    // them, so the warm-up shows in the first frame instead of stalling a later
    // frame mid-run when its lane first needs another unit.
    func function(modelPath path: String, unit: ComputeUnit) async throws -> InferenceFunction {
        if modelPath != path {
            modelPath = path
            loadTasks = [:]
            rebuilt = []
        }
        var started: [Task<InferenceFunction, Error>] = []
        for lane in Set(computeLanes).union([unit]) where loadTasks[lane] == nil {
            let task = Task<InferenceFunction, Error> {
                let model = try await AIModel(contentsOf: URL(fileURLWithPath: path), options: lane.options)
                guard let function = try model.loadFunction(named: "main") else {
                    throw NSError(domain: "CoreAI", code: 1, userInfo: [NSLocalizedDescriptionKey: "Core AI model function not found"])
                }
                return function
            }
            loadTasks[lane] = task
            started.append(task)
        }
        for other in started {
            _ = try? await other.value  // failures surface on the lane that needs it
        }
        guard let task = loadTasks[unit] else {
            throw coreAIError(1, "Core AI model load for \(unit.label) was reset")
        }
        do {
            return try await task.value
        } catch {
            // Do not cache a failed load, so a later call can retry.
            if modelPath == path {
                loadTasks[unit] = nil
            }
            throw error
        }
    }

    // Returns the function for `unit` after its specialization failed to load
    // or run. Core AI's cache keeps the specialized model, but the compiled
    // Neural Engine program it refers to lives in the Neural Engine daemon's
    // own cache. That one can lack the program while Core AI's entry exists:
    // it keeps programs per code-signing identifier (Core AI's cache is shared
    // by executables of the same name), and it was seen to drop a program
    // after the model file the entry was first built from had been deleted (a
    // copy with the same content reuses the entry). Core AI then fails with a
    // generic error instead of compiling again. Deleting the entry makes it
    // specialize anew, so that is done once per unit; every caller that failed
    // on the old function gets the new one.
    func rebuiltFunction(modelPath path: String, unit: ComputeUnit) async throws -> InferenceFunction {
        if modelPath == path && !rebuilt.contains(unit) {
            rebuilt.insert(unit)
            loadTasks[unit] = nil
            try AIModelCache.default.deleteEntry(for: URL(fileURLWithPath: path), options: unit.options)
            CoreAINotice.shared.once("The cached \(unit.label) specialization of the model could not be used; rebuilding it (takes a few seconds once).")
        }
        return try await function(modelPath: path, unit: unit)
    }
}

// Compute unit a model specialization prefers. Operations the unit cannot run
// still fall back to the others.
enum ComputeUnit: Hashable {
    case gpu, neuralEngine, cpu, all

    var options: SpecializationOptions {
        switch self {
        case .gpu: return SpecializationOptions(preferredComputeUnitKind: .gpu)
        case .neuralEngine: return SpecializationOptions(preferredComputeUnitKind: .neuralEngine)
        case .cpu: return .cpuOnly
        case .all: return .default
        }
    }

    var label: String {
        switch self {
        case .gpu: return "GPU"
        case .neuralEngine: return "Neural Engine"
        case .cpu: return "CPU"
        case .all: return "Core AI's choice"
        }
    }
}

// One lane per frame in flight, each running its inferences on a fixed compute
// unit (from COREAI_COMPUTE_UNIT):
//   gpu+neural_engine (default): GPU, GPU, Neural Engine, twice
//   gpu, neural_engine, all: four lanes on that unit; cpu: two
// The default was the fastest layout measured. With an FP16 yolo26x on an M5
// Pro one GPU inference takes 11 ms and one on the Neural Engine 19 ms, so the
// GPU gets two frames for each one on the Neural Engine (6.3 ms per frame in a
// pure inference benchmark, against 8.9 ms on the GPU alone). Each lane appears
// twice so that one frame's preprocessing and decoding overlap the other's
// inference instead of leaving the unit idle (in the app: 7.2 ms per frame with
// three lanes, 6.5 ms with six; on the GPU alone 9.7 ms with two lanes, 9.0-9.5
// with four). FP32 models barely use the Neural Engine; their Neural Engine
// specialization runs mostly on the GPU, which then simply gets more lanes.
// Mixing units needs a workaround for a Core AI issue (see the all-zero rerun
// in coreAIInfer).
let computeLanes: [ComputeUnit] = {
    let value = ProcessInfo.processInfo.environment["COREAI_COMPUTE_UNIT"]?.lowercased() ?? ""
    var note = ""
    let mixed: [ComputeUnit] = [.gpu, .gpu, .neuralEngine, .gpu, .gpu, .neuralEngine]
    let lanes: [ComputeUnit]
    switch value {
    case "", "gpu+neural_engine", "gpu+ane": lanes = mixed
    case "gpu": lanes = [.gpu, .gpu, .gpu, .gpu]
    case "neural_engine", "ane": lanes = [.neuralEngine, .neuralEngine, .neuralEngine, .neuralEngine]
    case "cpu": lanes = [.cpu, .cpu]  // more would only share the same cores
    case "all": lanes = [.all, .all, .all, .all]
    default:
        lanes = mixed
        note = " (unknown COREAI_COMPUTE_UNIT '\(value)'; use gpu+neural_engine, gpu, neural_engine, cpu or all)"
    }
    let labels = lanes.map(\.label).joined(separator: ", ")
    FileHandle.standardError.write(Data("[CoreAI] Compute units per frame in flight: \(labels)\(note)\n".utf8))
    return lanes
}()

// Whether lanes run on different units (GPU and Neural Engine at once).
let lanesMixUnits = Set(computeLanes).count > 1

// Prints an informational [CoreAI] note on stderr, once per distinct message.
final class CoreAINotice {
    static let shared = CoreAINotice()
    private let lock = NSLock()
    private var shown = Set<String>()

    func once(_ message: String) {
        lock.lock()
        let isNew = shown.insert(message).inserted
        lock.unlock()
        if isNew {
            FileHandle.standardError.write(Data("[CoreAI] \(message)\n".utf8))
        }
    }
}

// Number of frames the C++ backend keeps in flight: one per lane.
@_cdecl("coreAIFramesInFlight")
public func coreAIFramesInFlight() -> Int64 {
    Int64(computeLanes.count)
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
// Ultralytics export name "images", and checks that the preprocessed frame (tensor
// of `shape`) matches what the model expects, so a mismatch fails with a clear
// message rather than an opaque inference error. Returns the input name and its
// scalar type: float32, or float16 for FP16 exports (`quantize=16`), into which
// the float32 frame is converted.
private func imageInput(of function: InferenceFunction, shape: [Int]) throws -> (name: String, scalarType: NDArray.ScalarType) {
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

    let scalarType: NDArray.ScalarType
    switch descriptor.inputDescriptor(of: name) {
    case .ndArray(let input)?:
        guard input.scalarType == .float32 || input.scalarType == .float16 else {
            throw coreAIError(7, "Core AI model input '\(name)' expects \(input.scalarType) data; only float32 and float16 are supported")
        }
        scalarType = input.scalarType
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
    return (name, scalarType)
}

// Converts `count` floats between float32 and float16 with vImage (SIMD), which
// is several times faster than converting element by element.
private func convertFloat32ToFloat16(_ src: UnsafeRawPointer, _ dst: UnsafeMutableRawPointer, count: Int) throws {
    var source = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: src), height: 1,
                               width: vImagePixelCount(count), rowBytes: count * MemoryLayout<Float>.stride)
    var destination = vImage_Buffer(data: dst, height: 1, width: vImagePixelCount(count),
                                    rowBytes: count * MemoryLayout<Float16>.stride)
    let error = vImageConvert_PlanarFtoPlanar16F(&source, &destination, vImage_Flags(kvImageNoFlags))
    guard error == kvImageNoError else {
        throw coreAIError(11, "Converting the input to float16 failed (vImage error \(error))")
    }
}

private func convertFloat16ToFloat32(_ src: UnsafeRawPointer, _ dst: UnsafeMutableRawPointer, count: Int) throws {
    var source = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: src), height: 1,
                               width: vImagePixelCount(count), rowBytes: count * MemoryLayout<Float16>.stride)
    var destination = vImage_Buffer(data: dst, height: 1, width: vImagePixelCount(count),
                                    rowBytes: count * MemoryLayout<Float>.stride)
    let error = vImageConvert_Planar16FtoPlanarF(&source, &destination, vImage_Flags(kvImageNoFlags))
    guard error == kvImageNoError else {
        throw coreAIError(11, "Converting the output from float16 failed (vImage error \(error))")
    }
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
    try raw.withUnsafeBytes { rawPtr, _, _ in
        if scalarType == .float32 && strides == contiguous {
            // Fast path: packed float32, copy in one go.
            UnsafeMutableRawPointer(dst).copyMemory(from: rawPtr, byteCount: numel * MemoryLayout<Float>.stride)
            return
        }
        if scalarType == .float16 && strides == contiguous {
            // Fast path: packed float16 (FP16 exports), converted in one go.
            try convertFloat16ToFloat32(rawPtr, UnsafeMutableRawPointer(dst), count: numel)
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
                        _ outputRank: UnsafeMutablePointer<Int64>,
                        _ lane: Int64) -> Int32 {
    guard inputNumel > 0, inputRank > 0, outputCapacity > 0, outputShapeCapacity > 0, lane >= 0 else {
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

    var contiguousStrides = [Int](repeating: 1, count: shapeInts.count)
    if shapeInts.count > 1 {
        for d in stride(from: shapeInts.count - 2, through: 0, by: -1) {
            contiguousStrides[d] = contiguousStrides[d + 1] * shapeInts[d + 1]
        }
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
            let unit = computeLanes[Int(lane) % computeLanes.count]
            // A specialization that fails to load or run is rebuilt once (see
            // rebuiltFunction); if that does not help, the first error is reported.
            var rebuilt = false
            func rebuiltFunction(after error: Error) async throws -> InferenceFunction {
                guard !rebuilt, let fresh = try? await CoreAIState.shared.rebuiltFunction(modelPath: path, unit: unit) else {
                    throw error
                }
                rebuilt = true
                return fresh
            }
            var function: InferenceFunction
            do {
                function = try await CoreAIState.shared.function(modelPath: path, unit: unit)
            } catch {
                function = try await rebuiltFunction(after: error)
            }
            let input = try imageInput(of: function, shape: shapeInts)
            var inputNDArray = NDArray(shape: shapeInts, scalarType: input.scalarType)
            guard inputNDArray.strides.isEmpty || inputNDArray.strides == contiguousStrides else {
                throw coreAIError(10, "Core AI input tensor has non-contiguous strides \(inputNDArray.strides)")
            }
            // Populate the input tensor with the preprocessed frame data passed in by
            // the C++ backend (the NDArray is zero-filled by default, so without this
            // the model would run on an all-zero image). A freshly created NDArray
            // uses the contiguous row-major layout of the CHW blob, so one bulk copy
            // (or conversion, for float16 models) suffices.
            try inputNDArray.mutableRawView().withUnsafeMutableBytes { rawPtr, _, _ in
                if input.scalarType == .float16 {
                    try convertFloat32ToFloat16(UnsafeRawPointer(inputData), rawPtr, count: shapeNumel)
                } else {
                    rawPtr.copyMemory(from: UnsafeRawPointer(inputData), byteCount: shapeNumel * MemoryLayout<Float>.stride)
                }
            }
            let inputs: [String: NDArray] = [input.name: inputNDArray]
            // With GPU and Neural Engine lanes running at the same time, Core AI
            // (macOS 27.2, coreai-core 1.0.0b3) occasionally returns an all-zero
            // output for a GPU inference of an FP16 model: about 1 in 1000, the
            // tensor stays zero, and it never happens on one unit alone. A real
            // raw-head output is never all zero (box coordinates), so in that
            // mode such an output is run once more. (End-to-end outputs of a frame
            // without detections may legitimately be zero; they cost one extra
            // inference then.)
            for attempt in 1...2 {
                var o: InferenceFunction.Outputs
                do {
                    o = try await function.run(inputs: inputs)
                } catch {
                    function = try await rebuiltFunction(after: error)
                    o = try await function.run(inputs: inputs)
                }
                // Take the model's first declared output (deterministic order from
                // the descriptor); fall back to whatever the result holds.
                guard let name = function.descriptor.outputNames.first ?? o.names.first,
                      let value = o.remove(name) else { throw coreAIError(2, "Core AI inference returned no value") }
                guard let arr = value.ndArray else { throw coreAIError(3, "Core AI inference output '\(name)' is not a tensor") }
                // Never write past the caller's buffers. Too many dimensions is an
                // error; too many values is not: the required count is reported
                // with status 2 so the caller can grow its buffer and call again
                // (the output size depends on the model's input size and class
                // count, which the caller cannot know up front).
                guard arr.shape.count <= shapeCapacity else {
                    throw coreAIError(9, "Core AI output has rank \(arr.shape.count); at most \(shapeCapacity) dimensions are supported")
                }
                (valueCount, fits) = try copyDenseFloats(from: arr, into: outputData, capacity: capacity)
                outShape = arr.shape.map { Int64($0) }
                let allZero = fits && valueCount > 0 &&
                    !UnsafeBufferPointer(start: outputData, count: valueCount).contains { $0 != 0 }
                if attempt == 1 && allZero && lanesMixUnits {
                    CoreAINotice.shared.once("Rerunning an all-zero output on the \(unit.label) (a Core AI issue when GPU and Neural Engine run at the same time; set COREAI_COMPUTE_UNIT=gpu to avoid the mix).")
                    continue
                }
                break
            }
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
