#include "Detection_ORT.hpp"
#include "YoloPostprocess.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <string>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(__APPLE__) && defined(ENABLE_ORT_COREML)
namespace {

// The CoreML EP only takes nodes whose shapes are fully known
// (RequireStaticInputShapes). Exports with a dynamic batch (and sometimes
// dynamic height/width) therefore leave every node on the CPU, and with one
// intra-op thread a large model runs about ten times slower than on the GPU.
// Pin those free dimensions to what this program feeds anyway: batch 1 and the
// configured model size. The input shapes are read from a lightweight probe
// session, since session options must be complete before the real session is
// created. If the probe fails, nothing is pinned and the real session reports
// the problem.
void pinFreeInputDimensions(Ort::SessionOptions& opts, const Ort::Env& env,
                            const std::string& model_file, cv::Size2f model_shape) {
    std::vector<std::pair<std::string, int64_t>> pins;
    try {
        Ort::SessionOptions probe_opts;
        probe_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
        probe_opts.SetIntraOpNumThreads(1);
        Ort::Session probe(env, model_file.c_str(), probe_opts);
        for (size_t i = 0; i < probe.GetInputCount(); ++i) {
            const Ort::TypeInfo type_info = probe.GetInputTypeInfo(i);
            const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            const std::vector<int64_t> shape = tensor_info.GetShape();
            std::vector<const char*> names(shape.size(), nullptr);
            tensor_info.GetSymbolicDimensions(names.data(), names.size());
            for (size_t d = 0; d < shape.size(); ++d) {
                if (shape[d] >= 0 || names[d] == nullptr || *names[d] == '\0') continue;
                int64_t value = 0;
                if (d == 0) {
                    value = 1;  // batch
                } else if (shape.size() == 4 && d == 2) {
                    value = static_cast<int64_t>(model_shape.height);
                } else if (shape.size() == 4 && d == 3) {
                    value = static_cast<int64_t>(model_shape.width);
                }
                if (value > 0) pins.emplace_back(names[d], value);
            }
        }
    } catch (const Ort::Exception&) {
        return;
    }
    for (const auto& [name, value] : pins) {
        opts.AddFreeDimensionOverrideByName(name.c_str(), value);
        std::cout << "[Provider] Pinned dynamic input dimension '" << name << "' to " << value
                  << " for CoreML." << std::endl;
    }
}

}  // namespace
#endif

Ort::SessionOptions Detection_ORT::make_session_opts(const Ort::Env& env,
                                                     const std::string& model_file,
                                                     cv::Size2f model_shape,
                                                     bool& coreml_enabled) {
    Ort::SessionOptions opts;
    coreml_enabled = false;
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    opts.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

#if defined(__APPLE__) && defined(ENABLE_ORT_COREML)
    pinFreeInputDimensions(opts, env, model_file, model_shape);

    std::unordered_map<std::string, std::string> coreml_options;
    coreml_options[kCoremlProviderOption_ModelFormat] = "MLProgram";
    coreml_options[kCoremlProviderOption_RequireStaticInputShapes] = "1";
    coreml_options[kCoremlProviderOption_EnableOnSubgraphs] = "0";
    coreml_options[kCoremlProviderOption_AllowLowPrecisionAccumulationOnGPU] = "1";
    if (const char* compute_units = std::getenv("ORT_COREML_COMPUTE_UNITS")) {
        if (*compute_units != '\0') {
            coreml_options[kCoremlProviderOption_MLComputeUnits] = compute_units;
        }
    }
    // Persist compiled CoreML artifacts next to the model (not in the current
    // working directory, which scattered caches across build/, the repo root,
    // etc.), so every run of the same model reuses one cache wherever it starts.
    // The CoreML EP still uses the system temp dir for intermediate compilation
    // files. If the directory cannot be created (e.g. read-only model location),
    // run without a persistent cache instead of failing.
    std::error_code cache_ec;
    const std::filesystem::path model_dir =
        std::filesystem::absolute(std::filesystem::path(model_file), cache_ec).parent_path();
    const std::string cache_dir = (model_dir / ".ort_coreml_cache").string();
    if (!cache_ec) {
        std::filesystem::create_directories(cache_dir, cache_ec);
    }
    if (cache_ec) {
        std::cerr << "[Warning] CoreML model cache disabled: cannot create " << cache_dir
                  << " (" << cache_ec.message() << ")." << std::endl;
    } else {
        coreml_options[kCoremlProviderOption_ModelCacheDirectory] = cache_dir;
    }

    try {
        opts.AppendExecutionProvider("CoreML", coreml_options);
        // CoreML runs the graph, so a single intra-op thread suffices; the
        // CPU-side settings below stay untouched unless the provider fails.
        opts.SetIntraOpNumThreads(1);
        coreml_enabled = true;
        std::cout << "[Provider] CoreML EP enabled; compute units: "
                  << (coreml_options.count(kCoremlProviderOption_MLComputeUnits)
                          ? coreml_options[kCoremlProviderOption_MLComputeUnits]
                          : "CoreML default/all available units")
                  << "." << std::endl;
        if (!cache_ec) {
            std::cout << "[Provider] CoreML model cache: " << cache_dir << std::endl;
        }
    } catch (const Ort::Exception& e) {
        std::cerr << "[Warning] CoreML Execution Provider could not be enabled: "
                  << e.what() << ". Falling back to CPU." << std::endl;
    }
#elif defined(__linux__) && defined(__aarch64__)
    // Linux/ARM: enable XNNPACK, which automatically uses available CPU features.
    opts.AppendExecutionProvider("XNNPACK");
#endif

    if (!coreml_enabled) {
        // CPU path, including the fallback when CoreML could not be enabled: use
        // all cores. (No SetOptimizedModelFilePath: it only *writes* the optimized
        // graph for offline use; ONNX Runtime never loads it back, so it cost time
        // on every start and left <model>.optimized.onnx in the working directory.)
        opts.SetIntraOpNumThreads(std::thread::hardware_concurrency());
    }

    return opts;
}

Detection_ORT::Detection_ORT(float score_threshold,
                            cv::Size2f model_shape,
                            const std::string &model_file,
                            const float nms_threshold) :

                            Detection(score_threshold, model_shape, model_file),
                            env_(Ort::Env(ORT_LOGGING_LEVEL_ERROR, "yolo")),
                            session_opts_(make_session_opts(env_, model_file, model_shape, coreml_enabled_)),
                            allocator_(),
                            session_(env_, model_file_.c_str(), session_opts_),
                            input_name_(session_.GetInputNameAllocated(0, allocator_).get()),
                            nms_threshold_(nms_threshold),
                            memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
                            slots_(coreml_enabled_ ? 2 : 1, "[ONNX Runtime]") {

    const size_t output_count = session_.GetOutputCount();
    output_names_.reserve(output_count);
    for (size_t i = 0; i < output_count; ++i) {
        output_names_.emplace_back(session_.GetOutputNameAllocated(i, allocator_).get());
    }

    // Cache the C-string views once; output_names_ is never modified afterwards,
    // so these pointers stay valid and Run() no longer rebuilds them per frame.
    output_name_ptrs_.reserve(output_names_.size());
    for (const auto& output_name : output_names_) {
        output_name_ptrs_.push_back(output_name.c_str());
    }

    std::cout << "[YoloDetector] Model loaded: " << model_file_ << "\n";
    auto providers = Ort::GetAvailableProviders();
    std::cout << "[Provider] Execution providers available in this ONNX Runtime:\n";
    for (const auto& p : providers)
        std::cout << "  - " << p << "\n";
}

void Detection_ORT::detect(cv::Mat &frame) {
    run(blob_, frame);
}

std::future<void> Detection_ORT::detect_async(cv::Mat frame) {
    if (slots_.size() <= 1) {
        return Detection::detect_async(frame);  // synchronous, on the caller's thread
    }
    return slots_.submit(frame, [this](Scratch &slot, cv::Mat &f) { run(slot.blob, f); });
}

void Detection_ORT::run(cv::Mat &blob, cv::Mat &frame) {
    const int W = static_cast<int>(model_shape_.width);
    const int H = static_cast<int>(model_shape_.height);

    // Resize, scale to [0,1], swap BGR->RGB and pack into a 1x3xHxW CHW float blob
    // in a single call; the ORT input tensor wraps the blob's buffer directly.
    cv::dnn::blobFromImage(frame, blob, 1.0 / 255.0, cv::Size(W, H), cv::Scalar(),
                           /*swapRB=*/true, /*crop=*/false);

    std::array<int64_t, 4> input_shape = {1, 3, static_cast<int64_t>(H), static_cast<int64_t>(W)};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info_,
                                                            blob.ptr<float>(), blob.total(),
                                                            input_shape.data(), input_shape.size() );
    const char* in_names[]  = { input_name_.c_str()  };

    auto outputs = session_.Run( Ort::RunOptions{nullptr}, in_names,  &input_tensor, 1,
                                 output_name_ptrs_.data(), output_name_ptrs_.size());
    auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();

    // NMS-free YOLOE exports with two outputs: [1, numDets, numClasses] class
    // logits plus a separate [1, numDets, 4] box tensor. This layout is decoded
    // here rather than via decodeByFormat, which handles the single-output cases.
    if (outputs.size() >= 2) {
        auto boxes_shape = outputs[1].GetTensorTypeAndShapeInfo().GetShape();
        if (shape.size() == 3 && boxes_shape.size() == 3 &&
            shape[0] == 1 && boxes_shape[0] == 1 &&
            shape[1] == boxes_shape[1] && boxes_shape[2] == 4) {

            const int numDets = static_cast<int>(shape[1]);
            const int numClasses = static_cast<int>(std::min<int64_t>(shape[2], class_list_.size()));
            const float* logits = outputs[0].GetTensorData<float>();
            const float* boxes = outputs[1].GetTensorData<float>();
            const cv::Size frame_size(frame.cols, frame.rows);
            std::vector<float> scores;
            std::vector<int> class_ids;
            std::vector<cv::Rect> detections;
            scores.reserve(numDets);
            class_ids.reserve(numDets);
            detections.reserve(numDets);

            for (int i = 0; i < numDets; ++i) {
                float score = -std::numeric_limits<float>::infinity();
                int class_id = -1;
                for (int c = 0; c < numClasses; ++c) {
                    const float class_score = yolo_postprocess::confidence(logits[i * shape[2] + c]);
                    if (class_score > score) {
                        score = class_score;
                        class_id = c;
                    }
                }

                if (score < score_threshold_ || class_id < 0) continue;

                // This export emits boxes as cxcywh normalised to [0,1], so scale
                // by the frame size (unlike the single-output path, which decodes
                // model-space coordinates and rescales by frame/model in decode*).
                const float cx = boxes[i * 4] * frame.cols;
                const float cy = boxes[i * 4 + 1] * frame.rows;
                const float w = boxes[i * 4 + 2] * frame.cols;
                const float h = boxes[i * 4 + 3] * frame.rows;
                cv::Rect box = yolo_postprocess::clippedRect(cx - 0.5f * w, cy - 0.5f * h,
                                                             cx + 0.5f * w, cy + 0.5f * h,
                                                             frame_size);
                if (box.empty()) continue;

                scores.push_back(score);
                class_ids.push_back(class_id);
                detections.push_back(box);
            }
            yolo_postprocess::drawNmsDetections(frame, detections, class_ids, scores, class_list_,
                                                score_threshold_, nms_threshold_);
            return;
        }
    }

    if (shape.size() != 3) {
        yolo_postprocess::warnUnsupportedShape("ONNX Runtime", shape);
        return;
    }

    const float* const data = outputs[0].GetTensorData<float>();
    if (!yolo_postprocess::decodeByFormat(model_format_, data, shape[1], shape[2], frame,
                                          model_shape_, class_list_, score_threshold_, nms_threshold_)) {
        yolo_postprocess::warnUnsupportedShape("ONNX Runtime", shape);
    }
}
