#include "Detection_ORT.hpp"
#include "YoloPostprocess.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <string>
#include <unordered_map>

Ort::SessionOptions Detection_ORT::make_session_opts(const std::string& model_file) {
    Ort::SessionOptions opts;
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    opts.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

    (void)model_file;
#if defined(__APPLE__) && defined(ENABLE_ORT_COREML)
    const bool use_coreml = true;
#else
    const bool use_coreml = false;
#endif

    if (!use_coreml) {
        opts.SetIntraOpNumThreads(std::thread::hardware_concurrency());
        // Derive the cache name from the model so switching models does not make
        // ORT load a stale optimized graph produced for a different network.
        const std::string optimized_path =
            std::filesystem::path(model_file).filename().string() + ".optimized.onnx";
        opts.SetOptimizedModelFilePath(optimized_path.c_str());
    }
    
    #if defined(__APPLE__) && defined(ENABLE_ORT_COREML)
    if (use_coreml) {
        opts.SetIntraOpNumThreads(1);

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
        const std::string cache_dir = std::filesystem::absolute(".ort_coreml_cache").string();
        std::filesystem::create_directories(cache_dir);
        // Deliberate process-wide side effect: the CoreML EP compiles models into
        // TMPDIR before moving them into the model cache. Pointing TMPDIR at the
        // project-local cache keeps those (large) intermediates out of the system
        // temp dir and next to the persisted cache.
        setenv("TMPDIR", cache_dir.c_str(), 1);
        coreml_options[kCoremlProviderOption_ModelCacheDirectory] = cache_dir;

        try {
            opts.AppendExecutionProvider("CoreML", coreml_options);
            std::cout << "[Provider] CoreML EP enabled; compute units: "
                      << (coreml_options.count(kCoremlProviderOption_MLComputeUnits)
                              ? coreml_options[kCoremlProviderOption_MLComputeUnits]
                              : "CoreML default/all available units")
                      << "." << std::endl;
            std::cout << "[Provider] CoreML model cache: " << cache_dir << std::endl;
        } catch (const Ort::Exception& e) {
            std::cerr << "[Warning] CoreML Execution Provider could not be enabled: "
                      << e.what() << ". Falling back to CPU." << std::endl;
        }
    }
    #elif defined(__linux__) && defined(__aarch64__)
        // Linux/ARM: enable XNNPACK, which automatically uses available CPU features.
        opts.AppendExecutionProvider("XNNPACK");
    #endif

    return opts;
}

Detection_ORT::Detection_ORT(float score_threshold,
                            cv::Size2f model_shape,
                            const std::string &model_file,
                            const float nms_threshold) :

                            Detection(score_threshold, model_shape, model_file),
                            env_(Ort::Env(ORT_LOGGING_LEVEL_ERROR, "yolo")),
                            session_opts_(make_session_opts(model_file)),
                            allocator_(),
                            session_(env_, model_file_.c_str(), session_opts_),
                            input_name_(session_.GetInputNameAllocated(0, allocator_).get()),
                            nms_threshold_(nms_threshold) {

    plane_ = (size_t)model_shape_.width * model_shape_.height;
    input_ = std::vector<float>(3 * plane_);
    chans_ = std::vector<cv::Mat>(3);

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

void Detection_ORT::detect( cv::Mat &frame) {
    const int W = static_cast<int>(model_shape_.width);
    const int H = static_cast<int>(model_shape_.height);

    cv::resize(frame, resized_, {W, H});

    // BGR -> RGB, uint8 -> float32, normalise to [0,1]
    resized_.convertTo(blob_, CV_32F, 1.0 / 255.0);
    cvtColor(blob_, blob_, cv::COLOR_BGR2RGB);

    // HWC -> CHW: copy each channel plane into the input tensor
    split(blob_, chans_);

    for (int c = 0; c < 3; ++c)
        std::memcpy(input_.data() + c * plane_, chans_[c].ptr<float>(), plane_ * sizeof(float));

    std::array<int64_t, 4> input_shape = {1, 3, static_cast<int64_t>(H), static_cast<int64_t>(W)};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info,
                                                            input_.data(), input_.size(),
                                                            input_shape.data(), input_shape.size() );
    const char* in_names[]  = { input_name_.c_str()  };

    auto outputs = session_.Run( Ort::RunOptions{nullptr}, in_names,  &input_tensor, 1,
                                 output_name_ptrs_.data(), output_name_ptrs_.size());
    auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();

    if (outputs.size() >= 2) {
        auto boxes_shape = outputs[1].GetTensorTypeAndShapeInfo().GetShape();
        if (shape.size() == 3 && boxes_shape.size() == 3 &&
            shape[0] == 1 && boxes_shape[0] == 1 &&
            shape[1] == boxes_shape[1] && boxes_shape[2] == 4) {

            const int numDets = static_cast<int>(shape[1]);
            const int numClasses = static_cast<int>(std::min<int64_t>(shape[2], class_list.size()));
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
            yolo_postprocess::drawNmsDetections(frame, detections, class_ids, scores, class_list,
                                                score_threshold_, nms_threshold_);
            return;
        }
    }

    if (shape.size() != 3) {
        static bool warned = false;
        if (!warned) {
            std::cerr << "Unsupported ONNX output shape for detection." << std::endl;
            warned = true;
        }
        return;
    }

    const float* const data = outputs[0].GetTensorData<float>();

    // Pick the decoder from the known model format: raw (NMS-free) for YOLO12,
    // end-to-end for YOLO26, and raw-then-end-to-end for the YOLOE export, whose
    // ONNX may be either depending on the --nms export flag.
    switch (model_format_) {
        case MODEL_FORMAT::YOLO12:
            if (yolo_postprocess::decodeRaw(data, shape[1], shape[2], frame, model_shape_,
                                            class_list, score_threshold_, nms_threshold_)) {
                return;
            }
            break;
        case MODEL_FORMAT::YOLO26:
            if (yolo_postprocess::decodeEndToEnd(data, shape[1], shape[2], frame, model_shape_,
                                                 class_list, score_threshold_, nms_threshold_)) {
                return;
            }
            break;
        case MODEL_FORMAT::YOLOE:
            if (yolo_postprocess::decodeRaw(data, shape[1], shape[2], frame, model_shape_,
                                            class_list, score_threshold_, nms_threshold_) ||
                yolo_postprocess::decodeEndToEnd(data, shape[1], shape[2], frame, model_shape_,
                                                 class_list, score_threshold_, nms_threshold_)) {
                return;
            }
            break;
        default:
            break;
    }

    yolo_postprocess::warnUnsupportedShape("ONNX Runtime", shape);
}
