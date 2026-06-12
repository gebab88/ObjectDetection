#include "Detection_ORT.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <unordered_map>

namespace {
float sigmoid(float value) {
    return 1.0f / (1.0f + std::exp(-value));
}

float confidence(float value) {
    if (value >= 0.0f && value <= 1.0f) {
        return value;
    }
    return sigmoid(value);
}

cv::Rect clippedRect(float left, float top, float right, float bottom, const cv::Size& frame_size) {
    const int x1 = std::clamp(static_cast<int>(std::round(left)), 0, frame_size.width - 1);
    const int y1 = std::clamp(static_cast<int>(std::round(top)), 0, frame_size.height - 1);
    const int x2 = std::clamp(static_cast<int>(std::round(right)), 0, frame_size.width - 1);
    const int y2 = std::clamp(static_cast<int>(std::round(bottom)), 0, frame_size.height - 1);
    if (x2 <= x1 || y2 <= y1) {
        return {};
    }
    return cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2));
}
}

Ort::SessionOptions Detection_ORT::make_session_opts() {
    Ort::SessionOptions opts;
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    opts.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

    #if !(defined(__APPLE__) && defined(ENABLE_ORT_COREML))
        opts.SetIntraOpNumThreads(std::thread::hardware_concurrency());
        opts.SetOptimizedModelFilePath("model_optimized.onnx");
    #endif
    
    #if defined(__APPLE__) && defined(ENABLE_ORT_COREML)
        opts.SetIntraOpNumThreads(1);

        std::unordered_map<std::string, std::string> coreml_options;
        coreml_options[kCoremlProviderOption_ModelFormat] = "MLProgram";
        coreml_options[kCoremlProviderOption_MLComputeUnits] = "ALL";
        coreml_options[kCoremlProviderOption_RequireStaticInputShapes] = "1";
        coreml_options[kCoremlProviderOption_EnableOnSubgraphs] = "0";
        coreml_options[kCoremlProviderOption_AllowLowPrecisionAccumulationOnGPU] = "1";
        const std::string cache_dir = std::filesystem::absolute(".ort_coreml_cache").string();
        std::filesystem::create_directories(cache_dir);
        setenv("TMPDIR", cache_dir.c_str(), 1);
        coreml_options[kCoremlProviderOption_ModelCacheDirectory] = cache_dir;

        try {
            opts.AppendExecutionProvider("CoreML", coreml_options);
            std::cout << "[Provider] CoreML EP enabled with MLComputeUnits=ALL (GPU + Neural Engine)." << std::endl;
        } catch (const Ort::Exception& e) {
            std::cerr << "[Warning] CoreML Execution Provider could not be enabled: "
                      << e.what() << ". Falling back to CPU." << std::endl;
        }
    #elif defined(__linux__) && defined(__aarch64__)
        // Linux-spezifischer Code
        // XNNPACK aktivieren → nutzt automatisch verfügbare CPU-Features
        opts.AppendExecutionProvider("XNNPACK");
    #endif

    return opts;
}

Detection_ORT::Detection_ORT(float score_threshold,
                            cv::Size2f model_shape,
                            const std::string &model_file) :

                            Detection(score_threshold, model_shape, model_file),
                            env_(Ort::Env(ORT_LOGGING_LEVEL_ERROR, "yolo")),
                            session_opts_(make_session_opts()),
                            allocator_(),
                            session_(env_, model_file_.c_str(), session_opts_),
                            input_name_(session_.GetInputNameAllocated(0, allocator_).get()) {

    plane_ = (size_t)model_shape_.width * model_shape_.height;
    input_ = std::vector<float>(3 * plane_);
    chans_ = std::vector<cv::Mat>(3);

    const size_t output_count = session_.GetOutputCount();
    output_names_.reserve(output_count);
    for (size_t i = 0; i < output_count; ++i) {
        output_names_.emplace_back(session_.GetOutputNameAllocated(i, allocator_).get());
    }

    std::cout << "[YoloDetector] Modell geladen: " << model_file_ << "\n";
    auto providers = Ort::GetAvailableProviders();
    std::cout << "[Provider] Verfuegbare Execution Provider in dieser ONNX Runtime:\n";
    for (const auto& p : providers)
        std::cout << "  - " << p << "\n";
}

void Detection_ORT::detect( cv::Mat &frame) {
    // Skalierungsfaktoren berechnen, um Boxen auf Originalbildgröße zu ziehen
    const float x_scale = (float)frame.cols / model_shape_.width;
    const float y_scale = (float)frame.rows / model_shape_.height;

    const int W = static_cast<int>(model_shape_.width);
    const int H = static_cast<int>(model_shape_.height);

    cv::resize(frame, resized_, {W, H});

    // BGR → RGB, uint8 → float32, normalisieren [0,1]
    resized_.convertTo(blob_, CV_32F, 1.0 / 255.0);
    cvtColor(blob_, blob_, cv::COLOR_BGR2RGB);

    // HWC → CHW in Eingabetensor kopieren
    split(blob_, chans_);

    // data_  = input_tensor.data<float>();
    for (int c = 0; c < 3; ++c)
        std::memcpy(input_.data() + c * plane_, chans_[c].ptr<float>(), plane_ * sizeof(float));

    std::array<int64_t, 4> input_shape = {1, 3, static_cast<int64_t>(H), static_cast<int64_t>(W)};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info,
                                                            input_.data(), input_.size(),
                                                            input_shape.data(), input_shape.size() );
    const char* in_names[]  = { input_name_.c_str()  };
    std::vector<const char*> out_names;
    out_names.reserve(output_names_.size());
    for (const auto& output_name : output_names_) {
        out_names.push_back(output_name.c_str());
    }

    auto outputs = session_.Run( Ort::RunOptions{nullptr}, in_names,  &input_tensor, 1,
                                 out_names.data(), out_names.size());
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

            for (int i = 0; i < numDets; ++i) {
                float score = -std::numeric_limits<float>::infinity();
                int class_id = -1;
                for (int c = 0; c < numClasses; ++c) {
                    const float class_score = confidence(logits[i * shape[2] + c]);
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
                cv::Rect box = clippedRect(cx - 0.5f * w, cy - 0.5f * h,
                                           cx + 0.5f * w, cy + 0.5f * h,
                                           frame_size);
                if (box.empty()) continue;

                rectangle(frame, box, cv::Scalar(0,255,255), 1);
                std::string label = class_list[class_id] + "  " + std::to_string(int(100 * score)) + "%" ;
                std::cout << label << '\n';
                putText(frame, label, cv::Point(box.x, std::max(0, box.y - 5)),
                        cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
            }
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

    data_ = outputs[0].GetTensorMutableData<float>();

    float x1,x2,y1,y2,score;
    int class_id;
    cv::Rect box;

    const int numDets = shape[2] == 6 ? static_cast<int>(shape[1]) :
                        shape[1] == 6 ? static_cast<int>(shape[2]) : 0;

    if (numDets == 0) {
        static bool warned = false;
        if (!warned) {
            std::cerr << "Unsupported single-output detection shape: [";
            for (size_t i = 0; i < shape.size(); ++i) {
                std::cerr << shape[i] << (i + 1 < shape.size() ? ", " : "");
            }
            std::cerr << "]" << std::endl;
            warned = true;
        }
        return;
    }

    for (int i = 0; i < numDets; ++i) {
        if (shape[2] == 6) {
            data_ = outputs[0].GetTensorMutableData<float>() + i * 6;
            x1 = data_[0];
            y1 = data_[1];
            x2 = data_[2];
            y2 = data_[3];
            score = data_[4];
            class_id = (int)data_[5];
        } else {
            const float* data = outputs[0].GetTensorData<float>();
            x1 = data[0 * numDets + i];
            y1 = data[1 * numDets + i];
            x2 = data[2 * numDets + i];
            y2 = data[3 * numDets + i];
            score = data[4 * numDets + i];
            class_id = (int)data[5 * numDets + i];
        }

        // Frühzeitiger Abbruch – häufigster Pfad zuerst
        if (score < score_threshold_) continue;

        // Koordinaten extrahieren und auf Originalgröße skalieren
        x1 *= x_scale;
        y1 *= y_scale;
        x2 *= x_scale;
        y2 *= y_scale;
        if (class_id < 0 || class_id >= (int)class_list.size()) continue;

        // Ungültige Boxen (Breite/Höhe <= 0) direkt verwerfen
        if (x2 <= x1 || y2 <= y1) continue;

        box = cv::Rect(int(x1), int(y1), int(x2-x1), int(y2-y1));

        rectangle(frame, box, cv::Scalar(0,255,255), 1);
        std::string label = class_list[ class_id ] + "  " + std::to_string(int(100 * score)) + "%" ;
        std::cout << label << '\n';
        putText(frame, label, cv::Point( box.x, box.y ), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
    }
}
