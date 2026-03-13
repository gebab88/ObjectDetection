#include "Detection_ORT.hpp"

Ort::SessionOptions Detection_ORT::make_session_opts() {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(4);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    return opts;
}

Detection_ORT::Detection_ORT(float score_threshold,
                            cv::Size2f model_shape,
                            const std::string &model_file) :
                            Detection(score_threshold, model_shape, model_file),
                            env_(Ort::Env(ORT_LOGGING_LEVEL_VERBOSE, "yolo")),
                            session_opts_(make_session_opts()),
                            session_(env_, model_file_.c_str(), session_opts_),
                            input_name_(session_.GetInputNameAllocated(0, allocator_).get()),
                            output_name_(session_.GetOutputNameAllocated(0, allocator_).get()) {
    #ifdef __APPLE__
        // macOS-spezifischer Code
        // CoreML aktivieren → nutzt automatisch AMD Radeon via Metal
        uint32_t coreml_flags = 0;
        // Optional: nur GPU erzwingen:
        // coreml_flags |= COREML_FLAG_USE_CPU_ONLY;  // zum Deaktivieren
        OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CoreML(session_opts_, coreml_flags);
        if (status != nullptr) {
            std::cerr << "[Warning] CoreML nicht verfügbar: "
                  << Ort::GetApi().GetErrorMessage(status) << "\n";
            Ort::GetApi().ReleaseStatus(status);
        }
    #elif __aarch64__
        // Linux-spezifischer Code
        // XNNPACK aktivieren → nutzt automatisch verfügbare CPU-Features
        session_opts_.AppendExecutionProvider("XNNPACK");
    #endif

    allocator_ = Ort::AllocatorWithDefaultOptions();

    plane_ = 640 * 640;
    input_ = std::vector<float>(3 * 640 * 640);
    chans_ = std::vector<cv::Mat>(3);

    std::cout << "[YoloDetector] Modell geladen: " << model_file_ << "\n";
    auto providers = Ort::GetAvailableProviders();
    std::cout << "[Provider] Aktive Execution Provider:\n";
    for (const auto& p : providers)
        std::cout << "  - " << p << "\n";
}

void Detection_ORT::detect( cv::Mat &frame) {
    // Skalierungsfaktoren berechnen, um Boxen auf Originalbildgröße zu ziehen
    const float x_scale = (float)frame.cols / model_shape_.width;
    const float y_scale = (float)frame.rows / model_shape_.height;

    cv::resize(frame, resized_, {640, 640});

    // BGR → RGB, uint8 → float32, normalisieren [0,1]
    resized_.convertTo(blob_, CV_32F, 1.0 / 255.0);
    cvtColor(blob_, blob_, cv::COLOR_BGR2RGB);

    // HWC → CHW in Eingabetensor kopieren
    split(blob_, chans_);

    // data_  = input_tensor.data<float>();
    for (int c = 0; c < 3; ++c)
        std::memcpy(input_.data() + c * plane_, chans_[c].ptr<float>(), plane_ * sizeof(float));

    std::array<int64_t, 4> input_shape = {1, 3, 640, 640};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info,
                                                            input_.data(), input_.size(),
                                                            input_shape.data(), input_shape.size() );
    const char* in_names[]  = { input_name_.c_str()  };
    const char* out_names[] = { output_name_.c_str() };

    auto outputs = session_.Run( Ort::RunOptions{nullptr}, in_names,  &input_tensor, 1,
                                 out_names, 1);
    data_ = outputs[0].GetTensorMutableData<float>();
    auto   shape    = outputs[0].GetTensorTypeAndShapeInfo().GetShape();

    const int numDets = shape[1];
    const int numFields = shape[2];

    float x1,x2,y1,y2,score;
    int class_id;
    cv::Rect box;

    const float* end = data_ + numDets * numFields;

    for ( ; data_ != end; data_ += numFields) {
        score = data_[4];

        // Frühzeitiger Abbruch – häufigster Pfad zuerst
        if (score < score_threshold_) continue;

        // Koordinaten extrahieren und auf Originalgröße skalieren
        x1 = data_[0] * x_scale;
        y1 = data_[1] * y_scale;
        x2 = data_[2] * x_scale;
        y2 = data_[3] * y_scale;
        class_id = (int)data_[5];
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