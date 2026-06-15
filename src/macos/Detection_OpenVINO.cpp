#include "Detection_OpenVINO.hpp"
#include "YoloPostprocess.hpp"


Detection_OpenVINO::Detection_OpenVINO( float score_threshold,
                                        cv::Size2f model_shape,
                                        const std::string &model_file,
                                        const float nms_threshold ) :
                                        Detection(score_threshold, model_shape, model_file),
                                        nms_threshold_(nms_threshold){
    auto model = core_.read_model(model_file_);
    auto compiled = core_.compile_model(model, "AUTO", ov::hint::enable_hyper_threading(true));
    infer_request_       = compiled.create_infer_request();
    auto input_port = compiled.input();
    input_ = ov::Tensor(  input_port.get_element_type(),
                                input_port.get_shape()
                                );
    plane_ = static_cast<size_t>(model_shape_.width) * static_cast<size_t>(model_shape_.height);
    chans_ = std::vector<cv::Mat>(3);
};

void Detection_OpenVINO::detect( cv::Mat &frame) {
    const int W = static_cast<int>(model_shape_.width);
    const int H = static_cast<int>(model_shape_.height);

    cv::resize(frame, resized_, {W, H});

    // BGR → RGB, uint8 → float32, normalisieren [0,1]
    resized_.convertTo(blob_, CV_32F, 1.0 / 255.0);
    cvtColor(blob_, blob_, cv::COLOR_BGR2RGB);

    // HWC → CHW in Eingabetensor kopieren
    split(blob_, chans_);

    data_  = input_.data<float>();
    for (int c = 0; c < 3; ++c)
        std::memcpy(data_ + c * plane_, chans_[c].ptr<float>(), plane_ * sizeof(float));

    infer_request_.set_input_tensor(input_);

    // ── Inferenz ──────────────────────────────────────────────────────
    infer_request_.infer();

    output_ = infer_request_.get_output_tensor(0);
    if (output_.get_size() == 0) { std::cerr << "Leerer Output!\n"; }

    ov::Shape shape = output_.get_shape();
    if (shape.size() != 3) {
        yolo_postprocess::warnUnsupportedShape("OpenVINO", std::vector<int64_t>(shape.begin(), shape.end()));
        return;
    }

    data_ = output_.data<float>();
    if ((model_format_ == MODEL_FORMAT::YOLO12 || model_format_ == MODEL_FORMAT::YOLOE) &&
        yolo_postprocess::decodeRaw(data_, shape[1], shape[2], frame, model_shape_,
                                    class_list, score_threshold_, nms_threshold_)) {
        return;
    }

    if (yolo_postprocess::decodeEndToEnd(data_, shape[1], shape[2], frame, model_shape_,
                                         class_list, score_threshold_, nms_threshold_)) {
        return;
    }

    yolo_postprocess::warnUnsupportedShape("OpenVINO", std::vector<int64_t>(shape.begin(), shape.end()));
}
