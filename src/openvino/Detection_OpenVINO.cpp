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
};

void Detection_OpenVINO::detect( cv::Mat &frame) {
    const int W = static_cast<int>(model_shape_.width);
    const int H = static_cast<int>(model_shape_.height);

    // Resize, scale to [0,1], swap BGR->RGB and pack into a 1x3xHxW CHW float blob
    // in a single call, then copy it into the pre-allocated input tensor.
    cv::dnn::blobFromImage(frame, blob_, 1.0 / 255.0, cv::Size(W, H), cv::Scalar(),
                           /*swapRB=*/true, /*crop=*/false);
    std::memcpy(input_.data<float>(), blob_.ptr<float>(), blob_.total() * sizeof(float));

    infer_request_.set_input_tensor(input_);

    // ── Inference ───────────────────────────────────────────────────────
    infer_request_.infer();

    output_ = infer_request_.get_output_tensor(0);
    if (output_.get_size() == 0) { std::cerr << "Empty output tensor!\n"; }

    ov::Shape shape = output_.get_shape();
    if (shape.size() != 3) {
        yolo_postprocess::warnUnsupportedShape("OpenVINO", std::vector<int64_t>(shape.begin(), shape.end()));
        return;
    }

    const float* const data = output_.data<float>();
    if (!yolo_postprocess::decodeByFormat(model_format_, data, shape[1], shape[2], frame,
                                          model_shape_, class_list, score_threshold_, nms_threshold_)) {
        yolo_postprocess::warnUnsupportedShape("OpenVINO", std::vector<int64_t>(shape.begin(), shape.end()));
    }
}
