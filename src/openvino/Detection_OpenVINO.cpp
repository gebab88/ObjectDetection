#include "Detection_OpenVINO.hpp"
#include "YoloPostprocess.hpp"

#include <cstring>
#include <stdexcept>
#include <string>


Detection_OpenVINO::Detection_OpenVINO( float score_threshold,
                                        cv::Size2f model_shape,
                                        const std::string &model_file,
                                        const float nms_threshold ) :
                                        Detection(score_threshold, model_shape, model_file),
                                        nms_threshold_(nms_threshold){
    auto model = core_.read_model(model_file_);

    // detect() fills the input with a float32 1x3xHxW blob sized from
    // model_width/model_height in the config. Validate that against the model up
    // front: copying a blob of a different size into the input tensor would read
    // garbage (config smaller) or overflow the tensor (config larger).
    const ov::Shape expected_shape = {1, 3,
                                      static_cast<size_t>(model_shape_.height),
                                      static_cast<size_t>(model_shape_.width)};
    if (model->inputs().size() != 1) {
        throw std::runtime_error("OpenVINO model has " + std::to_string(model->inputs().size()) +
                                 " inputs; exactly one image input is supported.");
    }
    const ov::Output<ov::Node> model_input = model->input();
    if (model_input.get_element_type() != ov::element::f32) {
        throw std::runtime_error("OpenVINO model input expects " + model_input.get_element_type().get_type_name() +
                                 " data; only f32 is supported.");
    }
    const ov::PartialShape input_shape = model_input.get_partial_shape();
    if (input_shape.is_static()) {
        if (input_shape.to_shape() != expected_shape) {
            throw std::runtime_error("OpenVINO model input expects shape " + input_shape.to_string() +
                                     ", but model_width/model_height in config.yaml give " +
                                     ov::PartialShape(expected_shape).to_string() + ".");
        }
    } else {
        // Dynamic export: pin it to the configured size so the compiled model and
        // the pre-allocated input tensor agree with the blob.
        if (!input_shape.compatible(ov::PartialShape(expected_shape))) {
            throw std::runtime_error("OpenVINO model input shape " + input_shape.to_string() +
                                     " is incompatible with model_width/model_height in config.yaml (" +
                                     ov::PartialShape(expected_shape).to_string() + ").");
        }
        model->reshape(ov::PartialShape(expected_shape));
    }

    auto compiled = core_.compile_model(model, "AUTO", ov::hint::enable_hyper_threading(true));
    infer_request_       = compiled.create_infer_request();
    input_ = ov::Tensor(ov::element::f32, expected_shape);
}

void Detection_OpenVINO::detect( cv::Mat &frame) {
    const int W = static_cast<int>(model_shape_.width);
    const int H = static_cast<int>(model_shape_.height);

    // Resize, scale to [0,1], swap BGR->RGB and pack into a 1x3xHxW CHW float blob
    // in a single call, then copy it into the pre-allocated input tensor.
    cv::dnn::blobFromImage(frame, blob_, 1.0 / 255.0, cv::Size(W, H), cv::Scalar(),
                           /*swapRB=*/true, /*crop=*/false);
    // The constructor sized input_ from the same model shape, so this holds unless
    // the blob layout changes; guard anyway rather than overflow the tensor.
    const size_t blob_bytes = blob_.total() * sizeof(float);
    if (blob_bytes != input_.get_byte_size()) {
        yolo_postprocess::warnUnsupportedShape("OpenVINO input", {1, 3, H, W});
        return;
    }
    std::memcpy(input_.data<float>(), blob_.ptr<float>(), blob_bytes);

    infer_request_.set_input_tensor(input_);

    // ── Inference ───────────────────────────────────────────────────────
    infer_request_.infer();

    output_ = infer_request_.get_output_tensor(0);
    if (output_.get_size() == 0) {
        std::cerr << "Empty output tensor!\n";
        return;
    }

    ov::Shape shape = output_.get_shape();
    if (shape.size() != 3) {
        yolo_postprocess::warnUnsupportedShape("OpenVINO", std::vector<int64_t>(shape.begin(), shape.end()));
        return;
    }

    const float* const data = output_.data<float>();
    if (!yolo_postprocess::decodeByFormat(model_format_, data, shape[1], shape[2], frame,
                                          model_shape_, class_list_, score_threshold_, nms_threshold_)) {
        yolo_postprocess::warnUnsupportedShape("OpenVINO", std::vector<int64_t>(shape.begin(), shape.end()));
    }
}
