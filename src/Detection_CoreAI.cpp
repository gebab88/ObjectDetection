#include "Detection_CoreAI.hpp"
#include "YoloPostprocess.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <vector>

Detection_CoreAI::Detection_CoreAI(  const float score_threshold,
                                    const cv::Size2f model_shape,
                                    const std::string &model_file,
                                    const float nms_threshold) :

                                    Detection(score_threshold, model_shape, model_file),
                                    nms_threshold_(nms_threshold) {
}

void Detection_CoreAI::detect( cv::Mat &frame) {
    const int W = static_cast<int>(model_shape_.width);
    const int H = static_cast<int>(model_shape_.height);

    // Resize, scale to [0,1], swap BGR->RGB and pack into a 1x3xHxW CHW float blob
    // in a single call. The Swift interop layer owns the AIModel load/specialisation
    // and runs inference on this input, reporting the raw output tensor.
    cv::dnn::blobFromImage(frame, blob_, 1.0 / 255.0, cv::Size(W, H), cv::Scalar(),
                           /*swapRB=*/true, /*crop=*/false);

    std::array<int64_t, 4> input_shape = {1, 3, static_cast<int64_t>(H), static_cast<int64_t>(W)};
    CoreAITensor output;
    const int status = coreAIInfer(blob_.ptr<float>(), static_cast<int64_t>(blob_.total()),
                                   input_shape.data(), static_cast<int64_t>(input_shape.size()),
                                   &output);
    if (status != 0 || output.data == nullptr) {
        yolo_postprocess::warnUnsupportedShape("CoreAI", output.shape);
        return;
    }

    std::vector<float> values(output.data, output.data + output.numel);
    std::free(output.data);
    output.data = nullptr;

    const std::vector<int64_t> shape = output.shape;
    if (values.empty() || shape.size() < 2) {
        yolo_postprocess::warnUnsupportedShape("CoreAI", shape);
        return;
    }

    const float* const data = values.data();
    const int64_t dim1 = shape[1];
    const int64_t dim2 = shape[2];

    // The Core AI backend reuses the shared C++ decoders, so decoding, NMS and
    // drawing stay in C++ while inference runs in Swift.
    if (!yolo_postprocess::decodeByFormat(model_format_, data, dim1, dim2, frame, model_shape_,
                                          class_list_, score_threshold_, nms_threshold_)) {
        yolo_postprocess::warnUnsupportedShape("CoreAI", {dim1, dim2});
    }
}
