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
    // The backend contract is a single-output rank-3 tensor (batch, fields/detections,
    // values) — the same exact-rank requirement the ONNX Runtime backend enforces — so
    // shape[1] and shape[2] can be indexed as fields/detections without ambiguity.
    if (values.empty() || shape.size() != 3) {
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

// Placeholder implementation of the Swift Core AI interop entry point. The real
// implementation lives in the Swift interop layer (AIModel / InferenceFunction /
// InferenceValue) and is linked into the final app; until that layer is integrated
// through CMake/Swift, fail closed so callers fall back to the ONNX Runtime backend.
extern "C" int coreAIInfer(const float* input_data,
                           int64_t input_numel,
                           const int64_t* input_shape,
                           int64_t input_rank,
                           CoreAITensor* output) {
    (void)input_data;
    (void)input_numel;
    (void)input_shape;
    (void)input_rank;
    if (output != nullptr) {
        output->data = nullptr;
        output->numel = 0;
        output->shape.clear();
    }
    return -1;
}
