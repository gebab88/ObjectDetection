#include "Detection_CoreAI.hpp"
#include "YoloPostprocess.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <vector>

Detection_CoreAI::Detection_CoreAI(  const float score_threshold,
                                    const cv::Size2f model_shape,
                                    const std::string &model_file,
                                    const float nms_threshold) :

                                    Detection(score_threshold, model_shape, model_file),
                                    nms_threshold_(nms_threshold) {
    // The Swift Core AI interop layer (AIModel / InferenceFunction / InferenceValue)
    // provides the `coreAIInfer` C entry point and is compiled and linked into this
    // binary when ENABLE_CORE_AI=ON on Apple Silicon (see the CMake option). The
    // entry point fails closed (returns nonzero) for invalid input, and an inference
    // failure surfaces through the usual unsupported-shape path in `detect`; there is
    // no C++ placeholder entry point any more, so no initialisation probe is needed
    // here.
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

    // The Swift interop layer reports a single-output raw tensor. A raw YOLO26 head
    // output is (1, 84, 8400) values and an end-to-end (built-in NMS) output is
    // (1, N, 6); both fit into the raw-capacity buffer below (the backend is YOLO26
    // family), and the Swift layer reports the actual number of valid values through
    // *output_numel, so an oversized buffer is safe.
    std::vector<float> output_buffer(8400 * 84, 0.0f);
    int64_t output_numel = 0;
    std::array<int64_t, 3> output_shape = {};
    int64_t output_shape_capacity = static_cast<int64_t>(output_shape.size());
    const int status = coreAIInfer(blob_.ptr<float>(), static_cast<int64_t>(blob_.total()),
                                   input_shape.data(), static_cast<int64_t>(input_shape.size()),
                                   model_file_.c_str(), output_buffer.data(),
                                   static_cast<int64_t>(output_buffer.size()),
                                   &output_numel, output_shape.data(), output_shape_capacity);
    if (status != 0 || output_numel <= 0) {
        yolo_postprocess::warnUnsupportedShape("CoreAI", {});
        return;
    }

    CoreAITensor output;
    output.data = output_buffer.data();
    output.numel = output_numel;
    output.shape = {output_shape[0], output_shape[1], output_shape[2]};

    const float* const data = output.data;
    const int64_t dim1 = output.shape[1];
    const int64_t dim2 = output.shape[2];

    // The Core AI backend reuses the shared C++ decoders, so decoding, NMS and
    // drawing stay in C++ while inference runs in Swift.
    if (!yolo_postprocess::decodeByFormat(model_format_, data, dim1, dim2, frame, model_shape_,
                                          class_list_, score_threshold_, nms_threshold_)) {
        yolo_postprocess::warnUnsupportedShape("CoreAI", {dim1, dim2});
    }
}
