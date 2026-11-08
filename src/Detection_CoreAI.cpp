#include "Detection_CoreAI.hpp"
#include "YoloPostprocess.hpp"

#include <algorithm>
#include <array>
#include <exception>
#include <stdexcept>
#include <string>
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
    for (std::size_t i = 0; i < kFramesInFlight; ++i) {
        slots_.push_back(std::make_unique<Slot>());
    }
}

void Detection_CoreAI::detect(cv::Mat &frame) {
    detect_async(frame).get();
}

std::future<void> Detection_CoreAI::detect_async(cv::Mat frame) {
    Slot &slot = *slots_[next_slot_];
    if (slot.busy.exchange(true)) {
        throw std::logic_error("[CoreAI] more than max_frames_in_flight() frames in flight");
    }
    next_slot_ = (next_slot_ + 1) % slots_.size();

    // The whole frame (preprocessing, inference, decoding, drawing) runs on a
    // worker thread. The slot is released before the caller's future becomes
    // ready, so the caller may reuse it as soon as it has collected the frame.
    auto finished = std::make_shared<std::promise<void>>();
    std::future<void> result = finished->get_future();
    try {
        slot.done = std::async(std::launch::async, [this, &slot, frame, finished]() mutable {
            try {
                run(slot, frame);
                slot.busy = false;
                finished->set_value();
            } catch (...) {
                slot.busy = false;
                finished->set_exception(std::current_exception());
            }
        });
    } catch (...) {
        slot.busy = false;  // no thread started (std::system_error)
        throw;
    }
    return result;
}

void Detection_CoreAI::run(Slot &slot, cv::Mat &frame) {
    const int W = static_cast<int>(model_shape_.width);
    const int H = static_cast<int>(model_shape_.height);

    // Resize, scale to [0,1], swap BGR->RGB and pack into a 1x3xHxW CHW float blob
    // in a single call. The Swift interop layer owns the AIModel load/specialisation
    // and runs inference on this input, reporting the raw output tensor.
    cv::dnn::blobFromImage(frame, slot.blob, 1.0 / 255.0, cv::Size(W, H), cv::Scalar(),
                           /*swapRB=*/true, /*crop=*/false);

    std::array<int64_t, 4> input_shape = {1, 3, static_cast<int64_t>(H), static_cast<int64_t>(W)};

    // Initial output capacity: a raw YOLO head at this input size, i.e. one anchor
    // per cell of the stride-8/16/32 grids times (4 box + class) fields, which is
    // 8400 * 84 for 640x640 and 80 classes. End-to-end outputs (1, N, 6) are far
    // smaller. Anything larger (more classes, extra fields) is handled by the
    // grow-and-retry below, so this is only a starting point.
    if (slot.output.empty()) {
        int64_t anchors = 0;
        for (const int stride : {8, 16, 32}) {
            anchors += static_cast<int64_t>((H + stride - 1) / stride) * ((W + stride - 1) / stride);
        }
        const int64_t fields = 4 + std::max<int64_t>(static_cast<int64_t>(class_list_.size()), 80);
        slot.output.resize(static_cast<size_t>(anchors * fields));
    }

    int64_t output_numel = 0;
    // Room for more dimensions than the decoder accepts, so an unexpected rank is
    // still reported (and shown in the warning) instead of failing opaquely.
    std::array<int64_t, 8> output_shape = {};
    int64_t output_rank = 0;
    const auto infer = [&]() {
        return coreAIInfer(slot.blob.ptr<float>(), static_cast<int64_t>(slot.blob.total()),
                           input_shape.data(), static_cast<int64_t>(input_shape.size()),
                           model_file_.c_str(), slot.output.data(),
                           static_cast<int64_t>(slot.output.size()),
                           &output_numel, output_shape.data(),
                           static_cast<int64_t>(output_shape.size()), &output_rank);
    };
    int status = infer();
    if (status == kStatusBufferTooSmall &&
        output_numel > static_cast<int64_t>(slot.output.size())) {
        // The output is larger than estimated: grow once to the reported size and
        // rerun this frame. The buffer is kept, so later frames need one call only.
        slot.output.resize(static_cast<size_t>(output_numel));
        status = infer();
    }
    if (status != 0) {
        // Inference failures (input mismatch, model errors) repeat on every frame,
        // so surface them to main(), which aborts. The Swift layer has already
        // printed the specific reason on stderr.
        throw std::runtime_error("Core AI inference failed (status " + std::to_string(status) +
                                 "; see the [CoreAI] message above).");
    }
    if (output_numel == 0) {
        return;  // Valid empty output: nothing detected in this frame.
    }

    // The shared decoders take a [1, dim1, dim2] tensor; require exactly rank 3
    // with a matching element count rather than reading undefined dimensions.
    const std::vector<int64_t> reported_shape(output_shape.begin(),
                                              output_shape.begin() + output_rank);
    if (output_rank != 3 ||
        output_shape[0] * output_shape[1] * output_shape[2] != output_numel) {
        yolo_postprocess::warnUnsupportedShape("CoreAI", reported_shape);
        return;
    }

    CoreAITensor output;
    output.data = slot.output.data();
    output.numel = output_numel;
    output.shape = reported_shape;

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
