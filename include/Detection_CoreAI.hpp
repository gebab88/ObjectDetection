#ifndef OBJECTDETECTION_DETECTIONHANDLER_COREAI_HPP
#define OBJECTDETECTION_DETECTIONHANDLER_COREAI_HPP

#include "Detection.hpp"
#include <string>
#include <vector>

// Raw inference output reported by the Swift Core AI interop layer. `data` points
// at the raw float32 output tensor and is owned by the caller (released with
// std::free); `shape` holds the tensor dimensions, rank included (e.g. {1, 84, 8400}
// for a raw YOLO26 head output).
struct CoreAITensor {
    float* data = nullptr;
    int64_t numel = 0;
    std::vector<int64_t> shape;
};

// C-interop entry point implemented by the Swift Core AI interop layer (Task 3,
// Apple's Core AI framework: AIModel / InferenceFunction / InferenceValue) and
// called from the C++ backend. Runs one inference for a preprocessed frame and
// reports the raw output tensor through the out parameter; returns 0 on success,
// nonzero on failure. Input is flat float32 data plus its shape (CHW, [0,1]).
extern "C" int coreAIInfer(const float* input_data,
                           int64_t input_numel,
                           const int64_t* input_shape,
                           int64_t input_rank,
                           CoreAITensor* output);

class Detection_CoreAI : public Detection {
    public:
    Detection_CoreAI(  const float score_threshold,
                       const cv::Size2f model_shape,
                       const std::string &model_file,
                       const float nms_threshold);
    ~Detection_CoreAI() override = default;

    void detect(cv::Mat &frame) override;

    private:
    float nms_threshold_;
};

#endif //OBJECTDETECTION_DETECTIONHANDLER_COREAI_HPP
