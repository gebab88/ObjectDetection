#ifndef OBJECTDETECTION_DETECTIONHANDLER_COREAI_HPP
#define OBJECTDETECTION_DETECTIONHANDLER_COREAI_HPP

#include "Detection.hpp"
#include <string>
#include <vector>

// Raw inference output reported by the Swift Core AI interop layer. `data` points
// at the raw float32 output tensor and is owned by the backend (allocated by the
// backend for the inference buffer); `shape` holds the tensor dimensions, rank
// included (e.g. {1, 84, 8400} for a raw YOLO26 head output).
struct CoreAITensor {
    float* data = nullptr;
    int64_t numel = 0;
    std::vector<int64_t> shape;
};

// C-interop entry point implemented by the Swift Core AI interop layer (Task 3,
// Apple's Core AI framework: AIModel / InferenceFunction / InferenceValue) and
// called from the C++ backend. Runs one inference for a preprocessed frame and
// reports the raw output tensor through the out parameters; returns 0 on success,
// nonzero on failure. Input is flat float32 data plus its shape (CHW, [0,1]).
//   - input_data / input_numel / input_shape / input_rank: the preprocessed frame
//     blob (row-major CHW float32, scaled to [0,1]) and its shape (rank entries,
//     e.g. {1, 3, H, W}).
//   - model_path: path of the Core AI model asset (e.g. yolo26m.aimodel).
//   - output_data / output_capacity: caller-provided buffer that receives the raw
//     output tensor; the Swift layer writes at most `output_capacity` floats and
//     reports the actual number of valid values through *output_numel.
//   - output_numel / output_shape / output_shape_capacity: the Swift layer reports
//     the number of valid output values through *output_numel and the output
//     dimensions through output_shape[0 .. rank-1] (rank included, e.g. {1, 84, 8400}
//     for a raw YOLO26 head output; end-to-end outputs are {1, N, 6}).
//   - output_rank: receives the rank of the output tensor. The Swift layer fails
//     (nonzero) instead of writing past output_shape when the rank exceeds
//     output_shape_capacity, and likewise when the values exceed output_capacity.
// The output is always delivered as dense row-major float32: float16 outputs are
// converted and strided (non-contiguous) layouts are gathered.
// Returns 0 on success, nonzero on failure (including invalid input).
extern "C" int coreAIInfer(const float* input_data,
                           int64_t input_numel,
                           const int64_t* input_shape,
                           int64_t input_rank,
                           const char* model_path,
                           float* output_data,
                           int64_t output_capacity,
                           int64_t* output_numel,
                           int64_t* output_shape,
                           int64_t output_shape_capacity,
                           int64_t* output_rank);

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
