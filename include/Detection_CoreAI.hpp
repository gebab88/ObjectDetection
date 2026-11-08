#ifndef OBJECTDETECTION_DETECTIONHANDLER_COREAI_HPP
#define OBJECTDETECTION_DETECTIONHANDLER_COREAI_HPP

#include "Detection.hpp"
#include "FrameSlots.hpp"
#include <atomic>
#include <cstdint>
#include <future>
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
//     reports the actual number of valid values through *output_numel (see the
//     status 2 contract below when the buffer is too small).
//   - output_numel / output_shape / output_shape_capacity: the Swift layer reports
//     the number of valid output values through *output_numel and the output
//     dimensions through output_shape[0 .. rank-1] (rank included, e.g. {1, 84, 8400}
//     for a raw YOLO26 head output; end-to-end outputs are {1, N, 6}).
//   - output_rank: receives the rank of the output tensor. The Swift layer fails
//     (nonzero) instead of writing past output_shape when the rank exceeds
//     output_shape_capacity, and likewise when the values exceed output_capacity.
// The output is always delivered as dense row-major float32: float16 outputs are
// converted and strided (non-contiguous) layouts are gathered. Float16 models
// (FP16 exports) get the float32 input converted.
//   - lane: which frame in flight this call belongs to (0 .. coreAIFramesInFlight()-1);
//     it selects the compute unit the inference runs on (COREAI_COMPUTE_UNIT).
// Returns 0 on success (an empty output with *output_numel == 0 is valid), 2 when
// the output has more values than output_capacity (then *output_numel holds the
// required count, nothing else is written, and the caller may retry with a larger
// buffer), and any other nonzero value on failure (the Swift layer has already
// printed the reason to stderr).
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
                           int64_t* output_rank,
                           int64_t lane);

// Number of frames to keep in flight: one per compute lane configured in the
// Swift layer (COREAI_COMPUTE_UNIT; GPU, GPU, Neural Engine, twice, by default).
extern "C" int64_t coreAIFramesInFlight();

// Core AI backend: preprocessing and decoding in C++, inference in Swift.
//
// Several frames are kept in flight (one per compute lane of the Swift layer),
// each on its own worker thread with its own buffers: the next inferences start
// while earlier frames are still being decoded and drawn, concurrent inferences
// keep the GPU busier than back-to-back ones, and a lane on the Neural Engine
// adds a second accelerator.
class Detection_CoreAI : public Detection {
    public:
    Detection_CoreAI(  const float score_threshold,
                       const cv::Size2f model_shape,
                       const std::string &model_file,
                       const float nms_threshold);
    ~Detection_CoreAI() override = default;  // slots_ waits for its frames

    void detect(cv::Mat &frame) override;
    std::size_t max_frames_in_flight() const override { return slots_.size(); }
    std::future<void> detect_async(cv::Mat frame) override;

    // Status returned by coreAIInfer when output_capacity is too small.
    static constexpr int kStatusBufferTooSmall = 2;

    private:
    // Buffers of one frame in flight.
    struct Scratch {
        cv::Mat blob;
        // Reused across frames; sized lazily from the model shape and grown when
        // the Swift layer reports a larger output (status kStatusBufferTooSmall).
        std::vector<float> output;
        int64_t lane = -1;  // compute lane, assigned on the slot's first frame
    };

    void run(Scratch &slot, cv::Mat &frame);

    float nms_threshold_;
    std::atomic<int64_t> next_lane_{0};
    FrameSlots<Scratch> slots_;  // last member: destroyed (and waited for) first
};

#endif //OBJECTDETECTION_DETECTIONHANDLER_COREAI_HPP
