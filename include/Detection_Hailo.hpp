#ifndef DETECTION_HAILO_HPP
#define DETECTION_HAILO_HPP

#include "Detection.hpp"
#include "Yolo26Decoder.hpp"
#include "hailo/hailort.hpp"
#include <atomic>
#include <future>
#include <map>
#include <memory>
#include <string>
#include <vector>

// Hailo backend (Raspberry Pi with a Hailo accelerator).
//
// Supports two kinds of HEF:
//  - HEFs with on-chip NMS (e.g. Model Zoo yolov8m): HailoRT returns the final
//    boxes per class.
//  - Raw YOLO26 HEFs (e.g. Model Zoo yolo26m): per head scale one box tensor
//    (4 channels) and one class tensor; decoded on the CPU by yolo26::decode.
//
// Several frames are kept in flight (run_async with one buffer set per frame),
// so the chip works on the next frame while the CPU prepares, decodes and
// writes the others.
class Detection_Hailo : public Detection {
public:
    Detection_Hailo(float score_threshold,
                    float nms_iou_threshold,
                    cv::Size2f model_shape,
                    const std::string &model_file);
    ~Detection_Hailo() override;

    void detect(cv::Mat &frame) override;
    std::size_t max_frames_in_flight() const override { return slots_.size(); }
    std::future<void> detect_async(cv::Mat frame) override;

private:
    struct DetectionResult {
        cv::Rect box;  // model input pixels
        int class_id = -1;
        float score = 0.0f;
    };

    // Output pair of one YOLO26 head scale.
    struct RawLevel {
        std::string boxes_output;
        std::string scores_output;
        yolo26::Quant box_quant;
        yolo26::Quant score_quant;
        int height = 0;
        int width = 0;
        float stride = 0.0f;
    };

    // Buffers and bindings of one frame in flight.
    struct Slot {
        std::vector<uint8_t> input;
        std::map<std::string, std::vector<uint8_t>> outputs;
        hailort::ConfiguredInferModel::Bindings bindings;
        std::atomic<hailo_status> status{HAILO_SUCCESS};  // set by the completion callback
        bool busy = false;
        hailort::AsyncInferJob job;  // last member: its destructor waits for the job
    };

    void setup_raw_outputs(hailort::InferModel &infer_model);
    void finish(Slot &slot, cv::Mat &frame);
    void draw_nms_output(Slot &slot, cv::Mat &frame);
    std::vector<DetectionResult> decode_raw_output(Slot &slot);
    void draw_detections(cv::Mat &frame, const std::vector<DetectionResult> &detections);

    std::unique_ptr<hailort::VDevice>      vdevice_;
    hailort::ConfiguredInferModel          configured_model_;

    std::string   input_name_;
    std::string   output_name_;
    hailo_nms_shape_t nms_shape_{};
    bool nms_output_ = false;
    std::vector<RawLevel> raw_levels_;
    int raw_num_classes_ = 0;

    std::vector<std::unique_ptr<Slot>> slots_;  // after configured_model_: destroyed first
    std::size_t next_slot_ = 0;
};

#endif // DETECTION_HAILO_HPP
