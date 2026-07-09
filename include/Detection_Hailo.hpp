#ifndef DETECTION_HAILO_HPP
#define DETECTION_HAILO_HPP

#include "Detection.hpp"
#include "hailo/hailort.hpp"
#include <onnxruntime_cxx_api.h>
#include <array>
#include <memory>
#include <vector>
#include <map>

class Detection_Hailo : public Detection {
public:
    Detection_Hailo(float score_threshold,
                    float nms_iou_threshold,
                    cv::Size2f model_shape,
                    const std::string &model_file);

    void detect(cv::Mat &frame) override;

private:
    struct RawOutputInfo {
        std::string onnx_name;
        int channels = 0;
        int height = 0;
        int width = 0;
    };

    struct DetectionResult {
        cv::Rect box;
        int class_id = -1;
        float score = 0.0f;
    };

    void setup_yolo26_postprocess();
    std::vector<DetectionResult> run_yolo26_postprocess();
    void draw_detections(cv::Mat &frame, const std::vector<DetectionResult> &detections);

    std::unique_ptr<hailort::VDevice>      vdevice_;
    hailort::ConfiguredInferModel          configured_model_;
    hailort::ConfiguredInferModel::Bindings bindings_;

    std::string   input_name_;
    std::string   output_name_;
    std::vector<std::string> output_names_;
    hailo_nms_shape_t nms_shape_{};
    bool nms_output_ = false;
    bool raw_output_notice_printed_ = false;

    Ort::Env ort_env_;
    Ort::SessionOptions ort_session_opts_;
    Ort::AllocatorWithDefaultOptions ort_allocator_;
    std::unique_ptr<Ort::Session> postprocess_session_;
    std::vector<std::string> postprocess_output_names_;
    std::vector<std::string> raw_output_order_;
    std::map<std::string, RawOutputInfo> raw_output_info_;

    std::vector<uint8_t> input_buffer_;
    std::map<std::string, std::vector<uint8_t>> output_buffers_;
};

#endif // DETECTION_HAILO_HPP
