#include "HailoOutput.hpp"

#include <algorithm>

#include "YoloPostprocess.hpp"

namespace hailo_output {

std::vector<Detection> parseNmsByClass(const float* data, std::size_t float_count,
                                       int num_classes, int max_boxes_per_class,
                                       float score_threshold, cv::Size frame_size) {
    constexpr std::size_t kBoxFloats = 5;  // y_min, x_min, y_max, x_max, score
    const float width = static_cast<float>(frame_size.width);
    const float height = static_cast<float>(frame_size.height);

    std::vector<Detection> detections;
    std::size_t pos = 0;
    for (int cls = 0; cls < num_classes && pos < float_count; ++cls) {
        const std::size_t available = (float_count - pos - 1) / kBoxFloats;
        const float claimed = std::max(0.0f, data[pos++]);
        const std::size_t count = std::min({static_cast<std::size_t>(claimed),
                                            static_cast<std::size_t>(std::max(0, max_boxes_per_class)),
                                            available});
        for (std::size_t i = 0; i < count; ++i, pos += kBoxFloats) {
            const float* b = data + pos;
            if (b[4] < score_threshold) continue;
            const cv::Rect box = yolo_postprocess::clippedRect(b[1] * width, b[0] * height,
                                                               b[3] * width, b[2] * height, frame_size);
            if (box.empty()) continue;
            detections.push_back({box, cls, b[4]});
        }
    }
    return detections;
}

std::vector<Detection> fromYolo26(const std::vector<yolo26::Result>& results,
                                  cv::Size2f model_shape, cv::Size frame_size) {
    const float x_scale = static_cast<float>(frame_size.width) / model_shape.width;
    const float y_scale = static_cast<float>(frame_size.height) / model_shape.height;

    std::vector<Detection> detections;
    detections.reserve(results.size());
    for (const auto& r : results) {
        const cv::Rect box = yolo_postprocess::clippedRect(r.x1 * x_scale, r.y1 * y_scale,
                                                           r.x2 * x_scale, r.y2 * y_scale, frame_size);
        if (box.empty()) continue;
        detections.push_back({box, r.class_id, r.score});
    }
    return detections;
}

void drawDetections(cv::Mat& frame, const std::vector<Detection>& detections,
                    const std::vector<std::string>& class_list) {
    std::vector<const Detection*> order;
    order.reserve(detections.size());
    for (const auto& det : detections) order.push_back(&det);
    std::stable_sort(order.begin(), order.end(),
                     [](const Detection* a, const Detection* b) { return a->score > b->score; });

    std::vector<cv::Rect> occupied_labels;
    for (const Detection* det : order) {
        yolo_postprocess::drawDetection(frame, det->box, det->class_id, det->score, class_list,
                                        &occupied_labels);
    }
}

}  // namespace hailo_output
