#ifndef OBJECTDETECTION_YOLOPOSTPROCESS_HPP
#define OBJECTDETECTION_YOLOPOSTPROCESS_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include "Framework.hpp"

namespace yolo_postprocess {

// Optional debug logging of decoded detection labels to stdout. Off by default
// so the real-time detection path stays quiet; tests opt in to inspect labels.
inline bool& logDetectionsToStdout() {
    static bool enabled = false;
    return enabled;
}

// YOLO class scores come in two flavours: some exports already apply the final
// sigmoid (values in [0,1]), others emit raw logits. A value that is already a
// valid probability is returned unchanged, otherwise it is squashed through a
// sigmoid. NOTE: this is a best-effort heuristic for mixed exports, not an exact
// inverse — a raw logit that happens to fall inside [0,1] is returned unchanged.
// Prefer models whose post-processing matches the decoder used here.
inline float confidence(float value) {
    if (value >= 0.0f && value <= 1.0f) {
        return value;
    }
    return 1.0f / (1.0f + std::exp(-value));
}

inline cv::Rect clippedRect(float left, float top, float right, float bottom, const cv::Size& frame_size) {
    const int x1 = std::clamp(static_cast<int>(std::round(left)), 0, frame_size.width - 1);
    const int y1 = std::clamp(static_cast<int>(std::round(top)), 0, frame_size.height - 1);
    const int x2 = std::clamp(static_cast<int>(std::round(right)), 0, frame_size.width - 1);
    const int y2 = std::clamp(static_cast<int>(std::round(bottom)), 0, frame_size.height - 1);
    if (x2 <= x1 || y2 <= y1) {
        return {};
    }
    return cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2));
}

inline std::string detectionLabel(int class_id,
                                  float score,
                                  const std::vector<std::string>& class_list) {
    return class_list[class_id] + "  " + std::to_string(static_cast<int>(100 * score)) + "%";
}

inline bool intersectsAny(const cv::Rect& rect, const std::vector<cv::Rect>& occupied) {
    for (const auto& used : occupied) {
        if ((rect & used).area() > 0) {
            return true;
        }
    }
    return false;
}

inline cv::Rect makeLabelRect(const cv::Size& frame_size,
                              int preferred_left,
                              int preferred_top,
                              const cv::Size& text_size,
                              int baseline,
                              int padding,
                              cv::Point& text_origin) {
    const int label_width = std::min(frame_size.width, text_size.width + 2 * padding);
    const int label_height = std::min(frame_size.height, text_size.height + baseline + 2 * padding);
    const int left = std::clamp(preferred_left, 0, std::max(0, frame_size.width - label_width));
    const int top = std::clamp(preferred_top, 0, std::max(0, frame_size.height - label_height));

    text_origin = cv::Point(left + padding, top + padding + std::min(text_size.height, label_height));
    return cv::Rect(left, top, label_width, label_height);
}

inline void drawDetection(cv::Mat& frame,
                          const cv::Rect& box,
                          int class_id,
                          float score,
                          const std::vector<std::string>& class_list,
                          std::vector<cv::Rect>* occupied_labels = nullptr) {
    if (class_id < 0 || class_id >= static_cast<int>(class_list.size())) {
        return;
    }

    cv::rectangle(frame, box, cv::Scalar(0, 255, 255), 1);
    const std::string label = detectionLabel(class_id, score, class_list);
    if (logDetectionsToStdout()) {
        std::cout << label << '\n';
    }

    constexpr int font_face = cv::FONT_HERSHEY_SIMPLEX;
    constexpr int thickness = 1;
    constexpr int padding = 3;
    double font_scale = 0.55;
    int baseline = 0;
    cv::Size text_size = cv::getTextSize(label, font_face, font_scale, thickness, &baseline);
    while (font_scale > 0.35 && text_size.width + 2 * padding > frame.cols) {
        font_scale -= 0.05;
        text_size = cv::getTextSize(label, font_face, font_scale, thickness, &baseline);
    }

    const cv::Size frame_size(frame.cols, frame.rows);
    const int label_height = std::min(frame.rows, text_size.height + baseline + 2 * padding);
    const std::array<int, 3> candidate_tops = {
        box.y - label_height - 2,
        box.y + box.height + 2,
        box.y + 2,
    };

    cv::Point text_origin;
    cv::Rect label_rect;
    bool placed = false;
    for (size_t i = 0; i < candidate_tops.size(); ++i) {
        const int top = candidate_tops[i];
        if (i < 2 && (top < 0 || top + label_height > frame.rows)) {
            continue;
        }

        const cv::Rect candidate = makeLabelRect(frame_size, box.x, top, text_size,
                                                baseline, padding, text_origin);
        if (!occupied_labels || !intersectsAny(candidate, *occupied_labels)) {
            label_rect = candidate;
            placed = true;
            break;
        }
    }

    if (!placed) {
        label_rect = makeLabelRect(frame_size, box.x, box.y + 2, text_size,
                                   baseline, padding, text_origin);
    }

    if (occupied_labels) {
        occupied_labels->push_back(label_rect);
    }

    cv::rectangle(frame, label_rect, cv::Scalar(0, 0, 0), cv::FILLED);
    cv::rectangle(frame, label_rect, cv::Scalar(0, 255, 255), 1);
    cv::putText(frame, label, text_origin, font_face, font_scale,
                cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);
}

inline void drawNmsDetections(cv::Mat& frame,
                              const std::vector<cv::Rect>& boxes,
                              const std::vector<int>& class_ids,
                              const std::vector<float>& scores,
                              const std::vector<std::string>& class_list,
                              float score_threshold,
                              float nms_threshold) {
    if (boxes.empty()) {
        return;
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, scores, score_threshold, nms_threshold, nms_result);

    std::vector<cv::Rect> occupied_labels;
    for (const int idx : nms_result) {
        drawDetection(frame, boxes[idx], class_ids[idx], scores[idx], class_list, &occupied_labels);
    }
}

inline bool decodeEndToEnd(const float* data,
                           int64_t dim1,
                           int64_t dim2,
                           cv::Mat& frame,
                           cv::Size2f model_shape,
                           const std::vector<std::string>& class_list,
                           float score_threshold,
                           float nms_threshold) {
    constexpr int64_t max_end_to_end_detections = 1000;
    constexpr int64_t max_end_to_end_fields = 128;

    bool fields_first = false;
    int fields = 0;
    int num_dets = 0;

    auto plausible_fields = [](int64_t dim) {
        return dim >= 6 && dim <= max_end_to_end_fields;
    };
    const bool dim1_is_fields = plausible_fields(dim1) && dim2 > 0 && dim2 <= max_end_to_end_detections;
    const bool dim2_is_fields = plausible_fields(dim2) && dim1 > 0 && dim1 <= max_end_to_end_detections;

    if (dim1_is_fields && (!dim2_is_fields || dim1 <= dim2)) {
        fields_first = true;
        fields = static_cast<int>(dim1);
        num_dets = static_cast<int>(dim2);
    } else if (dim2_is_fields) {
        fields_first = false;
        fields = static_cast<int>(dim2);
        num_dets = static_cast<int>(dim1);
    } else {
        return false;
    }

    const float x_scale = static_cast<float>(frame.cols) / model_shape.width;
    const float y_scale = static_cast<float>(frame.rows) / model_shape.height;
    const cv::Size frame_size(frame.cols, frame.rows);

    auto value_at = [&](int field, int det) {
        return fields_first ? data[field * num_dets + det] : data[det * fields + field];
    };

    std::vector<float> scores;
    std::vector<int> class_ids;
    std::vector<cv::Rect> boxes;
    scores.reserve(num_dets);
    class_ids.reserve(num_dets);
    boxes.reserve(num_dets);

    for (int i = 0; i < num_dets; ++i) {
        const float score = value_at(4, i);
        if (score < score_threshold) {
            continue;
        }

        const int class_id = static_cast<int>(value_at(5, i));
        if (class_id < 0 || class_id >= static_cast<int>(class_list.size())) {
            continue;
        }

        const float x1 = value_at(0, i) * x_scale;
        const float y1 = value_at(1, i) * y_scale;
        const float x2 = value_at(2, i) * x_scale;
        const float y2 = value_at(3, i) * y_scale;
        const cv::Rect box = clippedRect(x1, y1, x2, y2, frame_size);
        if (box.empty()) {
            continue;
        }

        scores.push_back(score);
        class_ids.push_back(class_id);
        boxes.push_back(box);
    }

    drawNmsDetections(frame, boxes, class_ids, scores, class_list, score_threshold, nms_threshold);
    return true;
}

inline bool decodeRaw(const float* data,
                      int64_t dim1,
                      int64_t dim2,
                      cv::Mat& frame,
                      cv::Size2f model_shape,
                      const std::vector<std::string>& class_list,
                      float score_threshold,
                      float nms_threshold) {
    if (dim1 < 5 || dim2 < 1 || class_list.empty()) {
        return false;
    }

    const bool fields_first = dim1 <= dim2;
    const int fields = static_cast<int>(fields_first ? dim1 : dim2);
    const int num_dets = static_cast<int>(fields_first ? dim2 : dim1);
    const int class_count = static_cast<int>(class_list.size());
    if (class_count <= 0 || fields < 4 + class_count) {
        return false;
    }

    // Raw (NMS-free) layout has 4 + class_count fields. The caller decides which
    // decoder to run based on the known MODEL_FORMAT (raw for YOLO12, end-to-end
    // for YOLO26, raw-then-end-to-end for the open-vocabulary YOLOE export), so
    // we no longer guess between raw and end-to-end from the field count here.
    const float x_scale = static_cast<float>(frame.cols) / model_shape.width;
    const float y_scale = static_cast<float>(frame.rows) / model_shape.height;
    const cv::Size frame_size(frame.cols, frame.rows);

    auto value_at = [&](int field, int det) {
        return fields_first ? data[field * num_dets + det] : data[det * fields + field];
    };

    std::vector<float> scores;
    std::vector<int> class_ids;
    std::vector<cv::Rect> boxes;
    scores.reserve(num_dets);
    class_ids.reserve(num_dets);
    boxes.reserve(num_dets);

    for (int i = 0; i < num_dets; ++i) {
        float score = -std::numeric_limits<float>::infinity();
        int class_id = -1;
        for (int c = 0; c < class_count; ++c) {
            const float class_score = confidence(value_at(4 + c, i));
            if (class_score > score) {
                score = class_score;
                class_id = c;
            }
        }

        if (score < score_threshold || class_id < 0) {
            continue;
        }

        const float x = value_at(0, i) * x_scale;
        const float y = value_at(1, i) * y_scale;
        const float w = value_at(2, i) * x_scale;
        const float h = value_at(3, i) * y_scale;
        const cv::Rect box = clippedRect(x - 0.5f * w, y - 0.5f * h,
                                         x + 0.5f * w, y + 0.5f * h,
                                         frame_size);
        if (box.empty()) {
            continue;
        }

        scores.push_back(score);
        class_ids.push_back(class_id);
        boxes.push_back(box);
    }

    drawNmsDetections(frame, boxes, class_ids, scores, class_list, score_threshold, nms_threshold);
    return true;
}

// Dispatches to the right decoder(s) based on the known model format: raw
// (NMS-free) for YOLO12, end-to-end for YOLO26, and raw-then-end-to-end for the
// YOLOE export (whose ONNX may be either, depending on the --nms export flag).
// Returns true when a decoder handled the output. Shared by all backends so the
// format<->decoder rule lives in one place.
inline bool decodeByFormat(MODEL_FORMAT model_format,
                           const float* data,
                           int64_t dim1,
                           int64_t dim2,
                           cv::Mat& frame,
                           cv::Size2f model_shape,
                           const std::vector<std::string>& class_list,
                           float score_threshold,
                           float nms_threshold) {
    switch (model_format) {
        case MODEL_FORMAT::YOLO12:
            return decodeRaw(data, dim1, dim2, frame, model_shape, class_list,
                             score_threshold, nms_threshold);
        case MODEL_FORMAT::YOLO26:
            return decodeEndToEnd(data, dim1, dim2, frame, model_shape, class_list,
                                  score_threshold, nms_threshold);
        case MODEL_FORMAT::YOLOE:
            return decodeRaw(data, dim1, dim2, frame, model_shape, class_list,
                             score_threshold, nms_threshold) ||
                   decodeEndToEnd(data, dim1, dim2, frame, model_shape, class_list,
                                  score_threshold, nms_threshold);
        case MODEL_FORMAT::Unknown:
            return false;
    }
    return false;
}

inline void warnUnsupportedShape(const char* backend, const std::vector<int64_t>& shape) {
    static bool warned = false;
    if (warned) {
        return;
    }

    std::cerr << backend << " unsupported detection output shape: [";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cerr << shape[i] << (i + 1 < shape.size() ? ", " : "");
    }
    std::cerr << "]" << std::endl;
    warned = true;
}

} // namespace yolo_postprocess

#endif // OBJECTDETECTION_YOLOPOSTPROCESS_HPP
