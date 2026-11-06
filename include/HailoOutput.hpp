#ifndef HAILO_OUTPUT_HPP
#define HAILO_OUTPUT_HPP

#include <cstddef>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "Yolo26Decoder.hpp"

// Turns the two kinds of Hailo detector output into boxes in frame pixels and
// draws them with the same routine as the other backends
// (yolo_postprocess::drawDetection). Independent of HailoRT, so it is tested
// on every platform.
namespace hailo_output {

struct Detection {
    cv::Rect box;  // frame pixels
    int class_id = -1;
    float score = 0.0f;
};

// Parses HailoRT's HAILO_FORMAT_ORDER_HAILO_NMS_BY_CLASS float32 output: for
// each class the box count, followed by that many packed
// {y_min, x_min, y_max, x_max, score} with coordinates normalized to [0, 1].
// Boxes below score_threshold or outside the frame are skipped; a count beyond
// max_boxes_per_class or the buffer end is cut off.
std::vector<Detection> parseNmsByClass(const float* data, std::size_t float_count,
                                       int num_classes, int max_boxes_per_class,
                                       float score_threshold, cv::Size frame_size);

// Scales decoded YOLO26 results from model input pixels to the frame.
std::vector<Detection> fromYolo26(const std::vector<yolo26::Result>& results,
                                  cv::Size2f model_shape, cv::Size frame_size);

// Draws boxes and labels, best score first so its label gets the preferred
// place. Detections with a class id outside class_list are skipped.
void drawDetections(cv::Mat& frame, const std::vector<Detection>& detections,
                    const std::vector<std::string>& class_list);

}  // namespace hailo_output

#endif  // HAILO_OUTPUT_HPP
