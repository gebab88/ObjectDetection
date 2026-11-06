#ifndef YOLO26_DECODER_HPP
#define YOLO26_DECODER_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

// Decoder for the raw outputs of a YOLO26 detection head compiled to a Hailo
// HEF (e.g. the Model Zoo's yolo26m).
//
// The HEF ends at the head's last convolutions and delivers, per scale, two
// quantized NHWC tensors: 4 box distances (left, top, right, bottom in grid
// cells, measured from the cell centre) and one logit per class. YOLO26 is
// NMS-free, so decoding is: sigmoid of the logits, keep (cell, class) pairs at
// or above the score threshold, convert the distances to corner coordinates and
// keep the best max_detections pairs. This matches Ultralytics' end2end export
// (top 300 by score), without the ONNX postprocess model it needed before.
namespace yolo26 {

// Dequantization: value = (q - zero_point) * scale.
struct Quant {
    float zero_point = 0.0f;
    float scale = 1.0f;
};

// One scale of the head: a height x width grid, NHWC, UINT16.
struct HeadLevel {
    const uint16_t* boxes = nullptr;   // height * width * 4
    Quant box_quant;
    const uint16_t* scores = nullptr;  // height * width * num_classes
    Quant score_quant;
    int height = 0;
    int width = 0;
    float stride = 0.0f;  // model input pixels per grid cell
};

// Box corners in model input pixels.
struct Result {
    float x1 = 0.0f, y1 = 0.0f, x2 = 0.0f, y2 = 0.0f;
    int class_id = -1;
    float score = 0.0f;
};

// Returns the detections with score >= score_threshold, best first, at most
// max_detections of them.
std::vector<Result> decode(const std::vector<HeadLevel>& levels, int num_classes,
                           float score_threshold, std::size_t max_detections = 300);

}  // namespace yolo26

#endif  // YOLO26_DECODER_HPP
