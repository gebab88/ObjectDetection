#include "Yolo26Decoder.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace yolo26 {
namespace {

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// Smallest quantized logit worth looking at: everything below it has a score
// under the threshold. One step of slack absorbs rounding; the exact check is
// done on the dequantized value.
uint32_t quantizedLogitFloor(float score_threshold, Quant quant) {
    if (score_threshold <= 0.0f) return 0;
    if (score_threshold >= 1.0f) return std::numeric_limits<uint32_t>::max();
    const float logit = std::log(score_threshold / (1.0f - score_threshold));
    const float q = std::floor(logit / quant.scale + quant.zero_point) - 1.0f;
    if (q <= 0.0f) return 0;
    if (q >= static_cast<float>(std::numeric_limits<uint16_t>::max())) {
        return std::numeric_limits<uint32_t>::max();
    }
    return static_cast<uint32_t>(q);
}

float dequantize(uint16_t q, Quant quant) {
    return (static_cast<float>(q) - quant.zero_point) * quant.scale;
}

}  // namespace

std::vector<Result> decode(const std::vector<HeadLevel>& levels, int num_classes,
                           float score_threshold, std::size_t max_detections) {
    std::vector<Result> results;
    if (num_classes <= 0 || max_detections == 0) return results;

    for (const HeadLevel& level : levels) {
        const uint32_t q_floor = quantizedLogitFloor(score_threshold, level.score_quant);
        const uint16_t* cell_scores = level.scores;
        const uint16_t* cell_box = level.boxes;

        for (int y = 0; y < level.height; ++y) {
            for (int x = 0; x < level.width; ++x, cell_scores += num_classes, cell_box += 4) {
                for (int c = 0; c < num_classes; ++c) {
                    if (cell_scores[c] < q_floor) continue;  // fast path: almost all pairs
                    const float score = sigmoid(dequantize(cell_scores[c], level.score_quant));
                    if (score < score_threshold) continue;

                    const float ax = static_cast<float>(x) + 0.5f;
                    const float ay = static_cast<float>(y) + 0.5f;
                    Result r;
                    r.x1 = (ax - dequantize(cell_box[0], level.box_quant)) * level.stride;
                    r.y1 = (ay - dequantize(cell_box[1], level.box_quant)) * level.stride;
                    r.x2 = (ax + dequantize(cell_box[2], level.box_quant)) * level.stride;
                    r.y2 = (ay + dequantize(cell_box[3], level.box_quant)) * level.stride;
                    r.class_id = c;
                    r.score = score;
                    results.push_back(r);
                }
            }
        }
    }

    // Best first; stable so equal scores keep the head's order (level, cell, class).
    std::stable_sort(results.begin(), results.end(),
                     [](const Result& a, const Result& b) { return a.score > b.score; });
    if (results.size() > max_detections) results.resize(max_detections);
    return results;
}

}  // namespace yolo26
