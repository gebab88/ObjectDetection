#ifndef OBJECTDETECTION_FRAMECROP_HPP
#define OBJECTDETECTION_FRAMECROP_HPP

#include <algorithm>

#include <opencv2/core.hpp>

namespace frame_crop {

// Returns the largest centered square ROI that fits inside a width x height
// frame. Used to crop frames to the square input expected by the models.
// NOTE: this discards the frame borders along the longer side, so objects near
// those edges are lost; letterboxing would preserve the full field of view.
inline cv::Rect centeredSquare(int width, int height) {
    const int side = std::min(width, height);
    const int x = (width - side) / 2;
    const int y = (height - side) / 2;
    return cv::Rect(x, y, side, side);
}

} // namespace frame_crop

#endif // OBJECTDETECTION_FRAMECROP_HPP
