#ifndef DETECTION_PIPELINE_HPP
#define DETECTION_PIPELINE_HPP

#include <functional>

#include <opencv2/core.hpp>

#include "Detection.hpp"

// Runs frames through a detector, keeping up to
// detection.max_frames_in_flight() frames in progress at once.
//
// read() supplies the next frame and returns false at the end. Each frame is
// read into a fresh cv::Mat, so frames still in flight are never overwritten.
// deliver() receives the finished frames in reading order; returning false
// stops the run. With a synchronous backend (one frame in flight) each frame
// is delivered before the next one is read, exactly like a plain loop.
//
// Frames still in flight when reading stops are finished and delivered.
// Detection errors are rethrown. Returns false if deliver() failed.
namespace detection_pipeline {

bool run(Detection& detection,
         const std::function<bool(cv::Mat&)>& read,
         const std::function<bool(cv::Mat&)>& deliver);

}  // namespace detection_pipeline

#endif  // DETECTION_PIPELINE_HPP
