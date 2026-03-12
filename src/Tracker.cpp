#include "Tracker.hpp"

void track( const cv::Mat &frame, cv::Ptr<cv::Tracker> tracker, cv::Rect2i &trackingBox ) {
    // Update the tracker with the new frame
    if (tracker->update(frame, trackingBox)) {
        // Draw the tracking box
        cv::rectangle(frame, trackingBox, cv::Scalar(255, 0, 0), 2, 1);
    } else {
        std::cout << "Tracking failure detected." << std::endl;
    }
}