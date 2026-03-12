#ifndef TRACKER_HPP
#define TRACKER_HPP
#include <opencv2/opencv.hpp>
//#include <opencv2/tracking.hpp>


void track( cv::Mat &frame, cv::Ptr<cv::Tracker> tracker, cv::Rect2i &trackingBox );

#endif // TRACKER_HPP