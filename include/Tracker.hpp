#ifndef TRACKER_HPP
#define TRACKER_HPP
#include <opencv2/opencv.hpp>
//#include <opencv2/tracking.hpp>

using namespace cv;
using namespace std;

void track( Mat &frame, Ptr<Tracker> tracker, Rect2i &trackingBox );

#endif // TRACKER_HPP