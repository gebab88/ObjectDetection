#ifndef OBJECTDETECTION_DETECTIONHANDLER_OPENCV_H
#define OBJECTDETECTION_DETECTIONHANDLER_OPENCV_H

#include <opencv2/dnn.hpp>

#include "Detection.hpp"


class Detection_OpenCV : public Detection {
    public:
    Detection_OpenCV(       const float score_threshold,
                            const cv::Size2f model_shape,
                            const std::string &model_file,
                            const float nms,
                            const bool use_cuda);
    ~Detection_OpenCV() = default;

    void detect( cv::Mat &frame ) override;
    private:

    std::vector<cv::Mat> outs;
    cv::dnn::Net net;
    float nms_threshold_;
    bool use_cuda_;
};

#endif //OBJECTDETECTION_DETECTIONHANDLER_OPENCV_H