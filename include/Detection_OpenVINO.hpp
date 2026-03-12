#ifndef OBJECTDETECTION_DETECTIONHANDLER_OPENVINO_H
#define OBJECTDETECTION_DETECTIONHANDLER_OPENVINO_H

#include <openvino/openvino.hpp>

#include "Detection.hpp"

class Detection_OpenVINO : public Detection {
    public:
    Detection_OpenVINO( float score_threshold,
                        const cv::Size2f model_shape,
                        const std::string &model_file);
    virtual ~Detection_OpenVINO() = default;

    void detect(cv::Mat &frame) override;

    private:
    ov::Core core_;
    ov::Tensor input_, output_;
    ov::InferRequest infer_request_;
    size_t plane_;
    std::vector<cv::Mat> chans_;
};


#endif //OBJECTDETECTION_DETECTIONHANDLER_OPENVINO_H