#include "Detection_OpenCV.hpp"
#include "YoloPostprocess.hpp"

Detection_OpenCV::Detection_OpenCV(const float score_threshold,
                                   const cv::Size2f model_shape,
                                   const std::string &model_file,
                                   const float nms,
                                   const bool use_cuda) :
                                   
                                   Detection(score_threshold, model_shape, model_file),
                                   nms_threshold_(nms),
                                   use_cuda_(use_cuda) {
        net = cv::dnn::readNet(model_file);

        if (use_cuda_) {
            std::cout << "Running on CUDA" << std::endl << std::endl;
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        } else {
            std::cout << "Running on CPU" << std::endl << std::endl;
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        }
    }

    void Detection_OpenCV::detect( cv::Mat &frame) {
        outs.clear();

        // Convert the image into a blob and set it as input to the network
        //cv::resize(frame, resized_, {640, 640});
        cv::dnn::blobFromImage(frame, blob_, 1.0/255.0, model_shape_, cv::Scalar(), true, false);
        net.setInput(blob_);
        net.forward(outs, net.getUnconnectedOutLayersNames());
    

    if (outs.empty() || outs[0].dims != 3) {
        return;
    }

    float* const data = outs[0].ptr<float>();
    const int64_t dim1 = outs[0].size[1];
    const int64_t dim2 = outs[0].size[2];

    // OpenCV DNN cannot load the end-to-end YOLO26 graph, so main() never selects
    // this backend for YOLO26 (see supportedBackends). Only the raw YOLO12 layout
    // and the YOLOE export (raw, with an end-to-end fallback) are handled here.
    switch (model_format_) {
        case MODEL_FORMAT::YOLO12:
            if (yolo_postprocess::decodeRaw(data, dim1, dim2, frame, model_shape_, class_list,
                                            score_threshold_, nms_threshold_)) {
                return;
            }
            break;
        case MODEL_FORMAT::YOLOE:
            if (yolo_postprocess::decodeRaw(data, dim1, dim2, frame, model_shape_, class_list,
                                            score_threshold_, nms_threshold_) ||
                yolo_postprocess::decodeEndToEnd(data, dim1, dim2, frame, model_shape_, class_list,
                                                 score_threshold_, nms_threshold_)) {
                return;
            }
            break;
        default:
            break;
    }

    yolo_postprocess::warnUnsupportedShape("OpenCV", {1, dim1, dim2});
}
