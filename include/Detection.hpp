#ifndef DETECTION_HANDLER_HPP
#define DETECTION_HANDLER_HPP
#include <opencv2/opencv.hpp>

#include <vector>
#include <string>
#include <fstream>
#include <filesystem>
#include <future>

#include "Framework.hpp"


class Detection{
    public:
        Detection(  float score_threshold,
                    cv::Size2f model_shape,
                    const std::string &model_file);
        virtual ~Detection()  = default;

        bool load_class_list(const std::string &class_file);
        virtual void detect(cv::Mat &frame) = 0;

        // Frames the backend can work on at the same time (see detect_async).
        virtual std::size_t max_frames_in_flight() const { return 1; }

        // Starts detecting objects in frame and drawing them into it. The
        // future becomes ready when the frame is done (and rethrows detection
        // errors); until then the frame must not be touched. The default runs
        // detect() right away.
        virtual std::future<void> detect_async(cv::Mat frame);

    protected:
        std::string model_file_;
        MODEL_FORMAT model_format_;
        cv::Mat blob_;
        float score_threshold_;
        cv::Size2f model_shape_;
        std::vector<std::string> class_list_;
};
#endif // DETECTION_HANDLER_HPP