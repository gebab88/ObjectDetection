#ifndef DETECTION_HANDLER_HPP
#define DETECTION_HANDLER_HPP
#include <opencv2/opencv.hpp>

#include <vector>
#include <string>
#include <fstream>
#include <filesystem>


class Detection{
    public:
        Detection(  float score_threshold,
                    cv::Size2f model_shape,
                    const std::string &model_file);
        virtual ~Detection()  = default;

        void load_class_list(const std::string &class_file);
        virtual void detect(cv::Mat &frame) = 0;

    protected:
        std::string model_file_;
        cv::Mat blob_;
        float score_threshold_;
        cv::Mat resized_;
        float* data_;
        cv::Size2f model_shape_;
        std::vector<std::string> class_list;

    private:
        // void change_modell_file();
};
#endif // DETECTION_HANDLER_HPP