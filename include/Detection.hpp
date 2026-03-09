#ifndef DETECTION_HANDLER_HPP
#define DETECTION_HANDLER_HPP
#include <opencv2/opencv.hpp>

#include <vector>
#include <string>
#include <fstream>
#include <filesystem>

using namespace cv;
using namespace std;


class Detection{
    public:
        Detection(   float score_threshold,
                            Size2f model_shape,
                            const string &model_file);
        virtual ~Detection()  = default;

        void load_class_list(const string &class_file);
        virtual void detect(cv::Mat &frame) = 0;

    protected:
        string model_file_;
        Mat blob_;
        float score_threshold_;
        cv::Mat resized_;
        float* data_;
        Size2f model_shape_;
        vector<string> class_list;

    private:
        void change_modell_file();






};
#endif // DETECTION_HANDLER_HPP