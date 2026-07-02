#ifndef VIDEOHANDLER_HPP
#define VIDEOHANDLER_HPP    
#include <opencv2/opencv.hpp>

class VideoHandler {
    public:
        float fps = 0.0f;

        explicit VideoHandler(std::string output_path);
        ~VideoHandler();

        static float sanitizedFps(double reported_fps);

        void crop_frame( cv::Mat &frame );
        void printVideoProperties();
        bool open_file(const std::string &input_path);
        bool open_webcam(const int &camera_index);
        bool read( cv::Mat &frame );
        void showFrame( const std::string &windowName,  const cv::Mat &frame ) ;
        bool write( cv::Mat &frame );

    private:
        bool openWriter(const cv::Size &output_size);

        std::string output_path_;

        cv::VideoCapture cap_;
        cv::VideoWriter output_;
};
#endif // VIDEOHANDLER_HPP
