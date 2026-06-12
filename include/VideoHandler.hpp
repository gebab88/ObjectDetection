#ifndef VIDEOHANDLER_HPP
#define VIDEOHANDLER_HPP    
#include <opencv2/opencv.hpp>

class VideoHandler {
    public:
        float fps;
        int frame_width;
        int frame_height;

        VideoHandler(std::string output_path);
        ~VideoHandler();

        void crop_frame( cv::Mat &frame );
        void printVideoProperties();
        void open_file(const std::string &input_path);
        void open_webcam(const int &camera_index);
        bool read( cv::Mat &frame );
        void showFrame( const std::string &windowName,  const cv::Mat &frame ) ;
        void write( cv::Mat &frame );
        void set_video_writer();

    private:
        std::string output_path_;
        const int camera_index = 0;

        cv::VideoCapture cap_;
        cv::VideoWriter output_;
};
#endif // VIDEOHANDLER_HPP
