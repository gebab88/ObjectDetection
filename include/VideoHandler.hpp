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
        const std::string output_path_;
        const int camera_index = 0;
        #ifdef __APPLE__
            const int codec_ = cv::VideoWriter::fourcc('a', 'v', 'c', '1');
        #elif __linux__
        // Linux-spezifischer Code
            // const int codec_ = cv::VideoWriter::fourcc('X', 'V', 'I', 'D');
            // const int codec_ = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
            const int codec_ = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
        #endif

        cv::VideoCapture cap_;
        cv::VideoWriter output_;
};
#endif // VIDEOHANDLER_HPP