#ifndef VIDEOHANDLER_HPP
#define VIDEOHANDLER_HPP    
#include <opencv2/opencv.hpp>

#include <string>

class VideoHandler {
    public:
        float fps = 0.0f;

        explicit VideoHandler(std::string output_path);
        ~VideoHandler();

        static float sanitizedFps(double reported_fps);

        // Preferred OpenCV video capture backend for files and webcams. On Apple
        // platforms the FFmpeg-backed capture crashes (SIGSEGV inside
        // CvCapture_FFMPEG::get_rotation_angle while opening .MOV files with the
        // Homebrew/MacPorts OpenCV builds), so use the native AVFoundation backend.
        // Other platforms keep OpenCV's automatic selection. Inline so code that
        // does not link VideoHandler.cpp (e.g. the inference test) can use it.
        static int preferredCaptureBackend() {
#if defined(__APPLE__)
            return static_cast<int>(cv::CAP_AVFOUNDATION);
#else
            return static_cast<int>(cv::CAP_ANY);
#endif
        }

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
