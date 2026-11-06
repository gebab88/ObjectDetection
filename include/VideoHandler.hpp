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

        // Closes the output video. For a source opened with open_file(), the
        // source's audio track is then copied into the output (when built with
        // HAVE_AUDIO_PASSTHROUGH); failures only print a warning and keep the
        // video without sound. Safe to call when no frame was written.
        void finish_output();

    private:
        bool openWriter(const cv::Size &output_size);

        std::string output_path_;
        std::string input_file_;  // set by open_file(); empty for webcams

        cv::VideoCapture cap_;
        cv::VideoWriter output_;
};
#endif // VIDEOHANDLER_HPP
