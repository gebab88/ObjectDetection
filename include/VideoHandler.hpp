#ifndef VIDEOHANDLER_HPP
#define VIDEOHANDLER_HPP    
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

class VideoHandler {
    public:
        float fps;
        int frame_width;
        int frame_height;
        int min_side_len;
        
        Mat frame;

        VideoHandler(string output_path);
        ~VideoHandler();

        void crop_frame( Mat &frame );
        void printVideoProperties();
        void open_file(const string &input_path);
        void open_webcam(const int &camera_index);
        bool read( Mat &frame );
        void showFrame( const string &windowName,  const Mat &frame ) ;
        void write( Mat &frame );
        void set_video_writer();

    private:
        const string output_path_;
        const int camera_index = 0;
        const int codec_ = VideoWriter::fourcc('M', 'J', 'P', 'G');
        VideoCapture cap_;
        VideoWriter output_;
};
#endif // VIDEOHANDLER_HPP