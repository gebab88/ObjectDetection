#include "VideoHandler.hpp"

// int VideoHandler::fourcc = static_cast<int>(cap.get(cv::CAP_PROP_FOURCC));

VideoHandler::VideoHandler(const std::string output_path) : output_path_(output_path){
}
VideoHandler::~VideoHandler(){
        cap_.release();
        output_.release();
        std::cout << "Released video resources." << std::endl;
}

void VideoHandler::open_file(const std::string &input_path) {
    // Implementation to open video file
    cap_.open(input_path);
    if (!cap_.isOpened()) {
        std::cerr << "Error: Could not open video file: " << input_path << std::endl;
    } else {
        std::cout << "Successfully opened video file: " << input_path << std::endl;
    }
    fps = static_cast<float>(cap_.get(cv::CAP_PROP_FPS));
    frame_width = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
    frame_height = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
    printVideoProperties();
}

void VideoHandler::open_webcam(const int &camera_index) {
    // Implementation to open webcam
    cap_.open(camera_index);
    if (!cap_.isOpened()) {
        std::cerr << "Error: Could not open webcam with index: " << camera_index << std::endl;
    } else {
        std::cout << "Successfully opened webcam with index: " << camera_index << std::endl;
    }
    fps = 5.0;
    frame_width = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
    frame_height = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
}

void VideoHandler::set_video_writer() {
    int min_side_len = std::min(frame_height, frame_width);
    bool ret = output_.open(output_path_, codec_, fps, cv::Size(min_side_len, min_side_len));
    if (!ret) {
        std::cerr << "Error: Could not open video writer with path: " << output_path_ << std::endl;
        exit(EXIT_FAILURE);
    } else {
       std::cout << "Successfully opened video writer with path: " << output_path_ << std::endl;
    }
}

void VideoHandler::crop_frame( cv::Mat &frame ) {
        //Crop to square frame or yolov format
        int min_side_len = std::min(frame_height, frame_width);
        int excess_len_side;
        if (frame_width > frame_height) {
            excess_len_side = (frame_width - min_side_len) / 2;
            frame = frame(cv::Range(0,min_side_len), cv::Range(excess_len_side,min_side_len + excess_len_side));
        } else {
            excess_len_side = (frame_height - min_side_len) / 2;
            frame = frame(cv::Range(excess_len_side,min_side_len + excess_len_side), cv::Range(0,min_side_len));
        }
    }

void VideoHandler::printVideoProperties(){
    std::cout << "Video Properties:" << std::endl;
    std::cout << "Frame Width: " << cap_.get(cv::CAP_PROP_FRAME_WIDTH) << std::endl;
    std::cout << "Frame Height: " << cap_.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;
    std::cout << "FPS: " << cap_.get(cv::CAP_PROP_FPS) << std::endl;
    std::cout << "Frame Count: " << cap_.get(cv::CAP_PROP_FRAME_COUNT) << std::endl;
    std::cout << "Format: " << cap_.get(cv::CAP_PROP_FORMAT) << std::endl;
    std::cout << "Mode: " << cap_.get(cv::CAP_PROP_MODE) << std::endl;
}

void VideoHandler::showFrame( const std::string &windowName, const cv::Mat &frame ) {
        cv::imshow(windowName, frame);
        cv::waitKey(1);
        cv::destroyAllWindows();  // Close the window after displaying each frame
}

void VideoHandler::write( cv::Mat &frame ) {
    output_.write(frame);
}

bool VideoHandler::read( cv::Mat &frame ) {
        return cap_.read(frame);
}