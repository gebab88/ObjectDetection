#include "VideoHandler.hpp"

// int VideoHandler::fourcc = static_cast<int>(cap.get(cv::CAP_PROP_FOURCC));

VideoHandler::VideoHandler(const string output_path) : output_path_(output_path){
}
VideoHandler::~VideoHandler(){
        cap_.release();
        output_.release();
        cout << "Released video resources." << endl;
}

void VideoHandler::open_file(const string &input_path) {
    // Implementation to open video file
    cap_.open(input_path);
    if (!cap_.isOpened()) {
        cerr << "Error: Could not open video file: " << input_path << endl;
    } else {
        cout << "Successfully opened video file: " << input_path << endl;
    }
    fps = static_cast<float>(cap_.get(CAP_PROP_FPS));
    frame_width = static_cast<int>(cap_.get(CAP_PROP_FRAME_WIDTH));
    frame_height = static_cast<int>(cap_.get(CAP_PROP_FRAME_HEIGHT));
    printVideoProperties();
}

void VideoHandler::open_webcam(const int &camera_index) {
    // Implementation to open webcam
    cap_.open(camera_index);
    if (!cap_.isOpened()) {
        cerr << "Error: Could not open webcam with index: " << camera_index << endl;
    } else {
        cout << "Successfully opened webcam with index: " << camera_index << endl;
    }
    fps = 5.0;
    frame_width = static_cast<int>(cap_.get(CAP_PROP_FRAME_WIDTH));
    frame_height = static_cast<int>(cap_.get(CAP_PROP_FRAME_HEIGHT));
}

void VideoHandler::set_video_writer() {
    int min_side_len = min(frame_height, frame_width);
    bool ret = output_.open(output_path_, codec_, fps, Size(min_side_len, min_side_len));
    if (!ret) {
        cerr << "Error: Could not open video writer with path: " << output_path_ << endl;
        exit(EXIT_FAILURE);
    } else {
        cout << "Successfully opened video writer with path: " << output_path_ << endl;
    }
}

void VideoHandler::crop_frame( Mat &frame ) {
        //Crop to square frame or yolov format
        int min_side_len = min(frame_height, frame_width);
        int excess_len_side;
        if (frame_width > frame_height) {
            excess_len_side = (frame_width - min_side_len) / 2;
            frame = frame(Range(0,min_side_len), Range(excess_len_side,min_side_len + excess_len_side));
        } else {
            excess_len_side = (frame_height - min_side_len) / 2;
            frame = frame(Range(excess_len_side,min_side_len + excess_len_side), Range(0,min_side_len));
        }
    }

void VideoHandler::printVideoProperties(){
    cout << "Video Properties:" << endl;
    cout << "Frame Width: " << cap_.get(CAP_PROP_FRAME_WIDTH) << endl;
    cout << "Frame Height: " << cap_.get(CAP_PROP_FRAME_HEIGHT) << endl;
    cout << "FPS: " << cap_.get(CAP_PROP_FPS) << endl;
    cout << "Frame Count: " << cap_.get(CAP_PROP_FRAME_COUNT) << endl;
    cout << "Format: " << cap_.get(CAP_PROP_FORMAT) << endl;
    cout << "Mode: " << cap_.get(CAP_PROP_MODE) << endl;
}

void VideoHandler::showFrame( const string &windowName, const Mat &frame ) {
        imshow(windowName, frame);
        waitKey(1);
        destroyAllWindows();  // Close the window after displaying each frame
}

void VideoHandler::write( Mat &frame ) {
    output_.write(frame);
}

bool VideoHandler::read( Mat &frame ) {
        return cap_.read(frame);
}