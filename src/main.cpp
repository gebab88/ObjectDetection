#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/tracking.hpp>
#include <iostream>
#include <fstream>
#include <csignal>

#include "VideoHandler.hpp"
#include "Framework.hpp"
#include "Timer.hpp"
#include "Detection_OpenCV.hpp"
#include "Detection_ORT.hpp"

#ifdef __APPLE__
    #include "Detection_OpenVINO.hpp"
#elif __linux__
    // Linux-spezifische Includes
#endif

const float SCORE_THRESHOLD = 0.4;
const float NMS_THRESHOLD = 0.5;
const cv::Size2f MODELSHAPE(cv::Size(640,640));
const int FILE_OR_WEBCAM = 1; //1: FILE or 2: WEBCAM
const bool SHOW_FRAMES = false;
const std::string VIDEO_INPUT_FILE ="./test.mov";
const std::string CLASS_NAMES_FILE ="./coco.txt";
// const std::string MODEL_FILE = "../yolo12m.onnx";
const std::string MODEL_FILE = "./yolo26x.onnx";
const bool USE_CUDA = false; // Set to true to use CUDA backend (if available)
FRAMEWORK backend = FRAMEWORK::ONNXRuntime; //OpenCV; OpenVINO

bool ret = true;

// Signal handler function
void signalHandler(int sig) {
    fflush(stdout);
    std::cout << "Interrupted by user: Signal " << sig << std::endl;
    ret = false;
}

int main() {
    std::cout << cv::getBuildInformation() << std::endl;

    Timer timer;
    timer.start();
    if (MODEL_FILE=="./yolo26m.onnx" or MODEL_FILE=="./yolo26x.onnx") {
        std::list<FRAMEWORK> supportedBackends = {
            FRAMEWORK::ONNXRuntime,
            FRAMEWORK::OpenVINO
        };
        if ( std::find(supportedBackends.begin(), supportedBackends.end(), backend ) == supportedBackends.end() ) {
            return 1;
        };
    }
    VideoHandler video("output.mp4");

    std::unique_ptr<Detection> detection;
    if (backend == FRAMEWORK::OpenCV) {
        detection = std::make_unique<Detection_OpenCV>(SCORE_THRESHOLD, MODELSHAPE, MODEL_FILE, NMS_THRESHOLD, USE_CUDA);
    } else if (backend ==  FRAMEWORK::ONNXRuntime) {
        detection = std::make_unique<Detection_ORT>(SCORE_THRESHOLD, MODELSHAPE, MODEL_FILE);
    } 
    #ifdef __APPLE__
    else if (backend == FRAMEWORK::OpenVINO) {
        detection = std::make_unique<Detection_OpenVINO>(SCORE_THRESHOLD, MODELSHAPE, MODEL_FILE);
    }
    #endif
    else {
        std::cerr << "Unsupported backend for this platform." << std::endl;
        exit(EXIT_FAILURE);
    }

    switch (FILE_OR_WEBCAM ) {
        case 1:
            video.open_file(VIDEO_INPUT_FILE);
            break;
        case 2:
            video.open_webcam(0); // Default camera index 0
            break;
        default:
            std::cerr << "Please select mode: VIDEO or WEBCAM?" << std::endl;
            exit(EXIT_FAILURE);
    }

    video.set_video_writer();

    // Load class names
    detection->load_class_list(CLASS_NAMES_FILE);

    cv::Mat frame;
    while (ret) {
        // Read the next frame
        ret = video.read(frame);
        signal(SIGINT, signalHandler);
        if (ret) {
            std::cout << "Read a new frame: " << frame.cols << "x" << frame.rows << std::endl;
            video.crop_frame(frame);
            // track(formatted_frame, tracker, trackingBox);
            // Run detection on the input image
            detection->detect(frame);
            if (SHOW_FRAMES) {
                video.showFrame("Object Detector", frame);
            }
            video.write(frame);  // Write the processed frame to the output video
        } else {
            std::cout << "No more frames to read or error occurred." << std::endl;
            break;
        }
    }
    timer.stop();
    std::cout << "Total execution time: " << timer.getElapsedTime() << " seconds" << std::endl;
    return 0;
}