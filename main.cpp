#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/tracking.hpp>
#include <iostream>
#include <fstream>
#include <csignal>

#include "Detection_OpenCV.hpp"
#include "Detection_OpenVINO.hpp"
#include "Detection_ORT.hpp"
#include "VideoHandler.hpp"
#include "Framework.hpp"

using namespace cv;
using namespace std;

const float SCORE_THRESHOLD = 0.4;
const float NMS_THRESHOLD = 0.5;
const Size2f MODELSHAPE(Size(640,640));
const int FILE_OR_WEBCAM = 1; //1: FILE or 2: WEBCAM
const bool SHOW_FRAMES = false;
const string VIDEO_INPUT_FILE ="../test.mov";
const string CLASS_NAMES_FILE ="../coco.txt";
// const string MODEL_FILE = "../yolo12m.onnx";
const string MODEL_FILE = "../yolo26x.onnx";
const bool USE_CUDA = false; // Set to true to use CUDA backend (if available)
FRAMEWORK backend = FRAMEWORK::OpenVINO; //OpenCV; OpenVINO

bool ret = true;

// Signal handler function
void signalHandler(int sig) {
    fflush(stdout);
    cout << "Interrupted by user: Signal " << sig << endl;
    ret = false;
}

int main() {
    if (MODEL_FILE=="../yolo26m.onnx" or MODEL_FILE=="../yolo26x.onnx") {
        list<FRAMEWORK> supportedBackends = {
            FRAMEWORK::ONNXRuntime,
            FRAMEWORK::OpenVINO
        };
        if ( find(supportedBackends.begin(), supportedBackends.end(), backend ) == supportedBackends.end() ) {
            return 1;
        };
    }
    VideoHandler video("output.avi");

    std::unique_ptr<Detection> detection;
    if (backend == FRAMEWORK::OpenCV) {
        detection = std::make_unique<Detection_OpenCV>(SCORE_THRESHOLD, MODELSHAPE, MODEL_FILE, NMS_THRESHOLD, USE_CUDA);
    } else if (backend == FRAMEWORK::OpenVINO) {
        detection = std::make_unique<Detection_OpenVINO>(SCORE_THRESHOLD, MODELSHAPE, MODEL_FILE);
    } else if (backend ==  FRAMEWORK::ONNXRuntime) {
        detection = std::make_unique<Detection_ORT>(SCORE_THRESHOLD, MODELSHAPE, MODEL_FILE);
    }

    switch (FILE_OR_WEBCAM ) {
        case 1:
            video.open_file(VIDEO_INPUT_FILE);
            break;
        case 2:
            video.open_webcam(0); // Default camera index 0
            break;
        default:
            cerr << "Please select mode: VIDEO or WEBCAM?" << endl;
            exit(EXIT_FAILURE);
    }

    video.set_video_writer();

    // Load class names
    detection->load_class_list(CLASS_NAMES_FILE);

    Mat frame;
    while (ret) {
        // Read the next frame
        ret = video.read(frame);
        signal(SIGINT, signalHandler);
        if (ret) {
            cout << "Read a new frame: " << frame.cols << "x" << frame.rows << endl;
            video.crop_frame(frame);
            // track(formatted_frame, tracker, trackingBox);
            // Run detection on the input image
            detection->detect(frame);
            if (SHOW_FRAMES) {
                video.showFrame("Object Detector", frame);
            }
            video.write(frame);  // Write the processed frame to the output video
        } else {
            cout << "No more frames to read or error occurred." << endl;
            break;
        }
    }
    return 0;
}