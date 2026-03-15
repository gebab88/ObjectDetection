#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/tracking.hpp>
#include <iostream>
#include <fstream>
#include <csignal>


#define USE_YAML_CONFIG
#include "config_loader.hpp"

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

volatile sig_atomic_t ret = 1;

void signalHandler(int sig) {
    fflush(stdout);
    std::cout << "Interrupted by user: Signal " << sig << std::endl;
    ret = false;
}

int main() {
    std::cout << cv::getBuildInformation() << std::endl;

    // ── Load config ───────────────────────────────────────────────────────────
#if defined(USE_YAML_CONFIG)
    const Config cfg = loadConfigYAML("./config.yaml");
#else
    const Config cfg;  // use default values from struct
#endif

    Timer timer;
    timer.start();

    if (cfg.model_file == "./yolo26m.onnx" || cfg.model_file == "./yolo26x.onnx") {
        std::list<FRAMEWORK> supportedBackends = {
            FRAMEWORK::ONNXRuntime,
            FRAMEWORK::OpenVINO
        };
        if (std::find(supportedBackends.begin(), supportedBackends.end(), cfg.framework) == supportedBackends.end()) {
            return 1;
        }
    }

    VideoHandler video(cfg.video_output_file);

    std::unique_ptr<Detection> detection;
    if (cfg.framework == FRAMEWORK::OpenCV) {
        detection = std::make_unique<Detection_OpenCV>(cfg.score_threshold, cfg.model_shape, cfg.model_file, cfg.nms_threshold, cfg.use_cuda);
    } else if (cfg.framework == FRAMEWORK::ONNXRuntime) {
        detection = std::make_unique<Detection_ORT>(cfg.score_threshold, cfg.model_shape, cfg.model_file);
    }
    #ifdef __APPLE__
    else if (cfg.framework == FRAMEWORK::OpenVINO) {
        detection = std::make_unique<Detection_OpenVINO>(cfg.score_threshold, cfg.model_shape, cfg.model_file);
    }
    #endif
    else {
        std::cerr << "Unsupported backend for this platform." << std::endl;
        exit(EXIT_FAILURE);
    }

    switch (cfg.source) {
        case 1:
            video.open_file(cfg.video_input_file);
            break;
        case 2:
            video.open_webcam(cfg.webcam_index);
            break;
        default:
            std::cerr << "Please select mode: VIDEO or WEBCAM?" << std::endl;
            exit(EXIT_FAILURE);
    }

    video.set_video_writer();
    detection->load_class_list(cfg.class_names_file);

    cv::Mat frame;
    signal(SIGINT, signalHandler);
    while (ret) {
        ret = video.read(frame);
        if (ret) {
            std::cout << "Read a new frame: " << frame.cols << "x" << frame.rows << std::endl;
            video.crop_frame(frame);
            detection->detect(frame);
            if (cfg.show_frames) {
                video.showFrame("Object Detector", frame);
            }
            video.write(frame);
        } else {
            std::cout << "No more frames to read or error occurred." << std::endl;
            break;
        }
    }

    timer.stop();
    std::cout << "Total execution time: " << timer.getElapsedTime() << " seconds" << std::endl;
    return 0;
}
