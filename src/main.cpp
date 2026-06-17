#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <csignal>
#include <filesystem>
#include <list>
#include <memory>


#include "config_loader.hpp"

#include "VideoHandler.hpp"
#include "Framework.hpp"
#include "Source.hpp"
#include "Timer.hpp"
#include "Detection_OpenCV.hpp"

#ifdef HAVE_ONNX_RUNTIME
#include "Detection_ORT.hpp"
#endif

#ifdef HAVE_OPENVINO
#include "Detection_OpenVINO.hpp"
#endif

volatile sig_atomic_t keep_running = 1;

void signalHandler(int /*sig*/) {
    keep_running = 0;
}

int main() {
    // std::cout << cv::getBuildInformation() << std::endl;

    // ── Load config ───────────────────────────────────────────────────────────
#if defined(USE_YAML_CONFIG)
    const Config cfg = loadConfigYAML("config.yaml");
#else
    const Config cfg;  // use default values from struct
#endif

    Timer timer;
    timer.start();

    const auto model_format = detectModelFormat(cfg.model_file);
    std::list<FRAMEWORK> supportedBackends;

    if (model_format == MODEL_FORMAT::YOLO12 || model_format == MODEL_FORMAT::YOLOE) {
        supportedBackends.push_back(FRAMEWORK::OpenCV);
        #ifdef HAVE_ONNX_RUNTIME
        supportedBackends.push_back(FRAMEWORK::ONNXRuntime);
        #endif
        #ifdef HAVE_OPENVINO
        supportedBackends.push_back(FRAMEWORK::OpenVINO);
        #endif
    } else if (model_format == MODEL_FORMAT::YOLO26) {
        #ifdef HAVE_ONNX_RUNTIME
        supportedBackends.push_back(FRAMEWORK::ONNXRuntime);
        #endif
        #ifdef HAVE_OPENVINO
        supportedBackends.push_back(FRAMEWORK::OpenVINO);
        #endif
    } else {
        std::cerr << "Unknown model format. Please check your model file name." << std::endl;
        return 1;
    }

    if (std::find(supportedBackends.begin(), supportedBackends.end(), cfg.framework) == supportedBackends.end()) {
        std::cerr << "Selected backend does not support " << modelFormatName(model_format) << " in this build." << std::endl;
        return 1;
    }

    VideoHandler video(cfg.video_output_file);

    std::unique_ptr<Detection> detection;
    try {
        if (cfg.framework == FRAMEWORK::OpenCV) {
            detection = std::make_unique<Detection_OpenCV>(cfg.score_threshold, cfg.model_shape, cfg.model_file, cfg.nms_threshold, cfg.use_cuda);
        }
        #ifdef HAVE_ONNX_RUNTIME
        else if (cfg.framework == FRAMEWORK::ONNXRuntime) {
            detection = std::make_unique<Detection_ORT>(cfg.score_threshold, cfg.model_shape, cfg.model_file, cfg.nms_threshold);
        }
        #endif

        #ifdef HAVE_OPENVINO
        else if (cfg.framework == FRAMEWORK::OpenVINO) {
            detection = std::make_unique<Detection_OpenVINO>(cfg.score_threshold, cfg.model_shape, cfg.model_file, cfg.nms_threshold);
        }
        #endif

        else {
            std::cerr << "Unsupported backend for this platform." << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialise the detection backend (model '" << cfg.model_file
                  << "'): " << e.what() << std::endl;
        return 1;
    }

    bool source_opened = false;
    if (cfg.source == SOURCE::VIDEO) {
        source_opened = video.open_file(cfg.video_input_file);
    } else if (cfg.source == SOURCE::WEBCAM) {
        source_opened = video.open_webcam(cfg.webcam_index);
    } else {
        std::cerr << "Please select a valid source: VIDEO or WEBCAM?" << std::endl;
        return 1;
    }
    if (!source_opened) {
        return 1;
    }

    if (!video.set_video_writer()) {
        return 1;
    }
    detection->load_class_list(cfg.class_names_file);

    cv::Mat frame;
    signal(SIGINT, signalHandler);
    while (keep_running && video.read(frame)) {
        std::cout << "Read a new frame: " << frame.cols << "x" << frame.rows << std::endl;
        video.crop_frame(frame);
        detection->detect(frame);
        if (cfg.show_frames) {
            video.showFrame("Object Detector", frame);
        }
        video.write(frame);
    }

    std::cout << "No more frames to read or error occurred." << std::endl;
    timer.stop();
    std::cout << "Total execution time: " << timer.getElapsedTime() << " seconds" << std::endl;
    return 0;
}
