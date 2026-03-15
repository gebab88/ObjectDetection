#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <string>
#include <iostream>
#include <opencv2/core.hpp>
#include "Framework.hpp"

// ─── Config struct ────────────────────────────────────────────────────────────
struct Config {
    // Detection
    float        score_threshold  = 0.4f;
    float        nms_threshold    = 0.5f;
    
    // Model
    std::string  model_file       = "./yolo26x.onnx";
    std::string  class_names_file = "./coco.txt";
    cv::Size2f   model_shape      = cv::Size(640, 640);

    // Backend
    FRAMEWORK    framework        = FRAMEWORK::OpenVINO;
    bool         use_cuda         = false;

    // Input / Output
    int          source           = 1;
    int          webcam_index     = 0;
    std::string  video_input_file = "./test.mov";
    std::string  video_output_file= "./output.mp4";

    // Display
    bool         show_frames      = false;
};

// ─── Helper: string → FRAMEWORK ──────────────────────────────────────────────
inline FRAMEWORK frameworkFromString(const std::string& s) {
    if (s == "OpenCV")      return FRAMEWORK::OpenCV;
    if (s == "OpenVINO")    return FRAMEWORK::OpenVINO;
    if (s == "ONNXRuntime") return FRAMEWORK::ONNXRuntime;
    std::cerr << "Unknown framework: " << s << " – falling back to OpenCV." << std::endl;
    return FRAMEWORK::OpenCV;
}

// ═════════════════════════════════════════════════════════════════════════════
// YAML  (requires yaml-cpp – https://github.com/jbeder/yaml-cpp)
//   CMake:  find_package(yaml-cpp REQUIRED)
//           target_link_libraries(... yaml-cpp)
// ═════════════════════════════════════════════════════════════════════════════
#ifdef USE_YAML_CONFIG
#include <yaml-cpp/yaml.h>

inline Config loadConfigYAML(const std::string& path) {
    YAML::Node y;
    try {
        y = YAML::LoadFile(path);
    } catch (const YAML::Exception& e) {
        std::cerr << "Could not open YAML config: " << e.what() << std::endl;
        return {};
    }

    Config cfg;
    cfg.score_threshold   = y["detection"]["score_threshold"].as<float>();
    cfg.nms_threshold     = y["detection"]["nms_threshold"].as<float>();
    

    cfg.model_file        = y["model"]["model_file"].as<std::string>();
    cfg.class_names_file  = y["model"]["class_names_file"].as<std::string>();
    cfg.model_shape       = cv::Size(y["model"]["model_width"].as<int>(),
                                     y["model"]["model_height"].as<int>());

    cfg.framework         = frameworkFromString(y["backend"]["framework"].as<std::string>());
    cfg.use_cuda          = y["backend"]["use_cuda"].as<bool>();

    cfg.source            = y["input"]["source"].as<int>();
    cfg.webcam_index      = y["input"]["webcam_index"].as<int>();
    cfg.video_input_file  = y["input"]["video_input_file"].as<std::string>();
    cfg.video_output_file = y["input"]["video_output_file"].as<std::string>();

    cfg.show_frames       = y["display"]["show_frames"].as<bool>();
    return cfg;
}
#endif // USE_YAML_CONFIG
#endif // CONFIG_HPP