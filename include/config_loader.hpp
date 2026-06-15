#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <string>
#include <iostream>
#include <opencv2/core.hpp>

#include "Framework.hpp"
#include "Source.hpp"

// ─── Config struct ────────────────────────────────────────────────────────────
struct Config {
    // Detection
    float        score_threshold  = 0.4f;
    float        nms_threshold    = 0.5f;
    
    // Model
    std::string  model_file       = "yoloe-26m-seg-pf.onnx";
    std::string  class_names_file = "yoloe_pf_classes.txt";
    cv::Size2f   model_shape      = cv::Size(640, 640);

    // Backend
#if defined(HAVE_ONNX_RUNTIME)
    FRAMEWORK    framework        = FRAMEWORK::ONNXRuntime;
#elif defined(HAVE_OPENVINO)
    FRAMEWORK    framework        = FRAMEWORK::OpenVINO;
#else
    FRAMEWORK    framework        = FRAMEWORK::OpenCV;
#endif
    bool         use_cuda         = false;

    // Input / Output
    SOURCE       source           = SOURCE::VIDEO;
    int          webcam_index     = 0;
    std::string  video_input_file = "test.mov";
    std::string  video_output_file= "output.mp4";

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
};

// ─── Helper: string → SOURCE ──────────────────────────────────────────────
inline SOURCE sourceFromString(const std::string& s) {
    if (s == "VIDEO")   return SOURCE::VIDEO;
    if (s == "WEBCAM")  return SOURCE::WEBCAM;
    std::cerr << "Unknown source: " << s << " – falling back to VIDEO." << std::endl;
    return SOURCE::VIDEO;
};


// ═════════════════════════════════════════════════════════════════════════════
// YAML  (requires yaml-cpp – https://github.com/jbeder/yaml-cpp)
//   CMake:  find_package(yaml-cpp REQUIRED)
//           target_link_libraries(... yaml-cpp)
// ═════════════════════════════════════════════════════════════════════════════
#ifdef USE_YAML_CONFIG
#include <yaml-cpp/yaml.h>

inline Config loadConfigYAML(const std::string& path) {
    Config cfg;  // start from defaults; every field below falls back to these

    YAML::Node y;
    try {
        y = YAML::LoadFile(path);
    } catch (const YAML::Exception& e) {
        std::cerr << "Could not open YAML config: " << e.what()
                  << " – using default configuration." << std::endl;
        return cfg;
    }

    // Each lookup falls back to the existing default when the key is missing or
    // null, so a partial config.yaml degrades gracefully instead of throwing an
    // uncaught YAML::Exception. A single try/catch covers any remaining
    // type-conversion errors (e.g. a non-numeric threshold).
    try {
        cfg.score_threshold   = y["detection"]["score_threshold"].as<float>(cfg.score_threshold);
        cfg.nms_threshold     = y["detection"]["nms_threshold"].as<float>(cfg.nms_threshold);

        cfg.model_file        = y["model"]["model_file"].as<std::string>(cfg.model_file);
        cfg.class_names_file  = y["model"]["class_names_file"].as<std::string>(cfg.class_names_file);
        cfg.model_shape       = cv::Size(
            y["model"]["model_width"].as<int>(static_cast<int>(cfg.model_shape.width)),
            y["model"]["model_height"].as<int>(static_cast<int>(cfg.model_shape.height)));

        if (y["backend"]["framework"])
            cfg.framework     = frameworkFromString(y["backend"]["framework"].as<std::string>());
        cfg.use_cuda          = y["backend"]["use_cuda"].as<bool>(cfg.use_cuda);

        if (y["input"]["source"])
            cfg.source        = sourceFromString(y["input"]["source"].as<std::string>());
        cfg.webcam_index      = y["input"]["webcam_index"].as<int>(cfg.webcam_index);
        cfg.video_input_file  = y["input"]["video_input_file"].as<std::string>(cfg.video_input_file);
        cfg.video_output_file = y["input"]["video_output_file"].as<std::string>(cfg.video_output_file);

        cfg.show_frames       = y["display"]["show_frames"].as<bool>(cfg.show_frames);
    } catch (const YAML::Exception& e) {
        std::cerr << "Invalid value in YAML config: " << e.what()
                  << " – falling back to defaults for the remaining fields." << std::endl;
    }
    return cfg;
}
#endif // USE_YAML_CONFIG
#endif // CONFIG_HPP
