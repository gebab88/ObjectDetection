#ifndef OPENCV_OBJECT_DETECTION_FRAMEWORK_HPP
#define OPENCV_OBJECT_DETECTION_FRAMEWORK_HPP

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <string>

enum class FRAMEWORK {  OpenCV,
                        OpenVINO,
                        ONNXRuntime,
                        CoreAI,
#ifdef HAVE_HAILO
                        Hailo
#endif
};

enum class MODEL_FORMAT {
    YOLO12,   // classic YOLO layout:  [batch, fields, detections]
    YOLO26,   // end-to-end layout:    [batch, detections, 6]
    YOLOE,    // open-vocabulary YOLOE export; prompts/classes are baked in
#ifdef HAVE_HAILO
    HEF,      // Hailo Executable Format (Raspberry Pi backend)
#endif
    Unknown
};

// Detects the model layout from the file name. Substring matching (instead of
// exact file names) tolerates arbitrary variants/sizes and renames, e.g.
// yolo12s, yolo12n.onnx, yolo26m_v2.onnx, yoloe-26m-seg-pf.onnx.
inline MODEL_FORMAT detectModelFormat(const std::string& model_file) {
    std::string fname = std::filesystem::path(model_file).filename().string();
    std::string ext = std::filesystem::path(model_file).extension().string();
    std::transform(fname.begin(), fname.end(), fname.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (ext == ".hef") {
#ifdef HAVE_HAILO
        return MODEL_FORMAT::HEF;
#else
        // A YOLO-named HEF must not fall through to an ONNX decoder.
        return MODEL_FORMAT::Unknown;
#endif
    }
    if (fname.find("yolo12") != std::string::npos) return MODEL_FORMAT::YOLO12;
    if (fname.find("yolo26") != std::string::npos) return MODEL_FORMAT::YOLO26;
    if (fname.find("yoloe") != std::string::npos) return MODEL_FORMAT::YOLOE;
    return MODEL_FORMAT::Unknown;
}

inline const char* modelFormatName(const MODEL_FORMAT model_format) {
    switch (model_format) {
        case MODEL_FORMAT::YOLO12: return "YOLO12";
        case MODEL_FORMAT::YOLO26: return "YOLO26";
        case MODEL_FORMAT::YOLOE: return "YOLOE";
#ifdef HAVE_HAILO
        case MODEL_FORMAT::HEF: return "HEF";
#endif
        case MODEL_FORMAT::Unknown: return "Unknown";
    }
    return "Unknown";
}

inline const char* frameworkName(const FRAMEWORK framework) {
    switch (framework) {
        case FRAMEWORK::OpenCV:      return "OpenCV";
        case FRAMEWORK::OpenVINO:    return "OpenVINO";
        case FRAMEWORK::ONNXRuntime: return "ONNXRuntime";
        case FRAMEWORK::CoreAI:      return "CoreAI";
#ifdef HAVE_HAILO
        case FRAMEWORK::Hailo:       return "Hailo";
#endif
    }
    return "Unknown";
}

// Maps a Core AI (.aimodel) asset to the equivalent ONNX asset for backends that
// consume ONNX-format models (ONNX Runtime, OpenVINO, OpenCV DNN). The Core AI
// export writes <stem>.aimodel next to the source <stem>.onnx, and those backends
// cannot run the .aimodel package, so an .aimodel selection resolves to the plain
// ONNX file. Anything else passes through unchanged and is handled elsewhere.
inline std::string onnxFallbackModelPath(const std::string& model_file) {
    const std::filesystem::path p(model_file);
    std::string ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (ext != ".aimodel") {
        return model_file;
    }
    return (p.parent_path() / (p.stem().string() + ".onnx")).string();
}

#endif //OPENCV_OBJECT_DETECTION_FRAMEWORK_HPP
