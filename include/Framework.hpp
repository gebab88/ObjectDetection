#ifndef OPENCV_OBJECT_DETECTION_FRAMEWORK_HPP
#define OPENCV_OBJECT_DETECTION_FRAMEWORK_HPP

#include <filesystem>
#include <string>

enum class FRAMEWORK {  OpenCV, 
                        OpenVINO, 
                        ONNXRuntime 
};

enum class MODEL_FORMAT {
    YOLO12,   // klassisches YOLO-Format: [batch, fields, detections]
    YOLO26,   // end-to-end Format:       [batch, detections, 6]
    YOLOE,    // open-vocabulary YOLOE export; prompts/classes are baked in
    Unknown
};

// Hilfsfunktion (einmal, wiederverwendbar):
inline MODEL_FORMAT detectModelFormat(const std::string& model_file) {
    const auto fname = std::filesystem::path(model_file).filename().string();
    if (fname == "yolo12m.onnx" || fname == "yolo12x.onnx") return MODEL_FORMAT::YOLO12;
    if (fname == "yolo26m.onnx" || fname == "yolo26x.onnx") return MODEL_FORMAT::YOLO26;
    if (fname.rfind("yoloe", 0) == 0) return MODEL_FORMAT::YOLOE;
    return MODEL_FORMAT::Unknown;
};

inline const char* modelFormatName(const MODEL_FORMAT model_format) {
    switch (model_format) {
        case MODEL_FORMAT::YOLO12: return "YOLO12";
        case MODEL_FORMAT::YOLO26: return "YOLO26";
        case MODEL_FORMAT::YOLOE: return "YOLOE";
        case MODEL_FORMAT::Unknown: return "Unknown";
    }
    return "Unknown";
}

#endif //OPENCV_OBJECT_DETECTION_FRAMEWORK_HPP
