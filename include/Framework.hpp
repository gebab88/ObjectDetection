#ifndef OPENCV_OBJECT_DETECTION_FRAMEWORK_HPP
#define OPENCV_OBJECT_DETECTION_FRAMEWORK_HPP

#include <algorithm>
#include <cctype>
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
// Erkennt das Modellformat anhand des Dateinamens. Substring-Matching (statt
// exakter Dateinamen) erlaubt beliebige Varianten/Groessen und Umbenennungen,
// z. B. yolo12s, yolo12n.onnx, yolo26m_v2.onnx, yoloe-26m-seg-pf.onnx.
inline MODEL_FORMAT detectModelFormat(const std::string& model_file) {
    std::string fname = std::filesystem::path(model_file).filename().string();
    std::transform(fname.begin(), fname.end(), fname.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (fname.find("yolo12") != std::string::npos) return MODEL_FORMAT::YOLO12;
    if (fname.find("yolo26") != std::string::npos) return MODEL_FORMAT::YOLO26;
    if (fname.find("yoloe") != std::string::npos) return MODEL_FORMAT::YOLOE;
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
