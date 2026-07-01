#ifndef OBJECTDETECTION_DETECTORFACTORY_HPP
#define OBJECTDETECTION_DETECTORFACTORY_HPP

#include <memory>
#include <vector>

#include "Framework.hpp"

struct Config;
class Detection;

// Backends that can decode a given model format, independent of which backends
// were compiled into this binary. Pure function: it encodes only format
// compatibility (e.g. OpenCV's DNN module cannot load the end-to-end YOLO26
// graph), so it is unit-tested directly without needing the backend libraries.
inline std::vector<FRAMEWORK> supportedBackendsFor(MODEL_FORMAT format) {
    switch (format) {
        case MODEL_FORMAT::YOLO12:
        case MODEL_FORMAT::YOLOE:
            return {FRAMEWORK::OpenCV, FRAMEWORK::ONNXRuntime, FRAMEWORK::OpenVINO};
        case MODEL_FORMAT::YOLO26:
            return {FRAMEWORK::ONNXRuntime, FRAMEWORK::OpenVINO};
        case MODEL_FORMAT::Unknown:
            return {};
    }
    return {};
}

// Builds the detector selected in cfg. Throws std::runtime_error with a clear
// message when the model format is unknown, the chosen backend cannot handle
// that format, or the backend was not compiled into this binary. May also
// propagate backend exceptions (e.g. a missing or corrupt model file).
std::unique_ptr<Detection> createDetector(const Config& cfg);

#endif // OBJECTDETECTION_DETECTORFACTORY_HPP
