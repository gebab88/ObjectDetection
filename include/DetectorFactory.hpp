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
            // Core AI serves YOLO26 (end-to-end or raw-head exports; the shared
            // decoder picks the head via the output shape, host-side NMS for raw).
            return {FRAMEWORK::ONNXRuntime, FRAMEWORK::OpenVINO, FRAMEWORK::CoreAI};
#ifdef HAVE_HAILO
        case MODEL_FORMAT::HEF:
            return {FRAMEWORK::Hailo};
#endif
        case MODEL_FORMAT::Unknown:
            return {};
    }
    return {};
}

// Builds the detector selected in cfg. Throws std::runtime_error with a clear
// message when the model format is unknown, the chosen backend cannot handle
// that format, or no backend can be built. May also propagate backend exceptions
// (e.g. a missing or corrupt model file). The Core AI backend is the exception:
// when it is not compiled into this binary or fails during initialisation, the
// factory automatically falls back to the ONNX Runtime backend instead of throwing.
std::unique_ptr<Detection> createDetector(const Config& cfg);

#endif // OBJECTDETECTION_DETECTORFACTORY_HPP
