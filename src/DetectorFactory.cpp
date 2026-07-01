#include "DetectorFactory.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

#include "config_loader.hpp"
#include "Detection.hpp"
#include "Detection_OpenCV.hpp"

#ifdef HAVE_ONNX_RUNTIME
#include "Detection_ORT.hpp"
#endif

#ifdef HAVE_OPENVINO
#include "Detection_OpenVINO.hpp"
#endif

std::unique_ptr<Detection> createDetector(const Config& cfg) {
    const MODEL_FORMAT format = detectModelFormat(cfg.model_file);

    const std::vector<FRAMEWORK> supported = supportedBackendsFor(format);
    if (supported.empty()) {
        throw std::runtime_error("Unknown model format for '" + cfg.model_file +
                                 "'. Please check the model file name.");
    }
    if (std::find(supported.begin(), supported.end(), cfg.framework) == supported.end()) {
        throw std::runtime_error(std::string(frameworkName(cfg.framework)) +
                                 " backend does not support " + modelFormatName(format) + " models.");
    }

    switch (cfg.framework) {
        case FRAMEWORK::OpenCV:
            return std::make_unique<Detection_OpenCV>(cfg.score_threshold, cfg.model_shape,
                                                      cfg.model_file, cfg.nms_threshold, cfg.use_cuda);
        case FRAMEWORK::ONNXRuntime:
#ifdef HAVE_ONNX_RUNTIME
            return std::make_unique<Detection_ORT>(cfg.score_threshold, cfg.model_shape,
                                                   cfg.model_file, cfg.nms_threshold);
#else
            break;
#endif
        case FRAMEWORK::OpenVINO:
#ifdef HAVE_OPENVINO
            return std::make_unique<Detection_OpenVINO>(cfg.score_threshold, cfg.model_shape,
                                                        cfg.model_file, cfg.nms_threshold);
#else
            break;
#endif
    }

    throw std::runtime_error(std::string(frameworkName(cfg.framework)) +
                             " backend was not compiled into this binary.");
}
