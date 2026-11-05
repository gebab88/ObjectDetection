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

#ifdef HAVE_CORE_AI
#include "Detection_CoreAI.hpp"
#endif

#ifdef HAVE_HAILO
#include "Detection_Hailo.hpp"
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
#ifdef HAVE_HAILO
        case FRAMEWORK::Hailo:
            return std::make_unique<Detection_Hailo>(cfg.score_threshold, cfg.nms_threshold,
                                                     cfg.model_shape, cfg.model_file);
#endif
        case FRAMEWORK::CoreAI: {
            // Core AI is opt-in (Apple silicon, macOS 27+). When the backend is not
            // compiled into this binary or its initialisation fails (missing .aimodel,
            // Swift runtime, OS version, ...), fall back to the ONNX Runtime backend,
            // which has a CoreML/CPU fallback of its own. A hard failure here would
            // regress the existing ONNX Runtime path for platforms where Core AI is
            // unavailable, so the fallback is automatic and mirrors the documented
            // behaviour. The ONNX Runtime backend consumes ONNX-format assets, so an
            // .aimodel selection is mapped to its ONNX counterpart (yolo26x.aimodel ->
            // yolo26x.onnx) before the session is created instead of handing the
            // .aimodel package to a backend that cannot load it.
#ifdef HAVE_CORE_AI
            try {
                return std::make_unique<Detection_CoreAI>(cfg.score_threshold, cfg.model_shape, cfg.model_file, cfg.nms_threshold);
            } catch (const std::exception& e) {
                std::cerr << "[CoreAI] Backend initialisation failed: " << e.what()
                          << " – falling back to the ONNX Runtime backend." << std::endl;
            }
#else
            std::cerr << "[CoreAI] Backend was not compiled into this binary; falling back to the ONNX Runtime backend." << std::endl;
#endif
#ifdef HAVE_ONNX_RUNTIME
            // Fall back to the ONNX Runtime backend, which has a CoreML/CPU fallback.
            // Map an .aimodel selection to the equivalent ONNX asset first.
            return std::make_unique<Detection_ORT>(cfg.score_threshold, cfg.model_shape,
                                                   onnxFallbackModelPath(cfg.model_file), cfg.nms_threshold);
#else
            throw std::runtime_error("The Core AI backend is unavailable and the ONNX Runtime backend was not compiled into this binary.");
#endif
        }
    }

    throw std::runtime_error(std::string(frameworkName(cfg.framework)) +
                             " backend was not compiled into this binary.");
}
