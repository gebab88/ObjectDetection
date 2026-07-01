#ifndef OBJECTDETECTION_DETECTIONHANDLER_ORT_H
#define OBJECTDETECTION_DETECTIONHANDLER_ORT_H

#include <onnxruntime_cxx_api.h>
#include <thread>

#ifdef __APPLE__
    #include <coreml_provider_factory.h>
    #include <cpu_provider_factory.h>
#elif __linux__
    // Linux-specific includes (none required yet)
#endif

#include "Detection.hpp"


class Detection_ORT : public Detection {
    public:
    Detection_ORT(  const float score_threshold,
                    const cv::Size2f model_shape,
                    const std::string &model_file,
                    const float nms_threshold);
    virtual ~Detection_ORT() = default;

    void detect(cv::Mat &frame) override;

    private:
    static Ort::SessionOptions make_session_opts(const std::string& model_file);
    
    Ort::Env                          env_;
    Ort::SessionOptions               session_opts_;
    Ort::AllocatorWithDefaultOptions  allocator_;
    Ort::Session                      session_;
    std::string                       input_name_;
    std::vector<std::string>          output_names_;
    std::vector<const char*>          output_name_ptrs_;
    float                             nms_threshold_;
    Ort::MemoryInfo                   memory_info_;
};

#endif //OBJECTDETECTION_DETECTIONHANDLER_ORT_H
