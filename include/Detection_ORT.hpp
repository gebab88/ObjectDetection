#ifndef OBJECTDETECTION_DETECTIONHANDLER_ORT_H
#define OBJECTDETECTION_DETECTIONHANDLER_ORT_H

#include <onnxruntime_cxx_api.h>
#include <coreml_provider_factory.h>
#include <cpu_provider_factory.h>

#include "Detection.hpp"


class Detection_ORT : public Detection {
    public:
    Detection_ORT(  const float score_threshold,
                    const Size2f model_shape,
                    const string &model_file);
    virtual ~Detection_ORT() = default;

    void detect(cv::Mat &frame) override;

    private:
    Ort::Env                          env_;
    Ort::SessionOptions               session_opts_;
    Ort::AllocatorWithDefaultOptions  allocator_;
    Ort::Session                      session_;
    std::string                       input_name_, output_name_ ;
    size_t                            plane_;
};

#endif //OBJECTDETECTION_DETECTIONHANDLER_ORT_H