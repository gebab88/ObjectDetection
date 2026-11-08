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
#include "FrameSlots.hpp"


class Detection_ORT : public Detection {
    public:
    Detection_ORT(  const float score_threshold,
                    const cv::Size2f model_shape,
                    const std::string &model_file,
                    const float nms_threshold);
    ~Detection_ORT() override = default;

    void detect(cv::Mat &frame) override;

    // With CoreML two frames are in flight, each on its own worker thread
    // (Session::Run is thread-safe), so the next inference starts while the
    // previous frame is decoded and drawn. CPU-only sessions already use all
    // cores per frame and stay synchronous.
    std::size_t max_frames_in_flight() const override { return slots_.size(); }
    std::future<void> detect_async(cv::Mat frame) override;

    private:
    struct Scratch {
        cv::Mat blob;
    };

    static Ort::SessionOptions make_session_opts(const Ort::Env& env,
                                                 const std::string& model_file,
                                                 cv::Size2f model_shape,
                                                 bool& coreml_enabled);
    void run(cv::Mat &blob, cv::Mat &frame);

    Ort::Env                          env_;
    bool                              coreml_enabled_ = false;  // set by make_session_opts
    Ort::SessionOptions               session_opts_;
    Ort::AllocatorWithDefaultOptions  allocator_;
    Ort::Session                      session_;
    std::string                       input_name_;
    std::vector<std::string>          output_names_;
    std::vector<const char*>          output_name_ptrs_;
    float                             nms_threshold_;
    Ort::MemoryInfo                   memory_info_;
    FrameSlots<Scratch>               slots_;  // last member: destroyed (and waited for) first
};

#endif //OBJECTDETECTION_DETECTIONHANDLER_ORT_H
