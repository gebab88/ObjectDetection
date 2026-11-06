#include "Detection_Hailo.hpp"
#include <stdexcept>
#include <iostream>
#include <cstring>
#include <filesystem>

Detection_Hailo::Detection_Hailo(float score_threshold,
                                 float nms_iou_threshold,
                                 cv::Size2f model_shape,
                                 const std::string &model_file)
    : Detection(score_threshold, model_shape, model_file),
      ort_env_(ORT_LOGGING_LEVEL_ERROR, "yolo26-postprocess")
{
    // ── VDevice ───────────────────────────────────────────────────────────────
    auto vdevice_exp = hailort::VDevice::create();
    if (!vdevice_exp) {
        throw std::runtime_error("[Hailo] VDevice::create failed: "
                                 + std::to_string(vdevice_exp.status()));
    }
    vdevice_ = vdevice_exp.release();

    // ── InferModel from HEF ───────────────────────────────────────────────────
    auto infer_model_exp = vdevice_->create_infer_model(model_file_);
    if (!infer_model_exp) {
        throw std::runtime_error("[Hailo] create_infer_model failed: "
                                 + std::to_string(infer_model_exp.status()));
    }
    auto infer_model = infer_model_exp.release();

    // ── Input: query real shape and update model_shape_ ───────────────────────
    auto input_exp = infer_model->input();
    if (!input_exp) {
        throw std::runtime_error("[Hailo] input() failed");
    }
    const auto input_shape = input_exp->shape();
    model_shape_ = cv::Size2f(static_cast<float>(input_shape.width),
                              static_cast<float>(input_shape.height));
    input_name_  = input_exp->name();
    const size_t input_frame_size = input_exp->get_frame_size();

    // ── Outputs: support both Hailo NMS HEFs and raw multi-output HEFs ────────
    output_names_ = infer_model->get_output_names();
    if (output_names_.empty()) {
        throw std::runtime_error("[Hailo] Model has no outputs");
    }

    for (const auto &name : output_names_) {
        auto output_exp = infer_model->output(name);
        if (!output_exp) {
            throw std::runtime_error("[Hailo] output('" + name + "') failed");
        }

        output_exp->set_format_type(HAILO_FORMAT_TYPE_FLOAT32);

        if (output_exp->is_nms()) {
            if (output_names_.size() != 1) {
                throw std::runtime_error("[Hailo] NMS output mixed with raw outputs is unsupported");
            }
            output_exp->set_nms_score_threshold(score_threshold_);
            output_exp->set_nms_iou_threshold(nms_iou_threshold);
            nms_shape_ = output_exp->get_nms_shape().value();
            output_name_ = name;
            nms_output_ = true;
        }

        output_buffers_[name].resize(output_exp->get_frame_size());
    }

    if (!nms_output_) {
        setup_yolo26_postprocess();
    }

    // ── Configure ─────────────────────────────────────────────────────────────
    auto configured_exp = infer_model->configure();
    if (!configured_exp) {
        throw std::runtime_error("[Hailo] configure failed: "
                                 + std::to_string(configured_exp.status()));
    }
    configured_model_ = configured_exp.release();

    // ── Allocate I/O buffers and create bindings ───────────────────────────────
    input_buffer_.resize(input_frame_size);
    std::map<std::string, hailort::MemoryView> buf_map;
    buf_map[input_name_]  = hailort::MemoryView(input_buffer_.data(),  input_frame_size);
    for (auto &[name, buffer] : output_buffers_) {
        buf_map[name] = hailort::MemoryView(buffer.data(), buffer.size());
    }

    auto bindings_exp = configured_model_.create_bindings(buf_map);
    if (!bindings_exp) {
        throw std::runtime_error("[Hailo] create_bindings failed: "
                                 + std::to_string(bindings_exp.status()));
    }
    bindings_ = bindings_exp.release();

    std::cout << "[Hailo] Model:  " << model_file_ << "\n"
              << "[Hailo] Input:  " << input_shape.height << "x"
                                    << input_shape.width  << "x"
                                    << input_shape.features << "  ("
                                    << input_frame_size << " bytes)\n";

    if (nms_output_) {
        std::cout << "[Hailo] NMS:    " << nms_shape_.number_of_classes
                  << " classes, max "
                  << nms_shape_.max_bboxes_per_class
                  << " bboxes/class\n";
    } else {
        std::cout << "[Hailo] Raw outputs:\n";
        for (const auto &name : output_names_) {
            auto output_exp = infer_model->output(name);
            const auto shape = output_exp->shape();
            std::cout << "  - " << name << ": "
                      << shape.height << "x" << shape.width << "x"
                      << shape.features << " ("
                      << output_buffers_[name].size() << " bytes)\n";
        }
    }
}

void Detection_Hailo::detect(cv::Mat &frame) {
    const int W = static_cast<int>(model_shape_.width);
    const int H = static_cast<int>(model_shape_.height);

    // Resize → BGR→RGB → NHWC uint8 in input_buffer_
    cv::Mat resized;
    cv::resize(frame, resized, {W, H});
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    std::memcpy(input_buffer_.data(), resized.data, input_buffer_.size());

    // Synchronous inference
    const hailo_status status =
        configured_model_.run(bindings_, std::chrono::milliseconds(1000));
    if (status != HAILO_SUCCESS) {
        std::cerr << "[Hailo] run() failed: " << status << "\n";
        return;
    }

    if (!nms_output_) {
        if (!raw_output_notice_printed_) {
            std::cout << "[Hailo] Raw multi-output inference is running with "
                      << "YOLO26 ONNX postprocess.\n";
            raw_output_notice_printed_ = true;
        }
        draw_detections(frame, run_yolo26_postprocess());
        return;
    }

    // ── Parse NMS output (HAILO_FORMAT_ORDER_HAILO_NMS_BY_CLASS) ─────────────
    // Per class: float32 bbox_count, then max_bboxes_per_class × hailo_bbox_float32_t
    const float x_scale = static_cast<float>(frame.cols);
    const float y_scale = static_cast<float>(frame.rows);
    constexpr size_t BBOX_FLOATS = sizeof(hailo_bbox_float32_t) / sizeof(float); // 5

    const float* ptr = reinterpret_cast<const float*>(output_buffers_[output_name_].data());

    for (uint32_t cls = 0; cls < nms_shape_.number_of_classes; cls++) {
        const int count = std::min(static_cast<int>(*ptr++),
                                   static_cast<int>(nms_shape_.max_bboxes_per_class));
        for (int d = 0; d < count; d++) {
            const auto& bbox = *reinterpret_cast<const hailo_bbox_float32_t*>(ptr);

            if (bbox.score >= score_threshold_ && cls < class_list_.size()) {
                const float x1 = bbox.x_min * x_scale;
                const float y1 = bbox.y_min * y_scale;
                const float x2 = bbox.x_max * x_scale;
                const float y2 = bbox.y_max * y_scale;

                if (x2 > x1 && y2 > y1) {
                    const cv::Rect box(int(x1), int(y1), int(x2 - x1), int(y2 - y1));
                    cv::rectangle(frame, box, cv::Scalar(0, 255, 255), 1);
                    const std::string label =
                        class_list_[cls] + "  " + std::to_string(int(100 * bbox.score)) + "%";
                    std::cout << label << '\n';
                    cv::putText(frame, label, cv::Point(box.x, box.y),
                                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
                }
            }
            ptr += BBOX_FLOATS;
        }
    }
}

void Detection_Hailo::setup_yolo26_postprocess() {
    raw_output_order_ = {
        "yolo26m/conv71",
        "yolo26m/conv74",
        "yolo26m/conv87",
        "yolo26m/conv90",
        "yolo26m/conv101",
        "yolo26m/conv104",
    };

    raw_output_info_ = {
        {"yolo26m/conv71",  {"/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv_output_0", 4, 80, 80}},
        {"yolo26m/conv74",  {"/model.23/one2one_cv3.0/one2one_cv3.0.2/Conv_output_0", 80, 80, 80}},
        {"yolo26m/conv87",  {"/model.23/one2one_cv2.1/one2one_cv2.1.2/Conv_output_0", 4, 40, 40}},
        {"yolo26m/conv90",  {"/model.23/one2one_cv3.1/one2one_cv3.1.2/Conv_output_0", 80, 40, 40}},
        {"yolo26m/conv101", {"/model.23/one2one_cv2.2/one2one_cv2.2.2/Conv_output_0", 4, 20, 20}},
        {"yolo26m/conv104", {"/model.23/one2one_cv3.2/one2one_cv3.2.2/Conv_output_0", 80, 20, 20}},
    };

    for (const auto &name : raw_output_order_) {
        if (output_buffers_.find(name) == output_buffers_.end()) {
            throw std::runtime_error("[Hailo] YOLO26m postprocess missing output: " + name);
        }
    }

    // Look next to the HEF (whose path is already resolved against the config
    // directory), so running from build/ still finds the repo-root model.
    const std::filesystem::path postprocess_path =
        std::filesystem::path(model_file_).parent_path() / "yolo26m_postprocess.onnx";
    if (!std::filesystem::exists(postprocess_path)) {
        throw std::runtime_error("[Hailo] Missing YOLO26 postprocess model: "
                                 + postprocess_path.string());
    }

    ort_session_opts_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    ort_session_opts_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    ort_session_opts_.SetIntraOpNumThreads(1);
    postprocess_session_ = std::make_unique<Ort::Session>(
        ort_env_, postprocess_path.string().c_str(), ort_session_opts_);

    const size_t output_count = postprocess_session_->GetOutputCount();
    postprocess_output_names_.clear();
    postprocess_output_names_.reserve(output_count);
    for (size_t i = 0; i < output_count; ++i) {
        postprocess_output_names_.emplace_back(
            postprocess_session_->GetOutputNameAllocated(i, ort_allocator_).get());
    }

    std::cout << "[Hailo] YOLO26 postprocess: " << postprocess_path.string() << "\n";
}

std::vector<Detection_Hailo::DetectionResult> Detection_Hailo::run_yolo26_postprocess() {
    if (!postprocess_session_) {
        return {};
    }

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<std::vector<float>> chw_buffers;
    std::vector<std::array<int64_t, 4>> shapes;
    std::vector<Ort::Value> input_tensors;
    std::vector<const char*> input_names;

    chw_buffers.reserve(raw_output_order_.size());
    shapes.reserve(raw_output_order_.size());
    input_tensors.reserve(raw_output_order_.size());
    input_names.reserve(raw_output_order_.size());

    for (const auto &hef_name : raw_output_order_) {
        const auto &info = raw_output_info_.at(hef_name);
        const auto &bytes = output_buffers_.at(hef_name);
        const auto *nhwc = reinterpret_cast<const float*>(bytes.data());
        const size_t expected_floats = static_cast<size_t>(info.height) * info.width * info.channels;
        if (bytes.size() != expected_floats * sizeof(float)) {
            throw std::runtime_error("[Hailo] Unexpected output size for " + hef_name);
        }

        auto &chw = chw_buffers.emplace_back(expected_floats);
        for (int h = 0; h < info.height; ++h) {
            for (int w = 0; w < info.width; ++w) {
                for (int c = 0; c < info.channels; ++c) {
                    const size_t nhwc_idx = (static_cast<size_t>(h) * info.width + w) * info.channels + c;
                    const size_t chw_idx = (static_cast<size_t>(c) * info.height + h) * info.width + w;
                    chw[chw_idx] = nhwc[nhwc_idx];
                }
            }
        }

        shapes.push_back({1, info.channels, info.height, info.width});
        input_names.push_back(info.onnx_name.c_str());
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info, chw.data(), chw.size(), shapes.back().data(), shapes.back().size()));
    }

    std::vector<const char*> output_names;
    output_names.reserve(postprocess_output_names_.size());
    for (const auto &name : postprocess_output_names_) {
        output_names.push_back(name.c_str());
    }

    auto outputs = postprocess_session_->Run(
        Ort::RunOptions{nullptr},
        input_names.data(), input_tensors.data(), input_tensors.size(),
        output_names.data(), output_names.size());

    if (outputs.empty()) {
        return {};
    }

    auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    if (shape.size() != 3 || shape[2] < 6) {
        throw std::runtime_error("[Hailo] Unexpected YOLO26 postprocess output shape");
    }

    const int64_t num_dets = shape[1];
    const int64_t fields = shape[2];
    const float *data = outputs[0].GetTensorData<float>();
    const float x_scale = static_cast<float>(model_shape_.width) / 640.0f;
    const float y_scale = static_cast<float>(model_shape_.height) / 640.0f;

    std::vector<DetectionResult> detections;
    detections.reserve(static_cast<size_t>(num_dets));
    for (int64_t i = 0; i < num_dets; ++i) {
        const float *det = data + i * fields;
        const float score = det[4];
        const int class_id = static_cast<int>(det[5]);
        if (score < score_threshold_ || class_id < 0 || class_id >= static_cast<int>(class_list_.size())) {
            continue;
        }

        const float x1 = std::max(0.0f, det[0] * x_scale);
        const float y1 = std::max(0.0f, det[1] * y_scale);
        const float x2 = std::min(static_cast<float>(model_shape_.width), det[2] * x_scale);
        const float y2 = std::min(static_cast<float>(model_shape_.height), det[3] * y_scale);
        if (x2 <= x1 || y2 <= y1) {
            continue;
        }

        detections.push_back({
            cv::Rect(static_cast<int>(x1), static_cast<int>(y1),
                     static_cast<int>(x2 - x1), static_cast<int>(y2 - y1)),
            class_id,
            score
        });
    }

    return detections;
}

void Detection_Hailo::draw_detections(cv::Mat &frame, const std::vector<DetectionResult> &detections) {
    const float x_scale = static_cast<float>(frame.cols) / model_shape_.width;
    const float y_scale = static_cast<float>(frame.rows) / model_shape_.height;

    for (const auto &det : detections) {
        cv::Rect box(
            static_cast<int>(det.box.x * x_scale),
            static_cast<int>(det.box.y * y_scale),
            static_cast<int>(det.box.width * x_scale),
            static_cast<int>(det.box.height * y_scale));
        box &= cv::Rect(0, 0, frame.cols, frame.rows);
        if (box.empty()) {
            continue;
        }

        cv::rectangle(frame, box, cv::Scalar(0, 255, 255), 2);
        const std::string label = class_list_[det.class_id] + "  "
            + std::to_string(static_cast<int>(100 * det.score)) + "%";
        std::cout << label << '\n';
        cv::putText(frame, label, cv::Point(box.x, std::max(0, box.y - 4)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
    }
}
