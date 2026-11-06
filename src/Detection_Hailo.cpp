#include "Detection_Hailo.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdexcept>

namespace {

constexpr auto kTimeout = std::chrono::milliseconds(1000);

// Frames kept in flight, so the chip always has work while the CPU reads,
// decodes and writes frames. Measured on a Pi 5 with Hailo-10H: 4 reaches the
// chip's own throughput (yolov8m 77 of 78 FPS, yolo26m 59 of 59); 3 loses a
// few FPS on yolov8m, 2 about 30 %. More only add latency.
constexpr std::size_t kMaxFramesInFlight = 4;

std::string hailoError(const std::string &what, hailo_status status) {
    return "[Hailo] " + what + " failed: " + hailo_get_status_message(status)
         + " (" + std::to_string(status) + ")";
}

}  // namespace

Detection_Hailo::Detection_Hailo(float score_threshold,
                                 float nms_iou_threshold,
                                 cv::Size2f model_shape,
                                 const std::string &model_file)
    : Detection(score_threshold, model_shape, model_file)
{
    // ── VDevice ───────────────────────────────────────────────────────────────
    auto vdevice_exp = hailort::VDevice::create();
    if (!vdevice_exp) {
        throw std::runtime_error(hailoError("VDevice::create", vdevice_exp.status()));
    }
    vdevice_ = vdevice_exp.release();

    // ── InferModel from HEF ───────────────────────────────────────────────────
    auto infer_model_exp = vdevice_->create_infer_model(model_file_);
    if (!infer_model_exp) {
        throw std::runtime_error(hailoError("create_infer_model", infer_model_exp.status()));
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
    if (input_shape.features != 3
        || input_frame_size != static_cast<size_t>(input_shape.width) * input_shape.height * 3) {
        throw std::runtime_error("[Hailo] Expected an NHWC UINT8 RGB input");
    }

    // ── Outputs: Hailo NMS HEFs or raw YOLO26 HEFs ────────────────────────────
    const auto output_names = infer_model->get_output_names();
    if (output_names.empty()) {
        throw std::runtime_error("[Hailo] Model has no outputs");
    }
    if (output_names.size() == 1 && infer_model->output(output_names[0])->is_nms()) {
        auto output = infer_model->output(output_names[0]);
        output->set_format_type(HAILO_FORMAT_TYPE_FLOAT32);
        output->set_nms_score_threshold(score_threshold_);
        output->set_nms_iou_threshold(nms_iou_threshold);
        nms_shape_ = output->get_nms_shape().value();
        output_name_ = output_names[0];
        nms_output_ = true;
    } else {
        setup_raw_outputs(*infer_model);
    }

    // ── Configure ─────────────────────────────────────────────────────────────
    auto configured_exp = infer_model->configure();
    if (!configured_exp) {
        throw std::runtime_error(hailoError("configure", configured_exp.status()));
    }
    configured_model_ = configured_exp.release();

    // ── One set of buffers and bindings per frame in flight ───────────────────
    auto queue_size = configured_model_.get_async_queue_size();
    const std::size_t slot_count = std::clamp<std::size_t>(
        queue_size ? queue_size.value() : 1, 1, kMaxFramesInFlight);
    for (std::size_t i = 0; i < slot_count; ++i) {
        auto slot = std::make_unique<Slot>();
        slot->input.resize(input_frame_size);
        std::map<std::string, hailort::MemoryView> buf_map;
        buf_map[input_name_] = hailort::MemoryView(slot->input.data(), slot->input.size());
        for (const auto &name : output_names) {
            auto &buffer = slot->outputs[name];
            buffer.resize(infer_model->output(name)->get_frame_size());
            buf_map[name] = hailort::MemoryView(buffer.data(), buffer.size());
        }
        auto bindings_exp = configured_model_.create_bindings(buf_map);
        if (!bindings_exp) {
            throw std::runtime_error(hailoError("create_bindings", bindings_exp.status()));
        }
        slot->bindings = bindings_exp.release();
        slots_.push_back(std::move(slot));
    }

    std::cout << "[Hailo] Model:  " << model_file_ << "\n"
              << "[Hailo] Input:  " << input_shape.height << "x"
                                    << input_shape.width  << "x"
                                    << input_shape.features << "  ("
                                    << input_frame_size << " bytes)\n"
              << "[Hailo] Frames in flight: " << slots_.size() << "\n";

    if (nms_output_) {
        std::cout << "[Hailo] NMS:    " << nms_shape_.number_of_classes
                  << " classes, max "
                  << nms_shape_.max_bboxes_per_class
                  << " bboxes/class\n";
    } else {
        std::cout << "[Hailo] Raw YOLO26 head, decoded on the CPU: "
                  << raw_num_classes_ << " classes, strides";
        for (const auto &level : raw_levels_) {
            std::cout << " " << level.stride;
        }
        std::cout << "\n";
    }
}

Detection_Hailo::~Detection_Hailo() {
    // Let jobs still running finish before their buffers are freed.
    for (auto &slot : slots_) {
        if (slot->busy) {
            slot->job.wait(kTimeout);
        }
    }
}

// Pairs the raw outputs per head scale: a 4-channel box tensor and a class
// tensor with the same grid. The grid size gives the stride.
void Detection_Hailo::setup_raw_outputs(hailort::InferModel &infer_model) {
    struct Tensor {
        std::string name;
        yolo26::Quant quant;
    };
    std::map<std::pair<int, int>, Tensor> boxes, scores;  // keyed by (height, width)

    for (const auto &name : infer_model.get_output_names()) {
        auto output = infer_model.output(name);
        if (!output) {
            throw std::runtime_error("[Hailo] output('" + name + "') failed");
        }
        if (output->is_nms()) {
            throw std::runtime_error("[Hailo] NMS output mixed with raw outputs is unsupported");
        }
        // Keep the chip's quantized values: the decoder compares them against
        // the score threshold directly and dequantizes only what passes.
        output->set_format_type(HAILO_FORMAT_TYPE_UINT16);
        const auto quant_infos = output->get_quant_infos();
        if (quant_infos.size() != 1 || quant_infos[0].qp_scale <= 0.0f) {
            throw std::runtime_error("[Hailo] Unsupported quantization of output " + name);
        }
        const auto shape = output->shape();
        const yolo26::Quant quant{quant_infos[0].qp_zp, quant_infos[0].qp_scale};
        const auto grid = std::make_pair(static_cast<int>(shape.height), static_cast<int>(shape.width));

        if (shape.features == 4) {
            boxes[grid] = {name, quant};
        } else if (raw_num_classes_ == 0 || static_cast<int>(shape.features) == raw_num_classes_) {
            raw_num_classes_ = static_cast<int>(shape.features);
            scores[grid] = {name, quant};
        } else {
            throw std::runtime_error("[Hailo] Output " + name + " has an unexpected channel count");
        }
    }

    if (boxes.empty() || boxes.size() != scores.size()) {
        throw std::runtime_error("[Hailo] Raw HEF outputs do not look like a YOLO26 detection head "
                                 "(expected one 4-channel box and one class output per scale)");
    }
    for (const auto &[grid, box] : boxes) {
        const auto score = scores.find(grid);
        if (score == scores.end()) {
            throw std::runtime_error("[Hailo] No class output matches box output " + box.name);
        }
        RawLevel level;
        level.boxes_output = box.name;
        level.scores_output = score->second.name;
        level.box_quant = box.quant;
        level.score_quant = score->second.quant;
        level.height = grid.first;
        level.width = grid.second;
        level.stride = model_shape_.width / static_cast<float>(grid.second);
        raw_levels_.push_back(level);
    }
    // Finest grid first, like the head's own output order.
    std::sort(raw_levels_.begin(), raw_levels_.end(),
              [](const RawLevel &a, const RawLevel &b) { return a.stride < b.stride; });
}

void Detection_Hailo::detect(cv::Mat &frame) {
    detect_async(frame).get();
}

std::future<void> Detection_Hailo::detect_async(cv::Mat frame) {
    Slot &slot = *slots_[next_slot_];
    if (slot.busy) {
        throw std::logic_error("[Hailo] more than max_frames_in_flight() frames in flight");
    }
    next_slot_ = (next_slot_ + 1) % slots_.size();

    // Resize → BGR→RGB, written straight into the slot's NHWC uint8 input.
    const int W = static_cast<int>(model_shape_.width);
    const int H = static_cast<int>(model_shape_.height);
    cv::Mat resized;
    cv::resize(frame, resized, {W, H});
    cv::Mat input(H, W, CV_8UC3, slot.input.data());
    cv::cvtColor(resized, input, cv::COLOR_BGR2RGB);

    hailo_status status = configured_model_.wait_for_async_ready(kTimeout);
    if (status != HAILO_SUCCESS) {
        throw std::runtime_error(hailoError("wait_for_async_ready", status));
    }
    slot.status = HAILO_SUCCESS;
    auto job_exp = configured_model_.run_async(
        slot.bindings,
        [&slot](const hailort::AsyncInferCompletionInfo &info) { slot.status = info.status; });
    if (!job_exp) {
        throw std::runtime_error(hailoError("run_async", job_exp.status()));
    }
    slot.job = job_exp.release();
    slot.busy = true;

    // Decoding and drawing run when the caller collects the frame.
    return std::async(std::launch::deferred, [this, &slot, frame]() mutable {
        finish(slot, frame);
    });
}

void Detection_Hailo::finish(Slot &slot, cv::Mat &frame) {
    struct Release {
        Slot &slot;
        ~Release() { slot.busy = false; }
    } release{slot};

    hailo_status status = slot.job.wait(kTimeout);
    if (status == HAILO_SUCCESS) {
        status = slot.status;
    }
    if (status != HAILO_SUCCESS) {
        throw std::runtime_error(hailoError("inference", status));
    }

    if (nms_output_) {
        draw_nms_output(slot, frame);
    } else {
        draw_detections(frame, decode_raw_output(slot));
    }
}

void Detection_Hailo::draw_nms_output(Slot &slot, cv::Mat &frame) {
    // ── Parse NMS output (HAILO_FORMAT_ORDER_HAILO_NMS_BY_CLASS) ─────────────
    // Per class: float32 bbox_count, then max_bboxes_per_class × hailo_bbox_float32_t
    const float x_scale = static_cast<float>(frame.cols);
    const float y_scale = static_cast<float>(frame.rows);
    constexpr size_t BBOX_FLOATS = sizeof(hailo_bbox_float32_t) / sizeof(float); // 5

    const float* ptr = reinterpret_cast<const float*>(slot.outputs.at(output_name_).data());

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

std::vector<Detection_Hailo::DetectionResult> Detection_Hailo::decode_raw_output(Slot &slot) {
    std::vector<yolo26::HeadLevel> levels;
    levels.reserve(raw_levels_.size());
    for (const auto &level : raw_levels_) {
        levels.push_back({
            reinterpret_cast<const uint16_t*>(slot.outputs.at(level.boxes_output).data()),
            level.box_quant,
            reinterpret_cast<const uint16_t*>(slot.outputs.at(level.scores_output).data()),
            level.score_quant,
            level.height, level.width, level.stride});
    }

    const float max_x = model_shape_.width;
    const float max_y = model_shape_.height;
    std::vector<DetectionResult> detections;
    for (const auto &det : yolo26::decode(levels, raw_num_classes_, score_threshold_)) {
        if (det.class_id >= static_cast<int>(class_list_.size())) {
            continue;
        }
        const float x1 = std::max(0.0f, det.x1);
        const float y1 = std::max(0.0f, det.y1);
        const float x2 = std::min(max_x, det.x2);
        const float y2 = std::min(max_y, det.y2);
        if (x2 <= x1 || y2 <= y1) {
            continue;
        }
        detections.push_back({
            cv::Rect(static_cast<int>(x1), static_cast<int>(y1),
                     static_cast<int>(x2 - x1), static_cast<int>(y2 - y1)),
            det.class_id,
            det.score
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
