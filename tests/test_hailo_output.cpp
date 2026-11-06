#include <gtest/gtest.h>

#include <vector>

#include <opencv2/core.hpp>

#include "HailoOutput.hpp"
#include "YoloPostprocess.hpp"

namespace {

// Builds a HAILO_FORMAT_ORDER_HAILO_NMS_BY_CLASS float32 buffer: per class the
// box count, then that many {y_min, x_min, y_max, x_max, score} (normalized).
struct NmsBuffer {
    struct Box {
        float y_min, x_min, y_max, x_max, score;
    };

    std::vector<float> data;

    void addClass(const std::vector<Box>& boxes) {
        data.push_back(static_cast<float>(boxes.size()));
        for (const Box& b : boxes) {
            data.insert(data.end(), {b.y_min, b.x_min, b.y_max, b.x_max, b.score});
        }
    }
};

const std::vector<std::string> kClasses = {"person", "bicycle", "car"};

}  // namespace

TEST(HailoOutput, ParsesNmsBoxesOfEveryClassIntoFramePixels) {
    NmsBuffer nms;
    nms.addClass({{0.1f, 0.2f, 0.5f, 0.6f, 0.9f}});
    nms.addClass({});
    nms.addClass({{0.0f, 0.0f, 0.25f, 0.5f, 0.7f}, {0.5f, 0.5f, 1.0f, 1.0f, 0.6f}});

    const auto dets = hailo_output::parseNmsByClass(nms.data.data(), nms.data.size(), 3, 10,
                                                    0.5f, cv::Size(200, 100));

    ASSERT_EQ(dets.size(), 3u);
    EXPECT_EQ(dets[0].class_id, 0);
    EXPECT_EQ(dets[0].box, cv::Rect(cv::Point(40, 10), cv::Point(120, 50)));
    EXPECT_FLOAT_EQ(dets[0].score, 0.9f);
    EXPECT_EQ(dets[1].class_id, 2);
    EXPECT_EQ(dets[1].box, cv::Rect(cv::Point(0, 0), cv::Point(100, 25)));
    EXPECT_EQ(dets[2].class_id, 2);
    EXPECT_EQ(dets[2].box, cv::Rect(cv::Point(100, 50), cv::Point(200, 100)));
}

TEST(HailoOutput, SkipsNmsBoxesBelowThresholdOrOutsideTheFrame) {
    NmsBuffer nms;
    nms.addClass({{0.1f, 0.1f, 0.5f, 0.5f, 0.3f},     // score too low
                  {1.2f, 1.2f, 1.5f, 1.5f, 0.9f},     // entirely outside
                  {-0.1f, -0.1f, 0.5f, 0.5f, 0.8f}}); // clipped to the frame

    const auto dets = hailo_output::parseNmsByClass(nms.data.data(), nms.data.size(), 1, 10,
                                                    0.5f, cv::Size(100, 100));

    ASSERT_EQ(dets.size(), 1u);
    EXPECT_EQ(dets[0].box, cv::Rect(0, 0, 50, 50));
}

TEST(HailoOutput, NeverReadsPastTheNmsBuffer) {
    NmsBuffer nms;
    nms.data = {50.0f, 0.1f, 0.1f, 0.5f, 0.5f, 0.9f};  // claims 50 boxes, holds 1

    const auto dets = hailo_output::parseNmsByClass(nms.data.data(), nms.data.size(), 2, 100,
                                                    0.5f, cv::Size(100, 100));

    EXPECT_EQ(dets.size(), 1u);
}

TEST(HailoOutput, CapsNmsBoxCountAtMaxBoxesPerClass) {
    NmsBuffer nms;
    nms.addClass({{0.1f, 0.1f, 0.2f, 0.2f, 0.9f}, {0.3f, 0.3f, 0.4f, 0.4f, 0.9f}});

    const auto dets = hailo_output::parseNmsByClass(nms.data.data(), nms.data.size(), 1, 1,
                                                    0.5f, cv::Size(100, 100));

    EXPECT_EQ(dets.size(), 1u);
}

TEST(HailoOutput, ScalesYolo26ResultsFromModelInputToFrame) {
    std::vector<yolo26::Result> results(2);
    results[0] = {64.0f, 32.0f, 320.0f, 640.0f, 1, 0.8f};
    results[1] = {-20.0f, 600.0f, 100.0f, 700.0f, 0, 0.6f};  // partly outside

    const auto dets = hailo_output::fromYolo26(results, cv::Size2f(640, 640), cv::Size(1080, 1080));

    ASSERT_EQ(dets.size(), 2u);
    EXPECT_EQ(dets[0].box, cv::Rect(cv::Point(108, 54), cv::Point(540, 1080)));
    EXPECT_EQ(dets[0].class_id, 1);
    EXPECT_FLOAT_EQ(dets[0].score, 0.8f);
    EXPECT_EQ(dets[1].box, cv::Rect(cv::Point(0, 1013), cv::Point(169, 1080)));
}

TEST(HailoOutput, DrawsLikeTheOtherBackendsBestScoreFirst) {
    const std::vector<hailo_output::Detection> dets = {
        {cv::Rect(10, 40, 60, 50), 0, 0.6f},
        {cv::Rect(20, 45, 60, 50), 2, 0.9f},
    };
    cv::Mat drawn(120, 160, CV_8UC3, cv::Scalar::all(0));
    hailo_output::drawDetections(drawn, dets, kClasses);

    // Reference: the shared routine, best score first so its label gets the
    // preferred place, as in the ONNX Runtime / OpenVINO / OpenCV backends.
    cv::Mat expected(120, 160, CV_8UC3, cv::Scalar::all(0));
    std::vector<cv::Rect> occupied;
    yolo_postprocess::drawDetection(expected, dets[1].box, 2, 0.9f, kClasses, &occupied);
    yolo_postprocess::drawDetection(expected, dets[0].box, 0, 0.6f, kClasses, &occupied);

    EXPECT_EQ(cv::norm(drawn, expected, cv::NORM_INF), 0.0);
}
