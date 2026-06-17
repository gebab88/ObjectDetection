#include <gtest/gtest.h>

#include "Framework.hpp"
#include "YoloPostprocess.hpp"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>

// Tests for detectModelFormat helper

namespace {
// These decode tests assert on the labels printed to stdout, so opt into the
// detection logging that is disabled by default on the real-time path.
const bool kDetectionLoggingEnabled = [] {
    yolo_postprocess::logDetectionsToStdout() = true;
    return true;
}();

int changedPixelCount(const cv::Mat& before, const cv::Mat& after) {
    cv::Mat diff;
    cv::absdiff(before, after, diff);

    cv::Mat gray;
    cv::cvtColor(diff, gray, cv::COLOR_BGR2GRAY);

    cv::Mat mask;
    cv::threshold(gray, mask, 0, 255, cv::THRESH_BINARY);
    return cv::countNonZero(mask);
}

void expectYellowPixelNear(const cv::Mat& frame, int x, int y) {
    bool found = false;
    for (int yy = std::max(0, y - 1); yy <= std::min(frame.rows - 1, y + 1); ++yy) {
        for (int xx = std::max(0, x - 1); xx <= std::min(frame.cols - 1, x + 1); ++xx) {
            const cv::Vec3b pixel = frame.at<cv::Vec3b>(yy, xx);
            if (pixel[0] == 0 && pixel[1] == 255 && pixel[2] == 255) {
                found = true;
            }
        }
    }
    EXPECT_TRUE(found) << "Expected yellow bounding-box pixel near (" << x << ", " << y << ")";
}

size_t countOccurrences(const std::string& text, const std::string& needle) {
    size_t count = 0;
    size_t pos = 0;
    while ((pos = text.find(needle, pos)) != std::string::npos) {
        ++count;
        pos += needle.size();
    }
    return count;
}
}

TEST(DetectModelFormat, ReturnsYOLO12ForYolo12m) {
    EXPECT_EQ(detectModelFormat("/path/to/yolo12m.onnx"), MODEL_FORMAT::YOLO12);
}

TEST(DetectModelFormat, ReturnsYOLO12ForYolo12x) {
    EXPECT_EQ(detectModelFormat("yolo12x.onnx"), MODEL_FORMAT::YOLO12);
}

TEST(DetectModelFormat, ReturnsYOLO26ForYolo26m) {
    EXPECT_EQ(detectModelFormat("models/yolo26m.onnx"), MODEL_FORMAT::YOLO26);
}

TEST(DetectModelFormat, ReturnsYOLO26ForYolo26x) {
    EXPECT_EQ(detectModelFormat("yolo26x.onnx"), MODEL_FORMAT::YOLO26);
}

TEST(DetectModelFormat, ReturnsYOLOEForYoloE11Seg) {
    EXPECT_EQ(detectModelFormat("models/yoloe-11l-seg.onnx"), MODEL_FORMAT::YOLOE);
}

TEST(DetectModelFormat, ReturnsYOLOEForYoloE26Seg) {
    EXPECT_EQ(detectModelFormat("yoloe-26l-seg.onnx"), MODEL_FORMAT::YOLOE);
}

TEST(DetectModelFormat, ReturnsYOLOEForYoloE26PromptFree) {
    EXPECT_EQ(detectModelFormat("yoloe-26m-seg-pf.onnx"), MODEL_FORMAT::YOLOE);
}

TEST(DetectModelFormat, ReturnsUnknownForOtherFiles) {
    EXPECT_EQ(detectModelFormat("some_model.onnx"), MODEL_FORMAT::Unknown);
}

TEST(YoloPostprocess, DecodeRawDrawsBoundingBoxAndLabel) {
    constexpr int fields = 7; // x, y, w, h, 3 class scores
    constexpr int detections = 8;
    std::vector<float> output(fields * detections, 0.01f);

    auto set_value = [&](int field, int det, float value) {
        output[field * detections + det] = value;
    };

    set_value(0, 3, 50.0f);
    set_value(1, 3, 50.0f);
    set_value(2, 3, 20.0f);
    set_value(3, 3, 20.0f);
    set_value(4, 3, 0.05f);
    set_value(5, 3, 0.92f);
    set_value(6, 3, 0.10f);

    cv::Mat frame(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
    const cv::Mat before = frame.clone();
    const std::vector<std::string> classes = {"person", "helmet", "forklift"};

    testing::internal::CaptureStdout();
    const bool decoded = yolo_postprocess::decodeRaw(output.data(), fields, detections, frame,
                                                     cv::Size2f(100, 100), classes,
                                                     0.5f, 0.5f);
    const std::string stdout_text = testing::internal::GetCapturedStdout();

    EXPECT_TRUE(decoded);
    EXPECT_GT(changedPixelCount(before, frame), 0);
    expectYellowPixelNear(frame, 40, 40);
    EXPECT_NE(stdout_text.find("helmet"), std::string::npos);
}

TEST(YoloPostprocess, DecodeEndToEndDrawsBoundingBoxAndLabel) {
    constexpr int detections = 2;
    constexpr int fields = 6; // x1, y1, x2, y2, score, class id
    std::vector<float> output(detections * fields, 0.0f);

    output[0 * fields + 0] = 20.0f;
    output[0 * fields + 1] = 20.0f;
    output[0 * fields + 2] = 70.0f;
    output[0 * fields + 3] = 80.0f;
    output[0 * fields + 4] = 0.95f;
    output[0 * fields + 5] = 1.0f;

    cv::Mat frame(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
    const cv::Mat before = frame.clone();
    const std::vector<std::string> classes = {"person", "truck"};

    testing::internal::CaptureStdout();
    const bool decoded = yolo_postprocess::decodeEndToEnd(output.data(), detections, fields, frame,
                                                          cv::Size2f(100, 100), classes, 0.5f, 0.5f);
    const std::string stdout_text = testing::internal::GetCapturedStdout();

    EXPECT_TRUE(decoded);
    EXPECT_GT(changedPixelCount(before, frame), 0);
    expectYellowPixelNear(frame, 20, 20);
    EXPECT_NE(stdout_text.find("truck"), std::string::npos);
}

TEST(YoloPostprocess, DecodeEndToEndIgnoresSegmentationFields) {
    constexpr int detections = 300;
    constexpr int fields = 38; // x1, y1, x2, y2, score, class id, mask coefficients
    std::vector<float> output(detections * fields, 0.0f);

    constexpr int det = 17;
    output[det * fields + 0] = 20.0f;
    output[det * fields + 1] = 20.0f;
    output[det * fields + 2] = 70.0f;
    output[det * fields + 3] = 80.0f;
    output[det * fields + 4] = 0.95f;
    output[det * fields + 5] = 1.0f;
    output[det * fields + 6] = 0.25f;

    cv::Mat frame(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
    const cv::Mat before = frame.clone();
    const std::vector<std::string> classes = {"person", "truck"};

    testing::internal::CaptureStdout();
    const bool decoded = yolo_postprocess::decodeEndToEnd(output.data(), detections, fields, frame,
                                                          cv::Size2f(100, 100), classes, 0.5f, 0.5f);
    const std::string stdout_text = testing::internal::GetCapturedStdout();

    EXPECT_TRUE(decoded);
    EXPECT_GT(changedPixelCount(before, frame), 0);
    expectYellowPixelNear(frame, 20, 20);
    EXPECT_NE(stdout_text.find("truck"), std::string::npos);
}

TEST(YoloPostprocess, DecodeEndToEndAppliesNmsToOverlappingBoxes) {
    constexpr int detections = 2;
    constexpr int fields = 6; // x1, y1, x2, y2, score, class id
    std::vector<float> output(detections * fields, 0.0f);

    output[0 * fields + 0] = 20.0f;
    output[0 * fields + 1] = 20.0f;
    output[0 * fields + 2] = 70.0f;
    output[0 * fields + 3] = 80.0f;
    output[0 * fields + 4] = 0.95f;
    output[0 * fields + 5] = 1.0f;

    output[1 * fields + 0] = 21.0f;
    output[1 * fields + 1] = 21.0f;
    output[1 * fields + 2] = 71.0f;
    output[1 * fields + 3] = 81.0f;
    output[1 * fields + 4] = 0.90f;
    output[1 * fields + 5] = 1.0f;

    cv::Mat frame(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
    const std::vector<std::string> classes = {"person", "truck"};

    testing::internal::CaptureStdout();
    const bool decoded = yolo_postprocess::decodeEndToEnd(output.data(), detections, fields, frame,
                                                          cv::Size2f(100, 100), classes, 0.5f, 0.5f);
    const std::string stdout_text = testing::internal::GetCapturedStdout();

    EXPECT_TRUE(decoded);
    EXPECT_EQ(countOccurrences(stdout_text, "truck"), 1u);
}

TEST(YoloPostprocess, DecodeRawRejectsShapeThatDoesNotFitClassList) {
    constexpr int fields = 5;
    constexpr int detections = 8;
    std::vector<float> output(fields * detections, 0.9f);

    cv::Mat frame(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
    const cv::Mat before = frame.clone();
    const std::vector<std::string> classes = {"person", "helmet", "forklift"};

    const bool decoded = yolo_postprocess::decodeRaw(output.data(), fields, detections, frame,
                                                     cv::Size2f(100, 100), classes,
                                                     0.5f, 0.5f);

    EXPECT_FALSE(decoded);
    EXPECT_EQ(changedPixelCount(before, frame), 0);
}
