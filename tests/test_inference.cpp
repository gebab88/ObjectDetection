#include <gtest/gtest.h>

#include "Detection_ORT.hpp"

#include <algorithm>
#include <filesystem>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#ifndef INFERENCE_TEST_MODEL_FILE
#define INFERENCE_TEST_MODEL_FILE ""
#endif

#ifndef INFERENCE_TEST_CLASSES_FILE
#define INFERENCE_TEST_CLASSES_FILE ""
#endif

#ifndef INFERENCE_TEST_VIDEO_FILE
#define INFERENCE_TEST_VIDEO_FILE ""
#endif

namespace {
cv::Mat cropSquare(const cv::Mat& frame) {
    const int min_side = std::min(frame.cols, frame.rows);
    const int x = std::max(0, (frame.cols - min_side) / 2);
    const int y = std::max(0, (frame.rows - min_side) / 2);
    return frame(cv::Rect(x, y, min_side, min_side)).clone();
}

int changedPixelCount(const cv::Mat& before, const cv::Mat& after) {
    cv::Mat diff;
    cv::absdiff(before, after, diff);

    cv::Mat gray;
    cv::cvtColor(diff, gray, cv::COLOR_BGR2GRAY);

    cv::Mat mask;
    cv::threshold(gray, mask, 0, 255, cv::THRESH_BINARY);
    return cv::countNonZero(mask);
}
}

TEST(RealOnnxInference, DrawsBoundingBoxOnVideoFrame) {
    const std::string model_file = INFERENCE_TEST_MODEL_FILE;
    const std::string classes_file = INFERENCE_TEST_CLASSES_FILE;
    const std::string video_file = INFERENCE_TEST_VIDEO_FILE;

    if (!std::filesystem::exists(model_file)) {
        GTEST_SKIP() << "Inference test model not found: " << model_file;
    }
    if (!std::filesystem::exists(classes_file)) {
        GTEST_SKIP() << "Inference test classes file not found: " << classes_file;
    }
    if (!std::filesystem::exists(video_file)) {
        GTEST_SKIP() << "Inference test video not found: " << video_file;
    }

    Detection_ORT detector(0.25f, cv::Size2f(640, 640), model_file, 0.5f);
    detector.load_class_list(classes_file);

    cv::VideoCapture capture(video_file);
    ASSERT_TRUE(capture.isOpened()) << "Could not open test video: " << video_file;

    constexpr int max_frames = 80;
    constexpr int min_changed_pixels = 100;
    int best_changed_pixels = 0;

    for (int frame_index = 0; frame_index < max_frames; ++frame_index) {
        cv::Mat frame;
        if (!capture.read(frame)) {
            break;
        }

        cv::Mat cropped = cropSquare(frame);
        const cv::Mat before = cropped.clone();

        detector.detect(cropped);

        const int changed = changedPixelCount(before, cropped);
        best_changed_pixels = std::max(best_changed_pixels, changed);

        if (changed >= min_changed_pixels) {
            SUCCEED() << "Model drew detections on frame " << frame_index
                      << " with " << changed << " changed pixels.";
            return;
        }
    }

    FAIL() << "Model inference completed, but no bounding-box output was drawn. "
           << "Best changed pixel count: " << best_changed_pixels;
}
