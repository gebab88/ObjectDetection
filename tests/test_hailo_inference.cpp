// End-to-end tests of the Hailo backend on the real accelerator. Built only on
// a Raspberry Pi with HailoRT and local HEF files; skipped when no Hailo
// device is connected or the test video is missing.
#include <gtest/gtest.h>

#include <cstddef>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include "DetectionPipeline.hpp"
#include "Detection_Hailo.hpp"
#include "FrameCrop.hpp"
#include "YoloPostprocess.hpp"
#include "hailo/hailort.hpp"

#ifndef HAILO_TEST_HEFS
#define HAILO_TEST_HEFS ""
#endif
#ifndef HAILO_TEST_VIDEO
#define HAILO_TEST_VIDEO ""
#endif
#ifndef HAILO_TEST_CLASSES
#define HAILO_TEST_CLASSES ""
#endif
#ifndef HAILO_TEST_INCOMPATIBLE_HEF
#define HAILO_TEST_INCOMPATIBLE_HEF ""
#endif

namespace {

constexpr int kFrames = 40;

// The build passes the discovered HEFs as a single '|'-separated string.
std::vector<std::string> testHefs() {
    std::vector<std::string> hefs;
    std::stringstream stream{std::string(HAILO_TEST_HEFS)};
    std::string path;
    while (std::getline(stream, path, '|')) {
        if (!path.empty()) hefs.push_back(path);
    }
    return hefs;
}

bool hailoDeviceConnected() {
    auto devices = hailort::Device::scan();
    return devices && !devices->empty();
}

// First frames of the test video, cropped like the application does.
std::vector<cv::Mat> loadFrames(int count) {
    std::vector<cv::Mat> frames;
    cv::VideoCapture capture(HAILO_TEST_VIDEO);
    cv::Mat frame;
    while (static_cast<int>(frames.size()) < count && capture.read(frame)) {
        frames.push_back(frame(frame_crop::centeredSquare(frame.cols, frame.rows)).clone());
    }
    return frames;
}

std::unique_ptr<Detection_Hailo> makeDetector(const std::string& hef) {
    auto detector = std::make_unique<Detection_Hailo>(0.4f, 0.65f, cv::Size2f(640, 640), hef);
    if (!detector->load_class_list(HAILO_TEST_CLASSES)) {
        throw std::runtime_error(std::string("cannot load ") + HAILO_TEST_CLASSES);
    }
    return detector;
}

// Restores the global label logging after a test.
struct LogDetections {
    LogDetections() { yolo_postprocess::logDetectionsToStdout() = true; }
    ~LogDetections() { yolo_postprocess::logDetectionsToStdout() = false; }
};

}  // namespace

class HailoInference : public ::testing::TestWithParam<std::string> {
protected:
    void SetUp() override {
        if (!hailoDeviceConnected()) GTEST_SKIP() << "No Hailo device connected";
        if (!std::filesystem::exists(HAILO_TEST_VIDEO)) {
            GTEST_SKIP() << "Test video not found: " << HAILO_TEST_VIDEO;
        }
        frames_ = loadFrames(kFrames);
        ASSERT_EQ(static_cast<int>(frames_.size()), kFrames) << "Cannot read " << HAILO_TEST_VIDEO;
    }

    std::vector<cv::Mat> frames_;
};

// Frames processed with several in flight must come out exactly like frames
// processed one at a time: same order, no buffer of another frame mixed in.
TEST_P(HailoInference, PipelinedFramesMatchSynchronouslyProcessedFrames) {
    auto detector = makeDetector(GetParam());
    ASSERT_GT(detector->max_frames_in_flight(), 1u);

    std::vector<cv::Mat> expected;
    for (const cv::Mat& frame : frames_) {
        cv::Mat copy = frame.clone();
        detector->detect(copy);
        expected.push_back(copy);
    }

    std::size_t next_read = 0;
    std::size_t delivered = 0;
    const bool ok = detection_pipeline::run(
        *detector,
        [&](cv::Mat& frame) {
            if (next_read == frames_.size()) return false;
            frame = frames_[next_read++].clone();
            return true;
        },
        [&](cv::Mat& frame) {
            EXPECT_EQ(cv::norm(frame, expected[delivered], cv::NORM_INF), 0.0)
                << "frame " << delivered << " differs";
            ++delivered;
            return true;
        });

    EXPECT_TRUE(ok);
    EXPECT_EQ(delivered, frames_.size());
}

TEST_P(HailoInference, DetectsPeopleInTestVideo) {
    auto detector = makeDetector(GetParam());
    LogDetections log;

    ::testing::internal::CaptureStdout();
    for (cv::Mat& frame : frames_) {
        detector->detect(frame);
    }
    const std::string labels = ::testing::internal::GetCapturedStdout();

    EXPECT_NE(labels.find("person"), std::string::npos);
}

INSTANTIATE_TEST_SUITE_P(
    Hefs, HailoInference, ::testing::ValuesIn(testHefs()),
    [](const ::testing::TestParamInfo<std::string>& info) {
        std::string name = std::filesystem::path(info.param).stem().string();
        for (char& c : name) {
            if (!std::isalnum(static_cast<unsigned char>(c))) c = '_';
        }
        return name;
    });

// Optional: a HEF compiled for another Hailo chip (-DHAILO_TEST_INCOMPATIBLE_HEF=...).
TEST(HailoIncompatibleHef, ExplainsWhichChipTheHefIsFor) {
    const std::string hef = HAILO_TEST_INCOMPATIBLE_HEF;
    if (hef.empty() || !std::filesystem::exists(hef)) {
        GTEST_SKIP() << "Set HAILO_TEST_INCOMPATIBLE_HEF to a HEF for another Hailo chip";
    }
    if (!hailoDeviceConnected()) GTEST_SKIP() << "No Hailo device connected";

    try {
        Detection_Hailo detector(0.4f, 0.65f, cv::Size2f(640, 640), hef);
        FAIL() << "loading " << hef << " should fail";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("is compiled for"), std::string::npos) << e.what();
    }
}
