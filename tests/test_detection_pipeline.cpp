#include <gtest/gtest.h>

#include <algorithm>
#include <future>
#include <stdexcept>
#include <string>
#include <vector>

#include "DetectionPipeline.hpp"

namespace {

// Frames are 1x1 CV_32S images holding their read index.
int frameIndex(const cv::Mat& frame) {
    return frame.at<int>(0, 0);
}

// Reads `count` frames. Like cv::VideoCapture::read, it writes into the given
// Mat with create(), which reuses the buffer if the Mat already has one.
struct CountingReader {
    int count;
    int next = 0;
    std::vector<std::string>* log = nullptr;

    bool operator()(cv::Mat& frame) {
        if (next >= count) return false;
        frame.create(1, 1, CV_32S);
        frame.at<int>(0, 0) = next;
        if (log) log->push_back("read " + std::to_string(next));
        ++next;
        return true;
    }
};

// Synchronous detector using the default detect_async().
class SyncDetector : public Detection {
public:
    explicit SyncDetector(std::vector<std::string>& log)
        : Detection(0.5f, cv::Size2f(1, 1), "fake.onnx"), log_(log) {}
    void detect(cv::Mat& frame) override {
        log_.push_back("detect " + std::to_string(frameIndex(frame)));
    }

private:
    std::vector<std::string>& log_;
};

// Pipelined detector: the work of a frame runs only when its future is
// waited on, so frames are in flight between detect_async() and get().
class PipelinedDetector : public Detection {
public:
    explicit PipelinedDetector(std::size_t depth)
        : Detection(0.5f, cv::Size2f(1, 1), "fake.onnx"), depth_(depth) {}

    std::size_t max_frames_in_flight() const override { return depth_; }

    void detect(cv::Mat& frame) override { detect_async(frame).get(); }

    std::future<void> detect_async(cv::Mat frame) override {
        ++in_flight;
        max_seen = std::max(max_seen, in_flight);
        return std::async(std::launch::deferred, [this, frame]() mutable {
            --in_flight;
            if (frameIndex(frame) == fail_at) throw std::runtime_error("inference failed");
            frame.at<int>(0, 0) += 1000;  // "draws" the detections into the frame
        });
    }

    std::size_t in_flight = 0;
    std::size_t max_seen = 0;
    int fail_at = -1;

private:
    std::size_t depth_;
};

}  // namespace

TEST(DetectionPipeline, SynchronousDetectorFinishesEachFrameBeforeReadingTheNext) {
    std::vector<std::string> log;
    SyncDetector detector(log);
    CountingReader reader{2, 0, &log};

    const bool ok = detection_pipeline::run(detector, reader, [&](cv::Mat& frame) {
        log.push_back("deliver " + std::to_string(frameIndex(frame)));
        return true;
    });

    EXPECT_TRUE(ok);
    EXPECT_EQ(log, (std::vector<std::string>{"read 0", "detect 0", "deliver 0",
                                             "read 1", "detect 1", "deliver 1"}));
}

TEST(DetectionPipeline, KeepsUpToMaxFramesInFlightAndDeliversInReadOrder) {
    PipelinedDetector detector(3);
    CountingReader reader{7};
    std::vector<int> delivered;

    detection_pipeline::run(detector, reader, [&](cv::Mat& frame) {
        delivered.push_back(frameIndex(frame));
        return true;
    });

    EXPECT_EQ(detector.max_seen, 3u);
    EXPECT_EQ(delivered, (std::vector<int>{1000, 1001, 1002, 1003, 1004, 1005, 1006}));
}

// The first frame runs alone, so its time shows the backend's warm-up (lazy
// model loading, graph compilation) instead of also reading ahead.
TEST(DetectionPipeline, FinishesFirstFrameBeforeReadingAhead) {
    std::vector<std::string> log;
    PipelinedDetector detector(3);
    CountingReader reader{5, 0, &log};

    detection_pipeline::run(detector, reader, [&](cv::Mat& frame) {
        log.push_back("deliver " + std::to_string(frameIndex(frame) - 1000));
        return true;
    });

    ASSERT_GE(log.size(), 5u);
    EXPECT_EQ(std::vector<std::string>(log.begin(), log.begin() + 5),
              (std::vector<std::string>{"read 0", "deliver 0", "read 1", "read 2", "read 3"}));
    EXPECT_EQ(detector.max_seen, 3u);
}

TEST(DetectionPipeline, DeliversFramesStillInFlightWhenReadingStops) {
    PipelinedDetector detector(4);
    CountingReader reader{2};
    std::vector<int> delivered;

    detection_pipeline::run(detector, reader, [&](cv::Mat& frame) {
        delivered.push_back(frameIndex(frame));
        return true;
    });

    EXPECT_EQ(delivered, (std::vector<int>{1000, 1001}));
    EXPECT_EQ(detector.in_flight, 0u);
}

TEST(DetectionPipeline, StopsReadingWhenDeliveryFails) {
    PipelinedDetector detector(3);
    CountingReader reader{10};
    int delivered = 0;

    const bool ok = detection_pipeline::run(detector, reader, [&](cv::Mat&) {
        return ++delivered < 2;
    });

    EXPECT_FALSE(ok);
    EXPECT_EQ(delivered, 2);
    EXPECT_LE(reader.next, 2 + 3);
    EXPECT_EQ(detector.in_flight, 0u);  // remaining frames were waited for
}

TEST(DetectionPipeline, PropagatesDetectionErrors) {
    PipelinedDetector detector(3);
    detector.fail_at = 1;
    CountingReader reader{5};

    EXPECT_THROW(detection_pipeline::run(detector, reader, [](cv::Mat&) { return true; }),
                 std::runtime_error);
}
