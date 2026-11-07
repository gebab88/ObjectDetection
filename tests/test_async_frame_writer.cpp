#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <future>
#include <mutex>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>

#include "AsyncFrameWriter.hpp"

namespace {

using namespace std::chrono_literals;

cv::Mat frameWithValue(int value) {
    return cv::Mat(1, 1, CV_32S, cv::Scalar(value));
}

// Records what the writer thread received.
struct Recorder {
    std::mutex mutex;
    std::vector<int> values;
    std::vector<std::thread::id> threads;

    bool operator()(const cv::Mat& frame) {
        std::lock_guard<std::mutex> lock(mutex);
        values.push_back(frame.at<int>(0, 0));
        threads.push_back(std::this_thread::get_id());
        return true;
    }
};

}  // namespace

TEST(AsyncFrameWriter, WritesAllFramesInOrderOnAnotherThread) {
    Recorder recorder;
    AsyncFrameWriter writer([&](const cv::Mat& f) { return recorder(f); });

    for (int i = 0; i < 20; ++i) {
        ASSERT_TRUE(writer.push(frameWithValue(i)));
    }
    ASSERT_TRUE(writer.close());

    std::vector<int> expected(20);
    for (int i = 0; i < 20; ++i) expected[i] = i;
    EXPECT_EQ(recorder.values, expected);
    for (const auto& id : recorder.threads) {
        EXPECT_NE(id, std::this_thread::get_id());
    }
}

TEST(AsyncFrameWriter, CloseWaitsForQueuedFrames) {
    std::atomic<int> written{0};
    AsyncFrameWriter writer([&](const cv::Mat&) {
        std::this_thread::sleep_for(5ms);
        ++written;
        return true;
    });

    for (int i = 0; i < 5; ++i) writer.push(frameWithValue(i));
    EXPECT_TRUE(writer.close());
    EXPECT_EQ(written.load(), 5);
}

TEST(AsyncFrameWriter, ReportsFailedWriteAndStopsWriting) {
    std::atomic<int> calls{0};
    AsyncFrameWriter writer([&](const cv::Mat&) { return ++calls < 2; });  // second write fails

    writer.push(frameWithValue(0));
    writer.push(frameWithValue(1));
    // Once the failure is known, further frames are refused.
    bool refused = false;
    for (int i = 2; i < 200 && !refused; ++i) {
        refused = !writer.push(frameWithValue(i));
        std::this_thread::sleep_for(1ms);
    }

    EXPECT_TRUE(refused);
    EXPECT_FALSE(writer.close());
    EXPECT_EQ(calls.load(), 2);
}

TEST(AsyncFrameWriter, PushBlocksWhileQueueIsFull) {
    std::promise<void> gate;
    std::shared_future<void> open = gate.get_future().share();
    AsyncFrameWriter writer([open](const cv::Mat&) { open.wait(); return true; }, 2);

    writer.push(frameWithValue(0));  // taken by the writer thread, which then waits
    std::this_thread::sleep_for(20ms);
    writer.push(frameWithValue(1));  // queued
    writer.push(frameWithValue(2));  // queued: queue full

    auto blocked = std::async(std::launch::async, [&] { return writer.push(frameWithValue(3)); });
    EXPECT_EQ(blocked.wait_for(50ms), std::future_status::timeout);

    gate.set_value();
    EXPECT_EQ(blocked.wait_for(2s), std::future_status::ready);
    EXPECT_TRUE(blocked.get());
    EXPECT_TRUE(writer.close());
}

TEST(AsyncFrameWriter, DestructorFinishesQueuedFrames) {
    std::atomic<int> written{0};
    {
        AsyncFrameWriter writer([&](const cv::Mat&) { ++written; return true; });
        for (int i = 0; i < 3; ++i) writer.push(frameWithValue(i));
    }
    EXPECT_EQ(written.load(), 3);
}
