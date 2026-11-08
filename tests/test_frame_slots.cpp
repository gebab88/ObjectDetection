#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <future>
#include <stdexcept>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>

#include "FrameSlots.hpp"

namespace {

using namespace std::chrono_literals;

cv::Mat frameWithValue(int value) {
    return cv::Mat(1, 1, CV_32S, cv::Scalar(value));
}

struct Scratch {
    int uses = 0;
};

}  // namespace

TEST(FrameSlots, RunsWorkOnAnotherThreadAndWritesIntoTheFrame) {
    FrameSlots<Scratch> slots(1, "[Test]");
    cv::Mat frame = frameWithValue(1);
    std::thread::id worker;
    slots.submit(frame, [&](Scratch&, cv::Mat& f) {
        worker = std::this_thread::get_id();
        f.at<int>(0, 0) = 42;
    }).get();
    EXPECT_NE(worker, std::this_thread::get_id());
    EXPECT_EQ(frame.at<int>(0, 0), 42);  // same pixel buffer as the caller's frame
}

TEST(FrameSlots, RunsFramesConcurrentlyWithOwnScratch) {
    FrameSlots<Scratch> slots(2, "[Test]");
    std::atomic<int> running{0};
    std::atomic<int> max_running{0};
    std::vector<Scratch*> scratches(2, nullptr);
    auto work = [&](int index) {
        return [&, index](Scratch& scratch, cv::Mat&) {
            const int now = ++running;
            int seen = max_running.load();
            while (now > seen && !max_running.compare_exchange_weak(seen, now)) {}
            scratches[index] = &scratch;
            const auto deadline = std::chrono::steady_clock::now() + 2s;
            while (max_running.load() < 2 && std::chrono::steady_clock::now() < deadline) {
                std::this_thread::sleep_for(1ms);
            }
            --running;
        };
    };
    auto first = slots.submit(frameWithValue(0), work(0));
    auto second = slots.submit(frameWithValue(1), work(1));
    first.get();
    second.get();
    EXPECT_EQ(max_running.load(), 2);
    EXPECT_NE(scratches[0], scratches[1]);
}

TEST(FrameSlots, ReusesSlotsRoundRobinOnceCollected) {
    FrameSlots<Scratch> slots(2, "[Test]");
    std::vector<int> uses;
    for (int i = 0; i < 6; ++i) {
        slots.submit(frameWithValue(i), [&](Scratch& s, cv::Mat&) {
            uses.push_back(++s.uses);
        }).get();
    }
    EXPECT_EQ(uses, (std::vector<int>{1, 1, 2, 2, 3, 3}));
}

TEST(FrameSlots, ThrowsWhenMoreFramesThanSlotsAreInFlight) {
    FrameSlots<Scratch> slots(1, "[Test]");
    std::promise<void> release;
    std::shared_future<void> released = release.get_future().share();
    auto first = slots.submit(frameWithValue(0), [released](Scratch&, cv::Mat&) {
        released.wait();
    });
    EXPECT_THROW(slots.submit(frameWithValue(1), [](Scratch&, cv::Mat&) {}), std::logic_error);
    release.set_value();
    first.get();
    // The slot is free again after the frame was collected.
    EXPECT_NO_THROW(slots.submit(frameWithValue(2), [](Scratch&, cv::Mat&) {}).get());
}

TEST(FrameSlots, RethrowsWorkErrorsAndFreesTheSlot) {
    FrameSlots<Scratch> slots(1, "[Test]");
    auto failed = slots.submit(frameWithValue(0), [](Scratch&, cv::Mat&) {
        throw std::runtime_error("inference failed");
    });
    EXPECT_THROW(failed.get(), std::runtime_error);
    EXPECT_NO_THROW(slots.submit(frameWithValue(1), [](Scratch&, cv::Mat&) {}).get());
}

TEST(FrameSlots, DestructionWaitsForRunningFrames) {
    std::atomic<bool> finished{false};
    {
        FrameSlots<Scratch> slots(1, "[Test]");
        // The caller's future is dropped: only the destructor can wait now.
        (void)slots.submit(frameWithValue(0), [&](Scratch&, cv::Mat&) {
            std::this_thread::sleep_for(50ms);
            finished = true;
        });
    }
    EXPECT_TRUE(finished.load());
}
