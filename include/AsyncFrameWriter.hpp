#ifndef ASYNC_FRAME_WRITER_HPP
#define ASYNC_FRAME_WRITER_HPP

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>

#include <opencv2/core.hpp>

// Hands frames to a write function running on its own thread, in order.
//
// Encoding the output video is the most expensive step per frame for large
// frames (e.g. 12 ms for 1080x1080 MPEG-4 on a Pi 5), so it runs next to
// reading and detection instead of after them. The queue is bounded: push()
// blocks while it is full, so a slow writer slows the run down instead of
// piling up frames in memory.
class AsyncFrameWriter {
public:
    using WriteFunction = std::function<bool(const cv::Mat&)>;

    explicit AsyncFrameWriter(WriteFunction write, std::size_t capacity = 8);
    ~AsyncFrameWriter();  // close()

    AsyncFrameWriter(const AsyncFrameWriter&) = delete;
    AsyncFrameWriter& operator=(const AsyncFrameWriter&) = delete;

    // Queues the frame (which must not be modified afterwards). Returns false,
    // without queueing it, once a write has failed.
    bool push(cv::Mat frame);

    // Writes the frames still queued and stops the thread. Returns false if a
    // write failed. Further calls return the same result.
    bool close();

private:
    void run();

    WriteFunction write_;
    std::size_t capacity_;
    std::mutex mutex_;
    std::condition_variable changed_;
    std::deque<cv::Mat> queue_;
    bool closing_ = false;
    bool failed_ = false;
    std::thread thread_;  // last: starts after the other members exist
};

#endif  // ASYNC_FRAME_WRITER_HPP
