#include "AsyncFrameWriter.hpp"

#include <utility>

AsyncFrameWriter::AsyncFrameWriter(WriteFunction write, std::size_t capacity)
    : write_(std::move(write)),
      capacity_(capacity > 0 ? capacity : 1),
      thread_(&AsyncFrameWriter::run, this) {}

AsyncFrameWriter::~AsyncFrameWriter() {
    close();
}

bool AsyncFrameWriter::push(cv::Mat frame) {
    std::unique_lock<std::mutex> lock(mutex_);
    changed_.wait(lock, [&] { return failed_ || queue_.size() < capacity_; });
    if (failed_ || closing_) {
        return false;
    }
    queue_.push_back(std::move(frame));
    changed_.notify_all();
    return true;
}

bool AsyncFrameWriter::close() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        closing_ = true;
    }
    changed_.notify_all();
    if (thread_.joinable()) {
        thread_.join();
    }
    std::lock_guard<std::mutex> lock(mutex_);
    return !failed_;
}

void AsyncFrameWriter::run() {
    while (true) {
        cv::Mat frame;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            changed_.wait(lock, [&] { return closing_ || !queue_.empty(); });
            if (queue_.empty()) {
                return;  // closing and everything written
            }
            frame = std::move(queue_.front());
            queue_.pop_front();
        }
        changed_.notify_all();  // room for a blocked push()

        if (!write_(frame)) {
            std::lock_guard<std::mutex> lock(mutex_);
            failed_ = true;
            queue_.clear();  // the output is broken; drop the rest
            changed_.notify_all();
            return;
        }
    }
}
