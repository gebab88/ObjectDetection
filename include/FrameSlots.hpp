#ifndef OBJECTDETECTION_FRAMESLOTS_HPP
#define OBJECTDETECTION_FRAMESLOTS_HPP

#include <atomic>
#include <cstddef>
#include <exception>
#include <future>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

// Runs the frames of a backend on worker threads, one per frame in flight,
// each with its own Scratch buffers (e.g. the input blob and output tensor),
// so a backend can implement Detection::detect_async on top of a blocking
// per-frame routine.
//
// Slots are used round-robin. DetectionPipeline collects frames in order and
// never has more than size() in flight, so the next slot is always free; more
// frames than slots throw std::logic_error. Destroying the FrameSlots waits for
// the frames still running, so destroy it before anything the work uses.
template <typename Scratch>
class FrameSlots {
public:
    FrameSlots(std::size_t count, std::string owner) : owner_(std::move(owner)) {
        for (std::size_t i = 0; i < count; ++i) {
            slots_.push_back(std::make_unique<Slot>());
        }
    }

    std::size_t size() const { return slots_.size(); }

    // Calls work(scratch, frame) on a worker thread. The returned future
    // becomes ready when the work is done and rethrows its exceptions. The
    // slot is released before that, so the caller may submit again as soon as
    // it has collected the frame.
    template <typename Work>
    std::future<void> submit(cv::Mat frame, Work work) {
        Slot& slot = *slots_[next_];
        if (slot.busy.exchange(true)) {
            throw std::logic_error(owner_ + " more than max_frames_in_flight() frames in flight");
        }
        next_ = (next_ + 1) % slots_.size();

        auto finished = std::make_shared<std::promise<void>>();
        std::future<void> result = finished->get_future();
        try {
            slot.done = std::async(std::launch::async,
                                   [&slot, frame, finished, work = std::move(work)]() mutable {
                try {
                    work(slot.scratch, frame);
                    slot.busy = false;
                    finished->set_value();
                } catch (...) {
                    slot.busy = false;
                    finished->set_exception(std::current_exception());
                }
            });
        } catch (...) {
            slot.busy = false;  // no thread started (std::system_error)
            throw;
        }
        return result;
    }

private:
    struct Slot {
        Scratch scratch;
        std::atomic<bool> busy{false};
        std::future<void> done;  // last member: waited for before the scratch goes
    };

    std::string owner_;
    std::vector<std::unique_ptr<Slot>> slots_;
    std::size_t next_ = 0;
};

#endif  // OBJECTDETECTION_FRAMESLOTS_HPP
