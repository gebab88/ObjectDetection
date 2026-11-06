#include "DetectionPipeline.hpp"

#include <algorithm>
#include <deque>
#include <future>
#include <utility>

namespace detection_pipeline {

bool run(Detection& detection,
         const std::function<bool(cv::Mat&)>& read,
         const std::function<bool(cv::Mat&)>& deliver) {
    struct InFlight {
        cv::Mat frame;
        std::future<void> done;
    };
    const std::size_t depth = std::max<std::size_t>(1, detection.max_frames_in_flight());
    std::deque<InFlight> in_flight;

    // On early exit (delivery failure or detection error) wait for the frames
    // still in progress, so no backend work outlives the frames it draws into.
    struct Drain {
        std::deque<InFlight>& queue;
        ~Drain() {
            for (auto& item : queue) {
                if (item.done.valid()) item.done.wait();
            }
        }
    } drain{in_flight};

    // The first frame runs alone: backends may finish their setup lazily on it,
    // and its time should show that warm-up, not reading ahead.
    bool warmed_up = false;

    auto finishOldest = [&]() {
        InFlight item = std::move(in_flight.front());
        in_flight.pop_front();
        item.done.get();  // rethrows detection errors
        warmed_up = true;
        return deliver(item.frame);
    };

    while (true) {
        while (in_flight.size() >= (warmed_up ? depth : 1)) {
            if (!finishOldest()) return false;
        }
        cv::Mat frame;  // fresh buffer: frames in flight must not be overwritten
        if (!read(frame)) break;
        std::future<void> done = detection.detect_async(frame);
        in_flight.push_back({std::move(frame), std::move(done)});
    }
    while (!in_flight.empty()) {
        if (!finishOldest()) return false;
    }
    return true;
}

}  // namespace detection_pipeline
