#ifndef OPENCV_OBJECT_DETECTION_TIMER_HPP
#define OPENCV_OBJECT_DETECTION_TIMER_HPP

#include <chrono>

class Timer {
public:
    Timer();
    ~Timer() = default;
    void start();
    void stop();
    double getElapsedTime() const;

private:
    std::chrono::high_resolution_clock::time_point start_time_, end_time_;
    bool is_running_;
};

#endif //OPENCV_OBJECT_DETECTION_TIMER_HPP