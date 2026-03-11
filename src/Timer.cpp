#include "Timer.hpp"


Timer::Timer() : is_running_(false) {}

void Timer::start() {
    start_time_ = std::chrono::high_resolution_clock::now();
    is_running_ = true;
}

void Timer::stop() {
    end_time_ = std::chrono::high_resolution_clock::now();
    is_running_ = false;
}

double Timer::getElapsedTime() const {
    if (is_running_) {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(end - start_time_).count();
    } else {
        return std::chrono::duration<double>(end_time_ - start_time_).count();
    }
}