#pragma once

#include <condition_variable>
#include <mutex>

namespace vp_utils {
    // semaphore used to resume/pause loop structure, it blocks thread while received unactive signal and unblock thread while received active signal.
    // refer to vp_semaphore also
    class vp_gate
    {
    public:
        vp_gate() {
            opened_ = false;
        }

        void open() {
            std::unique_lock<std::mutex> lock(mutex_);
            opened_ = true;
            cv_.notify_one();
        }

        // wait until opened
        void knock() {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [=] { return opened_; });
        }

        void close() {
            std::unique_lock<std::mutex> lock(mutex_);
            opened_ = false;
        }

        bool is_open() {
            return opened_;
        }
    private:
        std::mutex mutex_;
        std::condition_variable cv_;
        bool opened_;
    };
}