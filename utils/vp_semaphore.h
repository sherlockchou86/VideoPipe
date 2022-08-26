
#pragma once

#include <condition_variable>
#include <mutex>

namespace vp_utils {
    // semaphore for queue menbers in vp_node.
    // used to block the consumer thread until queue has data.
    class vp_semaphore
    {
    public:
        vp_semaphore() {
            count_ = 0;
        }

        void signal() {
            std::unique_lock<std::mutex> lock(mutex_);
            ++count_;
            cv_.notify_one();
        }

        void wait() {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [=] { return count_ > 0; });
            --count_;
        }

    private:
        std::mutex mutex_;
        std::condition_variable cv_;
        int count_;
    };
}