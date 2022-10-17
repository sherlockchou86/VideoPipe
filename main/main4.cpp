
#include "VP.h"

#include "../utils/vp_utils.h"
#include "../utils/logger/vp_logger.h"

#include <iostream>
#include <chrono>
#include <thread>

/*
* sample for vp_logger
*/


#if MAIN4

int main() {
    // config
    VP_SET_LOG_INCLUDE_THREAD_ID(false);
    VP_SET_LOG_LEVEL(vp_utils::vp_log_level::WARN);
    
    // init
    VP_LOGGER_INIT();

    // 6 threads logging separately
    auto func1 = []() {
        while (true) {
            /* code */
            auto id = std::this_thread::get_id();
            std::stringstream ss;
            ss << std::hex << id;
            auto thread_id = ss.str(); 
            VP_ERROR(vp_utils::string_format("thread id: %s", thread_id.c_str()));
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        
    };

    auto func2 = []() {
        while (true) {
            /* code */
            auto id = std::this_thread::get_id();
            std::stringstream ss;
            ss << std::hex << id;
            auto thread_id = ss.str(); 
            VP_DEBUG(vp_utils::string_format("thread id: %s", thread_id.c_str()));
            std::this_thread::sleep_for(std::chrono::milliseconds(13));
        }
        
    };

    auto func3 = []() {
        while (true) {
            /* code */
            auto id = std::this_thread::get_id();
            std::stringstream ss;
            ss << std::hex << id;
            auto thread_id = ss.str(); 
            VP_INFO(vp_utils::string_format("thread id: %s", thread_id.c_str()));
            std::this_thread::sleep_for(std::chrono::milliseconds(4));
        }
        
    };

    auto func4 = []() {
        while (true) {
            /* code */
            auto id = std::this_thread::get_id();
            std::stringstream ss;
            ss << std::hex << id;
            auto thread_id = ss.str(); 
            VP_WARN(vp_utils::string_format("thread id: %s", thread_id.c_str()));
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        
    };

    auto func5 = []() {
        while (true) {
            /* code */
            auto id = std::this_thread::get_id();
            std::stringstream ss;
            ss << std::hex << id;
            auto thread_id = ss.str(); 
            VP_ERROR(vp_utils::string_format("thread id: %s", thread_id.c_str()));
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        
    };

    auto func6 = []() {
        while (true) {
            /* code */
            auto id = std::this_thread::get_id();
            std::stringstream ss;
            ss << std::hex << id;
            auto thread_id = ss.str(); 
            VP_INFO(vp_utils::string_format("thread id: %s", thread_id.c_str()));
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        
    };

    std::thread t1(func1);
    std::thread t2(func2);
    std::thread t3(func3);
    std::thread t4(func4);
    std::thread t5(func5);
    std::thread t6(func6);

    t1.join();
    t2.join();
    t3.join();
    t4.join();
    t5.join();
    t6.join();

    std::getchar();
}

#endif