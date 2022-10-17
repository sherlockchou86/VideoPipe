
#include "vp_logger.h"

namespace vp_utils {
        
    vp_logger::vp_logger(/* args */)
    {
    }
    
    vp_logger::~vp_logger()
    {
    }

    void vp_logger::init() {
        inited = true;
        auto t = std::thread(&vp_logger::log_write_run, this); 
        log_writer_th = std::move(t);
    }

    void vp_logger::log(vp_log_level level, const std::string& message, const char* code_file, int code_line) {
        // make sure logger is initialized
        if (!inited) {
            throw "vp_logger is not initialized yet!";
        }
        
        // filter
        if (level > log_level) {
            return;
        }
        
        /* create log */
        std::string new_log = "";
        // 100% true for log time
        if (include_time) {
            new_log += vp_utils::time_format(NOW, log_time_templete);
        }

        // log level
        if (include_level) {
            new_log += "[" + log_level_names.at(level) + "]";
        }

        // code location
        if (include_code_location) {
            new_log += "[" + std::string(code_file) + ":" + std::to_string(code_line) + "]";
        }

        // thread id
        if (include_thread_id) {
            auto id = std::this_thread::get_id();
            std::stringstream ss;
            ss << std::hex << id;  // to hex
            auto thread_id = ss.str(); 
            new_log += "[" + thread_id + "]";
        }
        
        new_log += " " + message;

        /* write to cache */
        std::lock_guard<std::mutex> guard(log_cache_mutex);
        log_cache.push(new_log);
        // notify
        log_cache_semaphore.signal();
    }

    void vp_logger::log_write_run() {
        while (inited) {
            // wait for data
            log_cache_semaphore.wait();
            auto log = log_cache.front();
            log_cache.pop();

            /* write to devices */

            if (log_to_console) {
                write_to_console(log);
            }
            
            if (log_to_file) {
                write_to_file(log);
            }

            if (log_to_kafka) {
                write_to_kafka(log);
            }
        }
    }

    void vp_logger::write_to_console(const std::string& log) {
        std::cout << log << std::endl;
    }

    void vp_logger::write_to_file(const std::string& log) {

    }

    void vp_logger::write_to_kafka(const std::string& log) {
        // TO-DO
    }
}