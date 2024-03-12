
#include "vp_logger.h"

namespace vp_utils {
        
    vp_logger::vp_logger(/* args */)
    {
    }
    
    vp_logger::~vp_logger() {
        die();
        if (log_writer_th.joinable()) {
            log_writer_th.join();
        }
    }

    void vp_logger::die() {
        alive = false;
        std::lock_guard<std::mutex> guard(log_cache_mutex);
        log_cache.push("die");
        log_cache_semaphore.signal();
    }

    void vp_logger::init() {
        inited = true;

        // initialize file writer
        file_writer.init(log_dir, log_file_name_template);

        // run thread
        auto t = std::thread(&vp_logger::log_write_run, this); 
        log_writer_th = std::move(t);
    }

    void vp_logger::log(vp_log_level level, const std::string& message, const char* code_file, int code_line) {
        // make sure logger is initialized
        if (!inited) {
            throw "vp_logger is not initialized yet!";
        }
        
        // level filter
        if (level > log_level) {
            return;
        }
        
        // keywords filter for debug level
        if (level == vp_log_level::DEBUG && keywords_for_debug_log.size() != 0) {
            bool filterd = true;
            for(auto& keywords: keywords_for_debug_log) {
                if (message.find(keywords) != std::string::npos) {
                    filterd = false;
                    break;
                }
            }
            if (filterd) {
                return;
            }
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

        // thread id
        if (include_thread_id) {
            auto id = std::this_thread::get_id();
            std::stringstream ss;
            ss << std::hex << id;  // to hex
            auto thread_id = ss.str(); 
            new_log += "[" + thread_id + "]";
        }
        
        // code location
        if (include_code_location) {
            new_log += "[" + std::string(code_file) + ":" + std::to_string(code_line) + "]";
        }

        new_log += " " + message;

        /* write to cache */
        // min lock range
        std::lock_guard<std::mutex> guard(log_cache_mutex);
        log_cache.push(new_log);
        // notify
        log_cache_semaphore.signal();
    }

    void vp_logger::log_write_run() {
        bool log_thres_warned = false;
        /* below code runs in single thread */
        while (inited && alive) {
            // wait for data
            log_cache_semaphore.wait();
            auto log = log_cache.front();
            log_cache.pop();

            if (log == "die") {
                continue;
            }
            
            /* watch the log cache size */
            auto log_cache_size = 0;
            {
                // min lock range
                std::lock_guard<std::mutex> guard(log_cache_mutex);
                log_cache_size = log_cache.size();
            }
            if (!log_thres_warned && log_cache_size > log_cache_warn_threshold) {
                VP_WARN(vp_utils::string_format("[logger] log cache size is exceeding threshold! cache size is: [%d], threshold is: [%d]", log_cache_size, log_cache_warn_threshold));
                log_thres_warned = true;  // warn 1 time
            }
            if (log_cache_size <= log_cache_warn_threshold) {
                log_thres_warned = false;
            }
            
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
        // file_writer.write(log);
        file_writer << log;
    }

    void vp_logger::write_to_kafka(const std::string& log) {
        // TO-DO
    }
}