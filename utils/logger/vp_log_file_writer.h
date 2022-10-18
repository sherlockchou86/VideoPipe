#pragma once

#include <string>
#include <fstream>
#include <chrono>
#include <experimental/filesystem>

#include "../vp_utils.h"

namespace vp_utils {
    // log file writer, write log to local file. auto create new log file bye date.
    // NOT thread safe, MUST use in single thread.
    class vp_log_file_writer
    {
    private:
        // ready to go
        bool inited = false;
        
        // root dir for log saving
        std::string log_dir;
        // log file name template, such as `<year>-<mon>-<day>.txt`, refer to vp_utils::time_format(...) for rule details
        std::string log_file_name_template;

        // file handle
        std::ofstream log_writer;
        // get valid log file name including path, name and extension
        std::string create_valid_log_file_name();

        // current log day (1 ~ 31)
        int log_day = 0;

        // get day of now
        int get_now_day();

        #define NOW std::chrono::system_clock::now()
    public:
        vp_log_file_writer();
        ~vp_log_file_writer();
        
        // write log
        void write(std::string log);

        // initialize writer
        void init(std::string log_dir, std::string log_file_name_template);

        // for << operator
        vp_log_file_writer& operator<<(std::string log);
    };
}