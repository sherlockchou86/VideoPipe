

#include "vp_log_file_writer.h"


namespace vp_utils {
        
    vp_log_file_writer::vp_log_file_writer() {

    }
    
    vp_log_file_writer::~vp_log_file_writer() {
        if (log_writer.is_open()) {
            log_writer.close();
        }
    }
    
    void vp_log_file_writer::init(std::string log_dir, std::string log_file_name_template) {
        this->log_dir = log_dir;
        this->log_file_name_template = log_file_name_template;
        
        // open log file first time
        auto f = create_valid_log_file_name();
        log_writer.open(f, std::ofstream::out | std::ofstream::app);

        inited = true;
    }


    void vp_log_file_writer::write(std::string log) {
        if (!inited) {
            throw "vp_log_file_writer not initialized!";
        }
        
        // check if need create new log file
        if (get_now_day() != log_day) {
            if (log_writer.is_open()) {
                log_writer.close();
            }

            auto f = create_valid_log_file_name();
            log_writer.open(f, std::ofstream::out | std::ofstream::app);
        }
        
        log_writer << log << std::endl;
    }

    std::string vp_log_file_writer::create_valid_log_file_name() {
        if (!std::experimental::filesystem::exists(log_dir)) {
            std::experimental::filesystem::create_directories(log_dir);
        }
        std::experimental::filesystem::path root_dir(log_dir);

        auto f_name = vp_utils::time_format(NOW, log_file_name_template);
        auto p = root_dir / f_name;
        
        // cache log start day
        log_day = get_now_day();

        return p.string();
    }

    int vp_log_file_writer::get_now_day() {
        std::vector<int> time_parts;
        vp_utils::time_split(NOW, time_parts);
        
        // refer to vp_utils::time_split(...), indice 2 is day
        return time_parts[2];
    }

    vp_log_file_writer& vp_log_file_writer::operator<<(std::string log) {
        write(log);
    }
}