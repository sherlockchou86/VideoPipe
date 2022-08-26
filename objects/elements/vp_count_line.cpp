

#include "vp_count_line.h"

namespace vp_objects {
    // vp_ba::vp_ba_flag::CROSSLINE only
    vp_count_line::vp_count_line(int element_id, 
                                vp_point start, 
                                vp_point end,
                                std::string element_name): 
        vp_frame_element(element_id, element_name, vp_ba::vp_ba_flag::CROSSLINE),
        line(start, end) {

    }
    
    vp_count_line::~vp_count_line() {

    }

    std::shared_ptr<vp_frame_element> vp_count_line::clone() {
        return std::make_shared<vp_objects::vp_count_line>(*this);
    }

    std::vector<vp_objects::vp_point> vp_count_line::key_points() {
        return {this->line.start, this->line.end};
    }
}