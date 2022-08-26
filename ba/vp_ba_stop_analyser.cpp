
#include "vp_ba_stop_analyser.h"

namespace vp_ba {
        
    vp_ba_stop_analyser::vp_ba_stop_analyser(/* args */)
    {
    }
    
    vp_ba_stop_analyser::~vp_ba_stop_analyser()
    {
    }
    
    vp_ba_flag vp_ba_stop_analyser::ba_ability() {
        return vp_ba::vp_ba_flag::STOP;
    }

    std::vector<std::tuple<std::shared_ptr<vp_objects::vp_frame_element>, std::shared_ptr<vp_objects::vp_frame_target>, vp_ba::vp_ba_flag>> 
        vp_ba_stop_analyser::analyse(std::shared_ptr<vp_objects::vp_frame_element>& element, std::vector<std::shared_ptr<vp_objects::vp_frame_target>>& targets) {
            assert(element->check_ba_ability(this->ba_ability()));
            assert(element->key_points().size() > 2);

            // logic

            return {};
    }
}