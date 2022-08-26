

#include "vp_ba_enter_analyser.h"

namespace vp_ba {
        
    vp_ba_enter_analyser::vp_ba_enter_analyser(/* args */) {
    }
    
    vp_ba_enter_analyser::~vp_ba_enter_analyser() {
    }

    vp_ba_flag vp_ba_enter_analyser::ba_ability() {
        return vp_ba::vp_ba_flag::ENTER;
    }

    std::vector<std::tuple<std::shared_ptr<vp_objects::vp_frame_element>, std::shared_ptr<vp_objects::vp_frame_target>, vp_ba::vp_ba_flag>> 
        vp_ba_enter_analyser::analyse(std::shared_ptr<vp_objects::vp_frame_element>& element, std::vector<std::shared_ptr<vp_objects::vp_frame_target>>& targets) {
            // assert
            assert(element->check_ba_ability(this->ba_ability()));
            assert(element->key_points().size() > 2);

            // logic

            return {};
        }
}