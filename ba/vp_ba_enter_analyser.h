#pragma once

#include "vp_ba_analyser.h"

namespace vp_ba {
    // built-in analyser
    // enter analyser, check if target enter a region.
    class vp_ba_enter_analyser: public vp_ba_analyser {
    private:
        /* data */
    public:
        vp_ba_enter_analyser(/* args */);
        ~vp_ba_enter_analyser();

        virtual vp_ba_flag ba_ability() override;
        virtual std::vector<std::tuple<std::shared_ptr<vp_objects::vp_frame_element>, std::shared_ptr<vp_objects::vp_frame_target>, vp_ba::vp_ba_flag>> 
            analyse(std::shared_ptr<vp_objects::vp_frame_element>& element, std::vector<std::shared_ptr<vp_objects::vp_frame_target>>& targets) override;
    };
}