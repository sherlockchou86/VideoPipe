
#pragma once

#include "vp_ba_analyser.h"

namespace vp_ba {
    // built-in analyser
    // stop analyser, check if target keep still or no moving.
    class vp_ba_stop_analyser: public vp_ba_analyser
    {
    private:
        /* data */
    public:
        vp_ba_stop_analyser(/* args */);
        ~vp_ba_stop_analyser();

        
        virtual vp_ba_flag ba_ability() override;
        virtual std::vector<std::tuple<std::shared_ptr<vp_objects::vp_frame_element>, std::shared_ptr<vp_objects::vp_frame_target>, vp_ba::vp_ba_flag>> 
            analyse(std::shared_ptr<vp_objects::vp_frame_element>& element, std::vector<std::shared_ptr<vp_objects::vp_frame_target>>& targets) override;
    };

}