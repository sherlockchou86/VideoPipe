#pragma once

#include "vp_ba_analyser.h"

namespace vp_ba {
    // built-in analyser
    // crossline analyser, check if target cross over a line.
    class vp_ba_crossline_analyser: public vp_ba_analyser {
    private:
        /* data */
    public:
        vp_ba_crossline_analyser(/* args */);
        ~vp_ba_crossline_analyser();

        virtual vp_ba_flag ba_ability() override;
        virtual std::vector<std::tuple<std::shared_ptr<vp_objects::vp_frame_element>, std::shared_ptr<vp_objects::vp_frame_target>, vp_ba::vp_ba_flag>> 
            analyse(std::shared_ptr<vp_objects::vp_frame_element>& element, std::vector<std::shared_ptr<vp_objects::vp_frame_target>>& targets) override;
    };

}