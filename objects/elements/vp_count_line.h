#pragma once

#include "vp_frame_element.h"
#include "../shapes/vp_line.h"

namespace vp_objects {
    // built-in element, a line with 2 vertexs in video scene.
    // usage case: counter increases by 1 if a target cross over the line.
    class vp_count_line: public vp_frame_element {
    private:
        vp_objects::vp_line line;
    public:
        vp_count_line(int element_id, vp_point start, vp_point end, std::string element_name = "");
        ~vp_count_line();

        // 2 points for vp_count_line
        virtual std::vector<vp_objects::vp_point> key_points() override;

        // clone myself
        virtual std::shared_ptr<vp_frame_element> clone() override;
    };

}