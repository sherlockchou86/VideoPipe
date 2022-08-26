#pragma once

#include "vp_frame_element.h"
#include "../shapes/vp_polygon.h"

namespace vp_objects {
    // built-in element, a polygon with more than 2 vertexs in video scene.
    // note: 
    // vp_region can be the base class of other elements which have more than 2 vertexs.
    class vp_region: public vp_frame_element {
    private:
         vp_objects::vp_polygon polygon;
    public:
        vp_region(int element_id, 
                std::vector<vp_point> vertexs, 
                std::string element_name = "", 
                int ba_abilities_mask = 0);
        ~vp_region();

        // more than 2 points for vp_region
        virtual std::vector<vp_objects::vp_point> key_points() override;

        // clone myself
        virtual std::shared_ptr<vp_frame_element> clone() override;

        // check if the region contains a point
        bool contains(const vp_point & p);
    };
}