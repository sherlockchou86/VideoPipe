

#pragma once

#include <vector>
#include "vp_point.h"

namespace vp_objects {
    class vp_polygon
    {
    private:
        /* data */
    public:
        vp_polygon() = default;
        vp_polygon(std::vector<vp_point> vertexs);
        ~vp_polygon();

        // vertexs of the polygon
        std::vector<vp_point> vertexs;

        // check if the polygon contains a point
        bool contains(const vp_point & p);
    };

}