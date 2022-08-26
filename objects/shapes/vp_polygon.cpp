
#include <assert.h>
#include "vp_polygon.h"

namespace vp_objects {
        
    vp_polygon::vp_polygon(std::vector<vp_point> vertexs): vertexs(vertexs) {
        assert(vertexs.size() > 2);
    }
    
    vp_polygon::~vp_polygon() {

    }
    
    bool vp_polygon::contains(const vp_point & p) {
        return true;
    }
}