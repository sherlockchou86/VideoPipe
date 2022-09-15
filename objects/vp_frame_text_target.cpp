
#include "vp_frame_text_target.h"

namespace vp_objects {
        
    vp_frame_text_target::vp_frame_text_target(std::vector<std::pair<int, int>> region_vertexes, 
                                                std::string text, 
                                                float score):
                                                region_vertexes(region_vertexes),
                                                text(text),
                                                score(score) {

    }
    
    vp_frame_text_target::~vp_frame_text_target() {
    
    }

    std::shared_ptr<vp_frame_text_target> vp_frame_text_target::clone() {
        return std::make_shared<vp_frame_text_target>(*this);
    }
}