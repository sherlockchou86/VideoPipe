#include "vp_frame_face_target.h"

namespace vp_objects {
        
    vp_frame_face_target::vp_frame_face_target(int x, 
                                                int y, 
                                                int width, 
                                                int height, 
                                                float score, 
                                                std::vector<std::pair<int, int>> key_points, 
                                                std::vector<float> embeddings):
                                                x(x),
                                                y(y),
                                                width(width),
                                                height(height),
                                                score(score),
                                                key_points(key_points),
                                                embeddings(embeddings) {
        
    }
    
    vp_frame_face_target::~vp_frame_face_target() {
    }

    std::shared_ptr<vp_frame_face_target> vp_frame_face_target::clone() {
        return std::make_shared<vp_frame_face_target>(*this);
    }

    vp_rect vp_frame_face_target::get_rect() const{
        return vp_rect(x, y, width, height);
    }
}