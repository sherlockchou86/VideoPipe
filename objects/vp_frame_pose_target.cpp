
#include "vp_frame_pose_target.h"

namespace vp_objects {
    
    vp_frame_pose_target::vp_frame_pose_target(vp_pose_type type, 
                                                std::vector<vp_pose_keypoint> key_points):
                                                type(type),
                                                key_points(key_points) {

    }
    
    vp_frame_pose_target::~vp_frame_pose_target() {
    }

    std::shared_ptr<vp_frame_pose_target> vp_frame_pose_target::clone() {
        return std::make_shared<vp_frame_pose_target>(*this);
    }
}