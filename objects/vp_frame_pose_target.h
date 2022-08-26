#pragma once

#include <vector>
#include <memory>

namespace vp_objects {

    // different types of datasets used to train openpose model.
    enum vp_pose_type {
        body_25,
        coco,
        mpi_15,
        face,
        hand
    };
    
    struct vp_pose_keypoint {
        int point_type;       // point type, nose, neck or left_eye 
        int x;                // x in 2D image
        int y;                // y in 2D image
        float score;          // probability
    };
    
    // target in frame detected by openpose(or other similar models), which mainly contains point collections.
    // note: we can define new target type like vp_frame_xxx_target... if need (see vp_frame_face_target also)
    class vp_frame_pose_target
    {
    private:
        /* data */
    public:
        vp_frame_pose_target(vp_pose_type type, std::vector<vp_pose_keypoint> key_points);
        ~vp_frame_pose_target();

        // target type, different models create different outputs which need specific parsing.
        vp_pose_type type;
        // keypoints array
        std::vector<vp_pose_keypoint> key_points;

        // clone myself
        std::shared_ptr<vp_frame_pose_target> clone();
    };
}