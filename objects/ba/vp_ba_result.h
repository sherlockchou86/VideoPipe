#pragma once

#include <vector>
#include <string>
#include <memory>
#include "../shapes/vp_point.h"


namespace vp_objects {
    // type of behaviour analysis
    enum class vp_ba_type {
        NONE = 0b00000000,
        CROSSLINE = 0b00000001,
        STOP = 0b00000010
        /* more */
    };


    // result of behaviour analysis
    class vp_ba_result
    {
    private:
        /* data */
    public:
        // type
        vp_ba_type type;
        // target ids which involved for this ba result, empty allowed.
        /* ids can be from any targets inside vp_frame_meta, such as vp_frame_target/vp_face_target/... */
        std::vector<int> involve_target_ids_in_frame;
        // region (or single line) involved for this ba result, empty allowed.
        std::vector<vp_objects::vp_point> involve_region_in_frame;

        // channel index of this ba result
        int channel_index;
        // frame index of this ba result
        int frame_index;

        vp_ba_result(vp_ba_type type, 
                    int channel_index,
                    int frame_index,
                    std::vector<int> involve_target_ids_in_frame, 
                    std::vector<vp_objects::vp_point> involve_region_in_frame);
        ~vp_ba_result();

        // get description for ba result
        virtual std::string to_string();

        // clone myself
        std::shared_ptr<vp_ba_result> clone();
    };

}