#pragma once

#include <vector>
#include <string>
#include <memory>
#include "../shapes/vp_point.h"


namespace vp_objects {
    // type of behaviour analysis
    enum class vp_ba_type {
        NONE = 0b00000000,       // none
        CROSSLINE = 0b00000001,  // cross line
        STOP = 0b00000010,       // enter stop status
        UNSTOP = 0b00000100,     // leave stop status
        JAM = 0b00001000,        // enter jam status
        UNJAM = 0b00010000       // leave jam status
        /* more */
    };

    // result of behaviour analysis
    // BA logic can ONLY works on vp_frame_target
    class vp_ba_result
    {
    private:
        /* data */
    public:
        // type
        vp_ba_type type;
        // target ids which involved for this ba result, empty allowed.
        std::vector<int> involve_target_ids_in_frame;
        // region (or single line) involved for this ba result, empty allowed.
        std::vector<vp_objects::vp_point> involve_region_in_frame;

        // channel index of this ba result
        int channel_index;
        // frame index of this ba result
        int frame_index;

        // name of ba
        std::string ba_label = "not specified";

        // record image name if exist
        std::string record_image_name = "";
        // record video name if exist
        std::string record_video_name = "";

        vp_ba_result(vp_ba_type type, 
                    int channel_index,
                    int frame_index,
                    std::vector<int> involve_target_ids_in_frame, 
                    std::vector<vp_objects::vp_point> involve_region_in_frame,
                    std::string ba_label = "not specified",
                    std::string record_image_name = "",
                    std::string record_video_name = "");
        ~vp_ba_result();

        // get description for ba result
        virtual std::string to_string();

        // clone myself
        std::shared_ptr<vp_ba_result> clone();
    };

}