#pragma once

#include "../vp_node.h"

namespace vp_nodes {
    // video/image recording node, save it to local disk.
    // it is a middle node but works asynchronously, so recording would not block the pipeline.
    class vp_record_node: public vp_node
    {
    private:
        /* data */
    protected:
        // re-implementation
        virtual std::shared_ptr<vp_objects::vp_meta> handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) override; 
        // re-implementation
        virtual std::shared_ptr<vp_objects::vp_meta> handle_control_meta(std::shared_ptr<vp_objects::vp_control_meta> meta) override;
    public:
        vp_record_node(std::string node_name, 
                        std::string video_save_dir, 
                        std::string image_save_dir, 
                        int pre_record_video_duration = 5, 
                        int record_video_duration = 20,
                        bool auto_sub_dir = true,
                        vp_objects::vp_size resolution_w_h = {},
                        int bitrate = 1024,
                        bool osd = false);
        ~vp_record_node();

        // video save directory
        std::string video_save_dir;
        // image save directory
        std::string image_save_dir;

        // pre record time for video (seconds)
        int pre_record_video_duration;
        // record time for video (seconds), not including pre_record_video_duration
        int record_video_duration;

        // create sub directory by date and channel index or not, such as `video_save_dir/2022-10-8/channel_index/**.mp4`
        bool auto_sub_dir;

        vp_objects::vp_size resolution_w_h = {};
        int bitrate = 1024;
        bool osd = false;
    };
}
