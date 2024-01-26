#pragma once

#include <list>
#include <deque>
#include <map>

#include "../vp_node.h"
#include "../../objects/vp_image_record_control_meta.h"
#include "../../objects/vp_video_record_control_meta.h"

#include "vp_image_record_task.h"
#include "vp_video_record_task.h"
#include "vp_record_status_hookable.h"

namespace vp_nodes {
    // video/image recording node, save it to local disk.
    // it is a middle node but works asynchronously, so recording would not block the pipeline.
    // note record node could work on multi channels at the same time.
    class vp_record_node: public vp_node, public vp_record_status_hookable
    {
    private:
        /* config data */
        // video save directory
        std::string video_save_dir;
        // image save directory
        std::string image_save_dir;
        // pre record time for video (seconds)
        int pre_record_video_duration;
        // record time for video (seconds), not including pre_record_video_duration
        int record_video_duration;
        // auto create sub directory by date and channel or not, such as `./video_save_dir/2022-10-8/1/**.mp4`
        bool auto_sub_dir;
        // width and height
        vp_objects::vp_size resolution_w_h = {};
        // bitrate for video record
        int bitrate = 1024;
        // record osd frame or not
        bool osd = false;

        // fps for current video
        // int fps = 0;
        std::map<int, int> all_fps;
        
        /* record task list */
        // std::list<std::shared_ptr<vp_nodes::vp_record_task>> record_tasks;
        std::map<int, std::list<std::shared_ptr<vp_nodes::vp_record_task>>> all_record_tasks;

        /* pre-record for video */
        // std::deque<std::shared_ptr<vp_objects::vp_frame_meta>> pre_records;
        std::map<int, std::deque<std::shared_ptr<vp_objects::vp_frame_meta>>> all_pre_records;

        // new record task
        void auto_new_record_task(std::shared_ptr<vp_objects::vp_control_meta>& meta);

    protected:
        // re-implementation
        virtual std::shared_ptr<vp_objects::vp_meta> handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) override; 
        // re-implementation
        virtual std::shared_ptr<vp_objects::vp_meta> handle_control_meta(std::shared_ptr<vp_objects::vp_control_meta> meta) override;
    public:
        vp_record_node(std::string node_name, 
                        std::string video_save_dir, 
                        std::string image_save_dir,
                        vp_objects::vp_size resolution_w_h = {}, 
                        bool osd = false,
                        int pre_record_video_duration = 5, 
                        int record_video_duration = 20,
                        bool auto_sub_dir = true,
                        int bitrate = 1024);
        ~vp_record_node();
    };
}
