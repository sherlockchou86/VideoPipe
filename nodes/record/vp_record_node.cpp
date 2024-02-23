#include "vp_record_node.h"


namespace vp_nodes {
        
    vp_record_node::vp_record_node(std::string node_name, 
                                    std::string video_save_dir, 
                                    std::string image_save_dir,
                                    vp_objects::vp_size resolution_w_h, 
                                    bool osd,
                                    int pre_record_video_duration, 
                                    int record_video_duration,
                                    bool auto_sub_dir,
                                    int bitrate):
                                    vp_node(node_name),
                                    video_save_dir(video_save_dir),
                                    image_save_dir(image_save_dir),
                                    resolution_w_h(resolution_w_h),
                                    osd(osd),
                                    pre_record_video_duration(pre_record_video_duration),
                                    record_video_duration(record_video_duration),
                                    auto_sub_dir(auto_sub_dir),
                                    bitrate(bitrate) {
        this->initialized();
    }
    
    vp_record_node::~vp_record_node() {
        deinitialized();
    }

    std::shared_ptr<vp_objects::vp_meta> vp_record_node::handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) {
        // cache fps for current channel
        if (all_fps.count(meta->channel_index) == 0) {
            all_fps[meta->channel_index] = meta->fps;
        }
        auto& fps = all_fps[meta->channel_index];

        // first time for current channel
        if (all_pre_records.count(meta->channel_index) == 0) {
            all_pre_records[meta->channel_index] = std::deque<std::shared_ptr<vp_objects::vp_frame_meta>>();
        }
        auto& pre_records = all_pre_records[meta->channel_index];
        
        // update pre_records for current channel
        pre_records.push_back(meta);
        auto frames_need_pre_record = fps * pre_record_video_duration;
        // keep max frames
        if (pre_records.size() > frames_need_pre_record) {
            pre_records.pop_front();
        }
        
        // first time for current channel
        if (all_record_tasks.count(meta->channel_index) == 0) {
            all_record_tasks[meta->channel_index] = std::list<std::shared_ptr<vp_nodes::vp_record_task>>();
        }
        auto& record_tasks = all_record_tasks[meta->channel_index];

        // then append data to all tasks of current channel
        for (auto i = record_tasks.begin(); i != record_tasks.end();) {
            if ((*i)->status == vp_nodes::vp_record_task_status::COMPLETE) {
                i = record_tasks.erase(i); // remove task which is complete already
            }
            else {
                (*i)->append_async(meta);  // no block here
                i++;
            }
        }    
        
        // done 
        return meta;
    }

    std::shared_ptr<vp_objects::vp_meta> vp_record_node::handle_control_meta(std::shared_ptr<vp_objects::vp_control_meta> meta) {
        
        if (meta->control_type == vp_objects::vp_control_type::IMAGE_RECORD ||
            meta->control_type == vp_objects::vp_control_type::VIDEO_RECORD) {
            /* code */
            // create record task, it will start asynchronously.
            auto_new_record_task(meta);

            /* no return since already handle it, do not need pass to next nodes. */
            return nullptr;
        }
        else {
            return vp_node::handle_control_meta(meta);
        }
    }

    void vp_record_node::auto_new_record_task(std::shared_ptr<vp_objects::vp_control_meta>& meta) {
        auto& record_tasks = all_record_tasks[meta->channel_index];
        auto& pre_records = all_pre_records[meta->channel_index];
        auto& fps = all_fps[meta->channel_index];

        // image record
        if (meta->control_type == vp_objects::vp_control_type::IMAGE_RECORD) {
            auto image_record_control_meta = std::dynamic_pointer_cast<vp_objects::vp_image_record_control_meta>(meta);
            
            // create image record task
            auto file_name_without_ext = image_record_control_meta->image_file_name_without_ext;
            auto _osd = image_record_control_meta->osd;
            VP_INFO(vp_utils::string_format("[%s] [record] create new image record task, file_name_without_ext is: `%s`", node_name.c_str(), file_name_without_ext.c_str()));
            auto image_record_task = std::make_shared<vp_nodes::vp_image_record_task>(meta->channel_index, 
                                                                                    file_name_without_ext, 
                                                                                    image_save_dir, 
                                                                                    auto_sub_dir, 
                                                                                    _osd, 
                                                                                    resolution_w_h, node_name);
            image_record_task->set_task_complete_hooker([this](int channel_index, vp_record_info record_info) {
                // just notify hooker which has attached on the node
                if (this->image_record_complete_hooker) {
                    this->image_record_complete_hooker(channel_index, record_info);
                }
                VP_INFO(vp_utils::string_format("[%s] [record] image record task completed, file_name_without_ext is: `%s`", node_name.c_str(), record_info.file_name_without_ext.c_str()));
            });
            record_tasks.push_back(image_record_task);
        }

        // video record
        if (meta->control_type == vp_objects::vp_control_type::VIDEO_RECORD) {
            auto video_record_control_meta = std::dynamic_pointer_cast<vp_objects::vp_video_record_control_meta>(meta);
            
            // create video record task
            auto file_name_without_ext = video_record_control_meta->video_file_name_without_ext;
            auto _osd = video_record_control_meta->osd;
            auto _record_video_duration = video_record_control_meta->record_video_duration;
            if (_record_video_duration == 0) {
                // use default value
                _record_video_duration = record_video_duration;
            }
            VP_INFO(vp_utils::string_format("[%s] [record] create new video record task, file_name_without_ext is: `%s`", node_name.c_str(), file_name_without_ext.c_str()));
            auto video_record_task = std::make_shared<vp_nodes::vp_video_record_task>(meta->channel_index, 
                                                                                    pre_records, 
                                                                                    file_name_without_ext, 
                                                                                    video_save_dir, 
                                                                                    auto_sub_dir, 
                                                                                    _osd, 
                                                                                    resolution_w_h, 
                                                                                    bitrate, fps, pre_record_video_duration, _record_video_duration, node_name);
            video_record_task->set_task_complete_hooker([this](int channel_index, vp_record_info record_info) {
                // just notify hooker which has attached on the node
                if (this->video_record_complete_hooker) {                
                    this->video_record_complete_hooker(channel_index, record_info);
                }
                VP_INFO(vp_utils::string_format("[%s] [record] video record task completed, file_name_without_ext is: `%s`", node_name.c_str(), record_info.file_name_without_ext.c_str()));
            });
            record_tasks.push_back(video_record_task);
        }
    }
}