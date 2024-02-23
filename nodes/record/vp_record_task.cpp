
#include "vp_record_task.h"

namespace vp_nodes {
    vp_record_task::vp_record_task(int channel_index, 
                    std::string file_name_without_ext, 
                    std::string save_dir, 
                    bool auto_sub_dir, 
                    vp_objects::vp_size resolution_w_h, 
                    bool osd,
                    std::string host_node_name):
                    channel_index(channel_index),
                    file_name_without_ext(file_name_without_ext),
                    save_dir(save_dir),
                    auto_sub_dir(auto_sub_dir),
                    resolution_w_h(resolution_w_h),
                    osd(osd),
                    host_node_name(host_node_name) {

    }

    vp_record_task::~vp_record_task() {

    }

    void vp_record_task::stop_task() {
        status = vp_record_task_status::NOSTRAT;
        frames_to_record.push_back(nullptr);
        cache_semaphore.signal();
        if (record_task_th.joinable()) {
            record_task_th.join();
        }        
    }
    std::string vp_record_task::get_full_record_path() {
        // full_record_path already generated
        if (!full_record_path.empty()) {
            return full_record_path;
        }
        
        // check save dir
        if (!std::experimental::filesystem::exists(save_dir)) {
            VP_INFO(vp_utils::string_format("[%s] [record] save dir not exists, now creating save dir: `%s`", host_node_name.c_str(), save_dir.c_str()));
            std::experimental::filesystem::create_directories(save_dir);
        }

        // do not generate sub folder
        if (!auto_sub_dir) {
            std::experimental::filesystem::path p1(save_dir);
            std::experimental::filesystem::path p2(file_name_without_ext + get_file_ext());
            
            // ./video/abc.mp4
            auto p = p1 / p2;
            if (std::experimental::filesystem::exists(p)) {
                // just check once
                auto new_file_name = file_name_without_ext + "_" + std::to_string(NOW.time_since_epoch().count()) + get_file_ext();
                VP_WARN(vp_utils::string_format("[%s] [record] `%s` already exists, changing to: `%s`", host_node_name.c_str(), p2.string().c_str(), new_file_name.c_str()));
                p = p1 / new_file_name;
            }
            full_record_path = p.string();
        }
        else {
            // generate sub folder by date and channel
            std::experimental::filesystem::path p1(save_dir);
            // just use year-mon-day
            std::experimental::filesystem::path p2(vp_utils::time_format(std::chrono::system_clock::now(), "<year>-<mon>-<day>"));
            std::experimental::filesystem::path p3(std::to_string(channel_index));
            std::experimental::filesystem::path p4(file_name_without_ext + get_file_ext());

            auto p1_3 = p1 / p2 / p3;
            if (!std::experimental::filesystem::exists(p1_3)) {
                VP_INFO(vp_utils::string_format("[%s] [record] sub dir not exists, now creating sub dir: `%s`", host_node_name.c_str(), p1_3.string().c_str()));
                std::experimental::filesystem::create_directories(p1_3);
            }
            
            // ./video/2022-10-10/1/abc.mp4
            auto p = p1_3 / p4;
            if (std::experimental::filesystem::exists(p)) {
                // just check once
                auto new_file_name = file_name_without_ext + "_" + std::to_string(NOW.time_since_epoch().count()) + get_file_ext();
                VP_WARN(vp_utils::string_format("[%s] [record] `%s` already exists, changing to: `%s`", host_node_name.c_str(), p4.string().c_str(), new_file_name.c_str()));
                p = p1_3 / new_file_name;
            }
            
            full_record_path = p.string();
        }
        VP_INFO(vp_utils::string_format("[%s] [record] get full record path: `%s`", host_node_name.c_str(), full_record_path.c_str()));
        return full_record_path;
    }

    void vp_record_task::preprocess(std::shared_ptr<vp_objects::vp_frame_meta>& frame_to_record, cv::Mat& data) {
        cv::Mat resize_frame;
        if (this->resolution_w_h.width != 0 && this->resolution_w_h.height != 0) {                 
            cv::resize((osd && !frame_to_record->osd_frame.empty()) ? frame_to_record->osd_frame : frame_to_record->frame, 
                        resize_frame, 
                        cv::Size(resolution_w_h.width, resolution_w_h.height));
        }
        else {
            resize_frame = (osd && !frame_to_record->osd_frame.empty()) ? frame_to_record->osd_frame : frame_to_record->frame;
        }

        resize_frame.copyTo(data);
    }

    void vp_record_task::set_task_complete_hooker(vp_record_task_complete_hooker task_complete_hooker) {
        // override if already set
        this->task_complete_hooker = task_complete_hooker;
    }

    void vp_record_task::notify_task_complete(vp_record_info record_info) {
        status = vp_record_task_status::COMPLETE;
        if (task_complete_hooker) {
            // notify to host
            // fill fields defined in base class
            record_info.channel_index = channel_index;
            record_info.file_name_without_ext = file_name_without_ext;
            record_info.full_record_path = get_full_record_path();
            record_info.osd = osd;

            task_complete_hooker(channel_index, record_info);
        }
    }

    void vp_record_task::start() {
        // check status
        if (status != vp_record_task_status::NOSTRAT) {
            return;
        }
        status = vp_record_task_status::STARTED;
        record_task_th = std::thread(&vp_record_task::record_task_run, this);
    }

    
    void vp_record_task::append_async(std::shared_ptr<vp_objects::vp_frame_meta> frame_meta) {
        // can append data only if NOSTART or STARTED
        if (status == vp_record_task_status::COMPLETE) {
            return;
        }
        
        // just push data into queue, it is a producer
        // note, since only one thread push, no lock needed here
        frames_to_record.push_back(frame_meta);
        cache_semaphore.signal();
    }  
}