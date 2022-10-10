#pragma once

#include <queue>
#include <thread>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "../../objects/vp_frame_meta.h"

namespace vp_nodes {
    // base class for record task
    class vp_record_task {
    private:
        int channel_index;
        std::string file_name_without_ext;
        std::string save_dir;
        bool auto_sub_dir;
        vp_objects::vp_size resolution_w_h;
        bool osd;
    protected:
        // preprocess
        void preprocess(std::shared_ptr<vp_objects::vp_frame_meta>& frame_to_record, cv::Mat& data);
        // file extension override bu child class
        virtual std::string get_file_ext() = 0;
    public:
        // get full path for recording file
        std::string get_full_path() const;

        vp_record_task(int channel_index, 
                        std::string file_name_without_ext, 
                        std::string save_dir, 
                        bool auto_sub_dir, 
                        vp_objects::vp_size resolution_w_h, 
                        bool osd);
        ~vp_record_task();
    };

}