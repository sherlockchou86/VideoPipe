
#include <iostream>


#include "vp_file_src_node.h"

namespace vp_nodes {
        
    vp_file_src_node::vp_file_src_node(std::string node_name, 
                                        int channel_index, 
                                        std::string file_path, 
                                        float resize_ratio, 
                                        bool cycle): 
                                        vp_src_node(node_name, channel_index, resize_ratio), 
                                        file_path(file_path), 
                                        cycle(cycle) {
        initialized();
    }
    
    vp_file_src_node::~vp_file_src_node() {

    }
    
    // define how to read video from local file, create frame meta etc.
    // please refer to the implementation of vp_node::handle_run.
    void vp_file_src_node::handle_run() {
        cv::Mat frame;
        int video_width = 0;
        int video_height = 0;
        int fps = 0;
        std::chrono::milliseconds delta;

        while(true) {
            if (this->active) {
                auto last_time = std::chrono::system_clock::now();
                // open capture
                if (!file_capture.isOpened()) {
                    assert(file_capture.open(this->file_path, cv::CAP_FFMPEG));
                }

                // video properties
                if (video_width == 0 || video_height == 0 || fps == 0) {
                    video_width = file_capture.get(cv::CAP_PROP_FRAME_WIDTH);
                    video_height = file_capture.get(cv::CAP_PROP_FRAME_HEIGHT);
                    fps = file_capture.get(cv::CAP_PROP_FPS);
                    delta = std::chrono::milliseconds(1000 / fps);
     
                    original_fps = fps;
                    original_width = video_width;
                    original_height = video_height;

                    // stream_info_hooker activated if need
                    if (stream_info_hooker) {
                        vp_stream_info stream_info {channel_index, fps, video_width, video_height, file_path};
                        stream_info_hooker(node_name, stream_info);
                    }
                }
                
                file_capture >> frame;
                if(frame.empty()) {
                    std::cout << "frame is empty, total frame==>" << this->frame_index << std::endl;
                    if (cycle) {
                        file_capture.set(cv::CAP_PROP_POS_FRAMES, 0);
                    }
                    continue;
                }

                // need resize
                cv::Mat resize_frame;
                if (this->resize_ratio != 1.0f) {                 
                    cv::resize(frame, resize_frame, cv::Size(), resize_ratio, resize_ratio);
                }
                else {
                    resize_frame = frame.clone();  // clone!
                }

                this->frame_index++;
                // create frame meta
                auto out_meta = 
                    std::make_shared<vp_objects::vp_frame_meta>(resize_frame, this->frame_index, this->channel_index, video_width, video_height, fps);

                if (out_meta != nullptr) {
                    //std::cout << this->node_name << ", handle meta, before out_queue.size()==>" << this->out_queue.size() << std::endl;
                    this->out_queue.push(out_meta);

                    // handled hooker activated if need
                    if (this->meta_handled_hooker) {
                        meta_handled_hooker(node_name, out_queue.size(), out_meta);
                    }

                    // important! notify consumer of out_queue in case it is waiting.
                    this->out_queue_semaphore.signal();
                    std::cout << this->node_name << ", handle meta, after out_queue.size()==>" << this->out_queue.size() << std::endl;
                }

                // for fps
                auto snap = std::chrono::system_clock::now() - last_time;
                snap = std::chrono::duration_cast<std::chrono::milliseconds>(snap);
                if (snap < delta) {
                    std::this_thread::sleep_for(delta - snap);
                }              
            }
            else {
                std::this_thread::sleep_for(std::chrono::microseconds{1000});
            }
        }
    }

    // return stream path
    std::string vp_file_src_node::to_string() {
        return file_path;
    }
}