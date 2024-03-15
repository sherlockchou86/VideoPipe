

#include "vp_udp_src_node.h"
#include "../utils/vp_utils.h"

namespace vp_nodes {
        
    vp_udp_src_node::vp_udp_src_node(std::string node_name, 
                                    int channel_index, 
                                    int port, 
                                    float resize_ratio,
                                    std::string gst_decoder_name,
                                    int skip_interval):
                                    vp_src_node(node_name, channel_index, resize_ratio),
                                    port(port), gst_decoder_name(gst_decoder_name), skip_interval(skip_interval) {
        assert(skip_interval >= 0 && skip_interval <= 9);
        this->gst_template = vp_utils::string_format(this->gst_template, port, gst_decoder_name.c_str());
        VP_INFO(vp_utils::string_format("[%s] [%s]", node_name.c_str(), gst_template.c_str()));
        this->initialized();
    }
    
    vp_udp_src_node::~vp_udp_src_node() {
        deinitialized();
    }

    // define how to receive video stream via udp, create frame meta etc.
    // please refer to the implementation of vp_node::handle_run.
    void vp_udp_src_node::handle_run() {
        cv::Mat frame;     
        int video_width = 0;
        int video_height = 0;
        int fps = 0;
        int skip = 0;
        while(alive) {
            // check if need work
            gate.knock();

            // try to open capture
            if (!udp_capture.isOpened()) {
                if (!udp_capture.open(this->gst_template, cv::CAP_GSTREAMER)) {
                    VP_WARN(vp_utils::string_format("[%s] open udp failed, try again...", node_name.c_str()));
                    continue;
                }
            }

            // video properties
            if (video_width == 0 || video_height == 0 || fps == 0) {
                video_width = udp_capture.get(cv::CAP_PROP_FRAME_WIDTH);
                video_height = udp_capture.get(cv::CAP_PROP_FRAME_HEIGHT);
                fps = udp_capture.get(cv::CAP_PROP_FPS);

                original_fps = fps;
                original_width = video_width;
                original_height = video_height;

                // set true fps because skip some frames
                fps = fps / (skip_interval + 1);
            }
            // stream_info_hooker activated if need
            vp_stream_info stream_info {channel_index, original_fps, original_width, original_height, to_string()};
            invoke_stream_info_hooker(node_name, stream_info);

            udp_capture >> frame;
            if(frame.empty()) {
                VP_WARN(vp_utils::string_format("[%s] reading frame empty, total frame==>%d", node_name.c_str(), frame_index));
                continue;
            }

            // need skip
            if (skip < skip_interval) {
                skip++;
                continue;
            }
            skip = 0;
            
            // need resize
            cv::Mat resize_frame;
            if (this->resize_ratio != 1.0f) {                 
                cv::resize(frame, resize_frame, cv::Size(), resize_ratio, resize_ratio);
            }
            else {
                resize_frame = frame.clone(); // clone!;
            }
            // set true size because resize
            video_width = resize_frame.cols;
            video_height = resize_frame.rows;

            this->frame_index++;
            // create frame meta
            auto out_meta = 
                std::make_shared<vp_objects::vp_frame_meta>(resize_frame, this->frame_index, this->channel_index, video_width, video_height, fps);

            if (out_meta != nullptr) {
                this->out_queue.push(out_meta);

                // handled hooker activated if need
                if (this->meta_handled_hooker) {
                    meta_handled_hooker(node_name, out_queue.size(), out_meta);
                }

                // important! notify consumer of out_queue in case it is waiting.
                this->out_queue_semaphore.signal();
                VP_DEBUG(vp_utils::string_format("[%s] after handling meta, out_queue.size()==>%d", node_name.c_str(), out_queue.size()));
            }       
        }

        // send dead flag for dispatch_thread
        this->out_queue.push(nullptr);
        this->out_queue_semaphore.signal();    
    }

    // return stream uri
    std::string vp_udp_src_node::to_string() {
        return "udp://127.0.0.1:" + std::to_string(port);
    }
}