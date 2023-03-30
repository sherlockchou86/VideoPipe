
#include "vp_image_src_node.h"

namespace vp_nodes {
    vp_image_src_node::vp_image_src_node(std::string node_name, 
                                        int channel_index, 
                                        std::string port_or_location,
                                        int interval,
                                        float resize_ratio, 
                                        bool cycle):
                                        vp_src_node(node_name, channel_index, resize_ratio),
                                        port_or_location(port_or_location),
                                        interval(interval),
                                        cycle(cycle) {
        // make sure not greater than 1 minute (too long) and not lower than 1 second (since it's too quick, use video stream instead directly)
        assert(interval >= 1 && interval <= 60);
        if (vp_utils::ends_with(port_or_location, "jpeg") || vp_utils::ends_with(port_or_location, "jpg")) {
            // read from file

            gst_template_file = vp_utils::string_format(gst_template_file, port_or_location.c_str(), cycle ? std::string("true").c_str() : std::string("false").c_str(), interval);
            from_file = true;
        }
        else if (port_or_location.find_first_not_of("0123456789") == std::string::npos) {
            // receive from remote via udp
            auto port = std::stoi(port_or_location); // try to get port

            gst_template_udp = vp_utils::string_format(gst_template_udp, port, interval);
            from_file = false;
        }
        else {
            throw "invalid input parameter for `port_or_location`!";
        }

        auto s = from_file ? gst_template_file : gst_template_udp;
        VP_INFO(vp_utils::string_format("[%s] [%s]", node_name.c_str(), s.c_str()));
        this->initialized();
    }
    
    vp_image_src_node::~vp_image_src_node() {

    }

    void vp_image_src_node::handle_run() {
        cv::Mat frame;
        int video_width = 0;
        int video_height = 0;
        int fps = 0;
        std::chrono::milliseconds delta;

        while(true) {
            // check if need work
            gate.knock();

            auto last_time = std::chrono::system_clock::now();
            // try to open capture
            if (!image_capture.isOpened()) {
                auto gst_launch_str = from_file ? gst_template_file : gst_template_udp;
                if(!image_capture.open(gst_launch_str, cv::CAP_GSTREAMER)) {
                    VP_WARN(vp_utils::string_format("[%s] open image capture failed, try again...", node_name.c_str()));
                    continue;
                }
            }

            // video properties
            if (video_width == 0 || video_height == 0 || fps == 0) {
                video_width = image_capture.get(cv::CAP_PROP_FRAME_WIDTH);
                video_height = image_capture.get(cv::CAP_PROP_FRAME_HEIGHT);
                fps = image_capture.get(cv::CAP_PROP_FPS);
                fps = fps < 1 ? 1 : fps;  // since fps is too small (0.1 means 1 frame every 10 seconds)
                
                delta = std::chrono::milliseconds(1000 / fps);
    
                original_fps = fps;
                original_width = video_width;
                original_height = video_height;

                // stream_info_hooker activated if need
                if (stream_info_hooker) {
                    vp_stream_info stream_info {channel_index, fps, video_width, video_height, to_string()};
                    stream_info_hooker(node_name, stream_info);
                }
            }
            
            image_capture >> frame;
            if(frame.empty()) {
                VP_WARN(vp_utils::string_format("[%s] reading frame empty, total frame==>%d", node_name.c_str(), frame_index));
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
                this->out_queue.push(out_meta);

                // handled hooker activated if need
                if (this->meta_handled_hooker) {
                    meta_handled_hooker(node_name, out_queue.size(), out_meta);
                }

                // important! notify consumer of out_queue in case it is waiting.
                this->out_queue_semaphore.signal();
                VP_DEBUG(vp_utils::string_format("[%s] after handling meta, out_queue.size()==>%d", node_name.c_str(), out_queue.size()));
            }

            // for fps
            auto snap = std::chrono::system_clock::now() - last_time;
            snap = std::chrono::duration_cast<std::chrono::milliseconds>(snap);
            if (snap < delta) {
                std::this_thread::sleep_for(delta - snap);
            }
        }
    }

    std::string vp_image_src_node::to_string() {
        return from_file ? port_or_location : "udp://127.0.0.1:" + port_or_location + "/jpg";
    }
}