
#include "vp_des_node.h"


namespace vp_nodes {

        
    vp_des_node::vp_des_node(std::string node_name, int channel_index): 
                            vp_node(node_name), channel_index(channel_index) {
        stream_status.channel_index = channel_index;
    }
    
    vp_des_node::~vp_des_node() {

    }

    // do nothing in des nodes
    void vp_des_node::dispatch_run() {
        // dispatch thread terminates immediately in all des nodes
    }

    // it is the end point of stream(frame meta) in pipe, this method MUST be called at the end of handle_frame_meta in derived class.
    // using 'return vp_des_node::handle_frame_meta(...);'
    std::shared_ptr<vp_objects::vp_meta> vp_des_node::handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) {
        // update cache of stream status
        stream_status.frame_index = meta->frame_index;
        stream_status.width = meta->frame.size().width;
        stream_status.height = meta->frame.size().height;
        stream_status.direction = to_string();

        // calculate the duration between now and the time when meta created, which is latency.
        auto delta_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - meta->create_time);
        stream_status.latency = delta_time.count();

        // update fps if need
        fps_counter++;
        delta_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - fps_last_time);
        if (delta_time.count() >= fps_epoch) {        
            stream_status.fps = fps_counter * 1000.0 / delta_time.count();
            fps_counter = 0;
            fps_last_time = std::chrono::system_clock::now();
        }

        // activate the stream status hooker if need
        if (stream_status_hooker) {
            stream_status_hooker(node_name, stream_status);
        }
        
        return nullptr;
    }
    
    // it is the end point of control meta in pipe, this method MUST be called at the end of handle_control_meta in derived class.
    // using 'return vp_des_node::handle_control_meta(...);'
    std::shared_ptr<vp_objects::vp_meta> vp_des_node::handle_control_meta(std::shared_ptr<vp_objects::vp_control_meta> meta) {
        // ...
        return nullptr;
    }

    vp_node_type vp_des_node::node_type() {
        return vp_node_type::DES;
    }
}