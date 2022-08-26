#pragma once

#include <string>

#include "vp_node.h"
#include "vp_stream_status_hookable.h"

namespace vp_nodes {
    // base class for des nodes, end point of meta/pipeline.
    class vp_des_node: public vp_node, public vp_stream_status_hookable {
    private:
        // cache for stream status at current des node.
        vp_stream_status stream_status;

        // period(ms) to calculate output fps
        int fps_epoch = 500;
        int fps_counter = 0;
        std::chrono::system_clock::time_point fps_last_time;
    protected:
        // do nothing in des nodes
        virtual void dispatch_run() override final;
        // sample implementation, return nullptr in all des nodes.
        virtual std::shared_ptr<vp_objects::vp_meta> handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) override; 
        // sample implementation, return nullptr in all des nodes.
        virtual std::shared_ptr<vp_objects::vp_meta> handle_control_meta(std::shared_ptr<vp_objects::vp_control_meta> meta) override;

        // protected as it can't be instanstiated directly.        
        vp_des_node(std::string node_name, int channel_index);
    public:
        ~vp_des_node();

        virtual vp_node_type node_type() override;

        int channel_index;
    };
}