#pragma once

#include "vp_node.h"
#include "vp_stream_info_hookable.h"
#include "../excepts/vp_not_implemented_error.h"
#include "../excepts/vp_invalid_calling_error.h"
#include "../utils/vp_gate.h"

namespace vp_nodes {
    // base class for src nodes, start point of meta/pipeline
    class vp_src_node: public vp_node, public vp_stream_info_hookable {
    private:
        /* data */
    
    protected:
        // force re-implemetation in child class
        virtual void handle_run() override;
        // force ignored in child class
        virtual std::shared_ptr<vp_objects::vp_meta> handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) override final; 
        // force ignored in child class
        virtual std::shared_ptr<vp_objects::vp_meta> handle_control_meta(std::shared_ptr<vp_objects::vp_control_meta> meta) override final;

        // protected as it can't be instanstiated directly.
        vp_src_node(std::string node_name, 
                    int channel_index, 
                    float resize_ratio = 1.0);

        // basic stream info
        int original_fps = -1;
        int original_width = 0;
        int original_height = 0;

        // basic channnel info
        int frame_index;
        int channel_index;
        float resize_ratio;

        // control to work or not
        // all derived class need depend on the value to check if work or not (start/stop)
        vp_utils::vp_gate gate;
    public:
        ~vp_src_node();

        virtual vp_node_type node_type() override;

        // start signal to pipeline
        void start();
        // stop signal to pipeline
        void stop();
        // speak signal to the pipeline (each node print some message such as current status)
        void speak();

        int get_original_fps() const;
        int get_original_width() const;
        int get_original_height() const;
    };
    
}