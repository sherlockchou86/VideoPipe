#pragma once

#include <iostream>
#include <memory>
#include <chrono>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "vp_des_node.h"
#include "../objects/vp_frame_meta.h"
#include "../objects/vp_control_meta.h"
#include "../utils/vp_utils.h"


namespace vp_nodes {
    // callback before data disappear inside vp_app_des_node
    typedef std::function<void(std::string, std::shared_ptr<vp_objects::vp_meta>)> vp_app_des_result_hooker;

    // app des node, send meta data to external host code using callbacks.
    class vp_app_des_node: public vp_des_node {
    private:
        /* data */
        vp_app_des_result_hooker app_des_result_hooker;

        void invoke_app_des_result_hooker(std::shared_ptr<vp_objects::vp_meta> meta);
    protected:
        // re-implementation, return nullptr.
        virtual std::shared_ptr<vp_objects::vp_meta> handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) override; 
        // re-implementation, return nullptr.
        virtual std::shared_ptr<vp_objects::vp_meta> handle_control_meta(std::shared_ptr<vp_objects::vp_control_meta> meta) override;
    public:
        vp_app_des_node(std::string node_name, 
                        int channel_index);
        ~vp_app_des_node();

        void set_app_des_result_hooker(vp_app_des_result_hooker app_des_result_hooker);
    };
}