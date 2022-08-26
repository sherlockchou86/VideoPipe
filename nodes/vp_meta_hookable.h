#pragma once
#include <functional>
#include <string>
#include <memory>

#include "../objects/vp_meta.h"

namespace vp_nodes {
    // callback when meta flowing through the whole pipe, MUST NOT be blocked.
    // we can do more work based on this callback, such as calculating fps/latency at each port of node, please refer to vp_analysis_board for details.
    typedef std::function<void(std::string, int, std::shared_ptr<vp_objects::vp_meta>)> vp_meta_hooker;

    // allow hookers attached to the pipe (nodes), get notified when meta flow through each port of node (total 4 ports in node).
    // this class is inherited by vp_node only.
    class vp_meta_hookable {
    protected:
        // hooker activated when meta is arriving at node (pushed to in_queue of vp_node, the 1st port in node).
        vp_meta_hooker meta_arriving_hooker;
        // hooker activated when meta is to be handled inside node (poped from in_queue of vp_node, the 2nd port in node).
        vp_meta_hooker meta_handling_hooker;
        // hooker activated when meta is handled inside node (pushed to out_queue of vp_node, the 3rd port in node). 
        vp_meta_hooker meta_handled_hooker;
        // hooker activated when meta is leaving from node (poped from out_queue of vp_node, the 4th port in node).
        vp_meta_hooker meta_leaving_hooker;
    public:
        vp_meta_hookable(/* args */) {}
        ~vp_meta_hookable() {}

        void set_meta_arriving_hooker(vp_meta_hooker meta_arriving_hooker) {
            this->meta_arriving_hooker = meta_arriving_hooker;
        }

        void set_meta_handling_hooker(vp_meta_hooker meta_handling_hooker) {
            this->meta_handling_hooker = meta_handling_hooker;
        }

        void set_meta_handled_hooker(vp_meta_hooker meta_handled_hooker) {
            this->meta_handled_hooker = meta_handled_hooker;
        }

        void set_meta_leaving_hooker(vp_meta_hooker meta_leaving_hooker) {
            this->meta_leaving_hooker = meta_leaving_hooker;
        }
    };
}