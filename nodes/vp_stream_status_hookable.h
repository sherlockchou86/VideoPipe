#pragma once

#include <functional>
#include <string>
#include <memory>

namespace vp_nodes {
    // stream status created by des nodes.
    struct vp_stream_status {
        int channel_index = -1;  // channel index
        int frame_index = -1;    // latest frame index
        int latency = 0;         // latency(ms) relative to src node

        int fps = 0;             // output fps of stream, maybe it is not equal to original_fps
        int width = 0;           // output width of stream, maybe it is not equal to original_width
        int height = 0;          // output height of stream, maybe it is not equal to original_height
        std::string direction;   // where the stream goes to
    };

    // callback when stream is going out of pipe, happens in des nodes. MUST not be blocked.
    typedef std::function<void(std::string, vp_stream_status)> vp_stream_status_hooker;

    // allow hookers attached to the pipe (des nodes specifically), hookers get notified when stream is going out of pipe.
    // this class is inherited by vp_des_node only.
    class vp_stream_status_hookable
    {
    private:
        /* data */
    protected:
        vp_stream_status_hooker stream_status_hooker;
    public:
        vp_stream_status_hookable(/* args */) {}
        ~vp_stream_status_hookable() {}
        
        void set_stream_status_hooker(vp_stream_status_hooker stream_status_hooker) {
            this->stream_status_hooker = stream_status_hooker;
        }
    };
}