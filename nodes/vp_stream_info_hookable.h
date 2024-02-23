#pragma once

#include <functional>
#include <mutex>
#include <string>
#include <memory>

namespace vp_nodes {
    // stream info created by src nodes.
    struct vp_stream_info {
        int channel_index = -1;   // channel index
        int original_fps = 0;     // original fps of stream
        int original_width = 0;   // width of resolution
        int original_height = 0;  // height of resolution
        std::string uri = "";     // uri of stream
    };
    
    // callback when stream info changed, happens in src nodes. MUST not be blocked.
    typedef std::function<void(std::string, vp_stream_info)> vp_stream_info_hooker;

    // allow hookers attached to the pipe (src nodes specifically), hookers get notified when stream info changed.
    // this class is inherited by vp_src_node only.
    class vp_stream_info_hookable
    {
    private:
        /* data */
    protected:
        std::mutex stream_info_hooker_lock;
        vp_stream_info_hooker stream_info_hooker;
    public:
        vp_stream_info_hookable(/* args */) {}
        ~vp_stream_info_hookable() {}

        void set_stream_info_hooker(vp_stream_info_hooker stream_info_hooker) {
            std::lock_guard<std::mutex> guard(stream_info_hooker_lock);
            this->stream_info_hooker = stream_info_hooker;
        }

        void invoke_stream_info_hooker(std::string node_name, vp_stream_info stream_info) {
            std::lock_guard<std::mutex> guard(stream_info_hooker_lock);
            if (this->stream_info_hooker) {
                this->stream_info_hooker(node_name, stream_info);
            }
        }
    };
}