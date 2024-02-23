

#include "vp_rtsp_des_node.h"


namespace vp_nodes {
        
    vp_rtsp_des_node::vp_rtsp_des_node(std::string node_name, 
                        int channel_index, 
                        int rtsp_port, 
                        std::string rtsp_name, 
                        vp_objects::vp_size resolution_w_h, 
                        int bitrate,
                        bool osd):
                        vp_des_node(node_name, channel_index),
                        rtsp_port(rtsp_port),
                        rtsp_name(rtsp_name),
                        resolution_w_h(resolution_w_h),
                        bitrate(bitrate),
                        osd(osd) {
        if (rtsp_name.empty()) {
            // use channel index as rtsp name
            this->rtsp_name = std::to_string(channel_index);
        }
        gst_template = vp_utils::string_format(gst_template, bitrate, base_udp_port + channel_index);
        VP_INFO(vp_utils::string_format("[%s] [%s]", node_name.c_str(), gst_template.c_str()));

        // start rtsp server asynchronously
        start_rtsp_streaming();

        this->initialized();
    }
    
    vp_rtsp_des_node::~vp_rtsp_des_node() { 
        deinitialized();
    }
    GstRTSPServer* vp_rtsp_des_node::rtsp_server = NULL;
    std::shared_ptr<vp_objects::vp_meta> vp_rtsp_des_node::handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) {
        VP_DEBUG(vp_utils::string_format("[%s] received frame meta, channel_index=>%d, frame_index=>%d", node_name.c_str(), meta->channel_index, meta->frame_index));
        
        cv::Mat resize_frame;
        if (this->resolution_w_h.width != 0 && this->resolution_w_h.height != 0) {                 
            cv::resize((osd && !meta->osd_frame.empty()) ? meta->osd_frame : meta->frame, resize_frame, cv::Size(resolution_w_h.width, resolution_w_h.height));
        }
        else {
            resize_frame = (osd && !meta->osd_frame.empty()) ? meta->osd_frame : meta->frame;
        }

        if (!rtsp_writer.isOpened()) {
            assert(rtsp_writer.open(this->gst_template, cv::CAP_GSTREAMER, 0, meta->fps, {resize_frame.cols, resize_frame.rows}));
        }

        rtsp_writer.write(resize_frame);

        // for general works defined in base class
        return vp_des_node::handle_frame_meta(meta);
    }

    void vp_rtsp_des_node::start_rtsp_streaming () {
        // create a rtsp server using gst-rtsp-server
        if (rtsp_server == NULL) {
            char port_num_Str[64] = { 0 };
            sprintf (port_num_Str, "%d", rtsp_port);
            rtsp_server = gst_rtsp_server_new ();
            g_object_set (rtsp_server, "service", port_num_Str, NULL);
            gst_rtsp_server_attach (rtsp_server, NULL);
        }

        char udpsrc_pipeline[512];
        if (udp_buffer_size == 0)
            udp_buffer_size = 512 * 1024;
        
        // receive stream data from udpsrc internally and push it via rtsp
        sprintf (udpsrc_pipeline,
                "( udpsrc name=pay0 port=%d buffer-size=%lu caps=\"application/x-rtp, media=video, "
                "clock-rate=90000, encoding-name=H264, payload=96 \" )",
                base_udp_port + channel_index, udp_buffer_size);
            
        auto mounts = gst_rtsp_server_get_mount_points (rtsp_server);

        auto factory = gst_rtsp_media_factory_new ();
        gst_rtsp_media_factory_set_launch (factory, udpsrc_pipeline);
        gst_rtsp_media_factory_set_shared (factory, TRUE);
        gst_rtsp_mount_points_add_factory (mounts, ("/" + rtsp_name).c_str(), factory);
        g_object_unref (mounts);

        VP_INFO(vp_utils::string_format("[%s] is going to push rtsp stream, please visit:[%s]", node_name.c_str(), to_string().c_str()));
    }

    std::string vp_rtsp_des_node::to_string() {
        return "rtsp://localhost:" + std::to_string(rtsp_port) + "/" + rtsp_name;
    }
}