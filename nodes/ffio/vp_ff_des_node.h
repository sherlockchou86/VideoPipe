#pragma once
#ifdef VP_WITH_FFMPEG
#include "ff_des.h"
#include "../vp_des_node.h"

namespace vp_nodes {
    /**
     * universal DES node using FFmpeg.
     * 
     * support output uri:
     * 1. path of file streams like `./vp_data/out_vp_test.mp4`.
     * 2. url of network streams like `rtmp://192.168.77.68/live/stream`.
     */
    class vp_ff_des_node final: public vp_des_node {
    private:
        /* inner members. */
        std::string m_out_uri = "";
        bool m_use_osd = true;
        int m_out_bitrate = 1024;
        std::string m_encoder_name = "";
        vp_objects::vp_size m_resolution_w_h;

        /**
         * encode & enmux.
         */
        ff_des_ptr m_ff_des = nullptr;
        /**
         * SwsContext used fot scale by FFmpeg.
         */
        SwsContext* sws_ctx = NULL;
    protected:
        // re-implementation, return nullptr.
        virtual std::shared_ptr<vp_objects::vp_meta> handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) override; 
    public:
        /**
         * create vp_ff_des_node instance using initial parameters.
         * 
         * @param node_name specify the name of DES node.
         * @param channel_index specify the channel index of DES node.
         * @param out_uri specify the uri to be written.
         * @param use_osd specify use osd frame as output or not.
         * @param out_width specify the final width of output, 0 means use the width of frame flowing in pipeline.
         * @param out_height specify the final height of output, 0 means use the height of frame flowing in pipeline.
         * @param out_fps specify the fps of output, 0 means use the fps of original stream in pipeline.
         * @param out_bitrate specify the bitrate of output (kbit/s).
         * @param out_max_b_frames specify the max B frames in a GOP for encoding.
         * @param encoder_name specify the encoder name (`libx264`/`libx265`/`h264_nvenc`/`hevc_nvenc`) used for encoding in FFmpeg.
         * @param out_sw_pix_fmt specify the pixel format of output.
         * 
         * @note
         * the encoder specified by `encoder_name` MUST be supported already in FFmpeg, 
         * we can run `ffmpeg -encoders` to show list of encoders supported in FFmpeg.
         * if the encoder not found, please reconfigure & rebuild your FFmpeg.
         */
        vp_ff_des_node(const std::string& node_name,
                              int channel_index,
                              const std::string& out_uri,
                              vp_objects::vp_size resolution_w_h = {}, 
                              int bitrate = 1024,
                              bool osd = true,
                              std::string encoder_name = "libx264");
        ~vp_ff_des_node();

        /**
         * return out uri of DES node.
         */
        virtual std::string to_string() override;
    };
}
#endif