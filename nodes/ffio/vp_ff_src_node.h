
#pragma once
#ifdef VP_WITH_FFMPEG
#include "ff_src.h"
#include "../vp_src_node.h"

namespace vp_nodes {
    /**
     * universal SRC node using FFmpeg.
     * 
     * support uri:
     * 1. path of file streams like `./vp_data/vp_test.mp4`.
     * 2. url of network streams like `rtsp://192.168.77.68/main_stream`.
     */
    class vp_ff_src_node final: public vp_src_node {
    private:
        /* inner members. */
        std::string m_decoder_name = "";
        std::string m_uri = "";
        // 0 means no skip
        int m_skip_interval = 0;

        /**
         * demux & decode.
         */
        ff_src_ptr m_ff_src = nullptr;
    protected:
        /**
         * get frames using FFmpeg.
         */
        virtual void handle_run() override;
    public:
        /**
         * create vp_ff_src_node instance using initial parameters.
         * 
         * @param node_name specify the name of SRC node.
         * @param channel_index specify the channel index of SRC node.
         * @param uri specify the uri to be opened by SRC node.
         * @param decoder_name specify the decoder name (`h264`/`hevc`/`h264_cuvid`/`hevc_cuvid`) used for decoding in FFmpeg.
         * @param resize_ratio specify the resize ratio applied to frames.
         * 
         * @note
         * the decoder specified by `decoder_name` MUST be supported already in FFmpeg, 
         * we can run `ffmpeg -decoders` to show list of decoders supported in FFmpeg.
         * if the decoder not found, please reconfigure & rebuild your FFmpeg.
         */
        vp_ff_src_node(const std::string& node_name, 
                              int channel_index,
                              const std::string& uri,
                              const std::string& decoder_name = "h264",
                              float resize_ratio = 1.0,
                              int skip_interval = 0);
        ~vp_ff_src_node();

        /**
         * return uri of SRC node.
         */
        virtual std::string to_string() override;
    };
}
#endif