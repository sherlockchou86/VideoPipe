//
// Created by lijin on 2023/8/2.
//
#ifdef VP_WITH_FFMPEG
#ifndef VIDEO_PIPE_VP_FFMPEG_SRC_NODE_H
#define VIDEO_PIPE_VP_FFMPEG_SRC_NODE_H

#include "vp_src_node.h"
#include "../third_party/ffmpeg/Scaler.h"
#include "../third_party/ffmpeg/Demux.h"
#include "../third_party/ffmpeg/Decoder.h"

namespace vp_nodes {

    /*
     *  ffmpeg输入节点，支持rtsp/rtmp/hls/mp4等，支持硬件解码
     */
    class vp_ffmpeg_src_node : public vp_src_node {
    private:
        std::shared_ptr<vp::Scaler> m_scaler;
        std::shared_ptr<vp::Demux> m_demux;
        std::shared_ptr<vp::Decoder> m_decoder;
    protected:
        void handle_run() override;

        bool open(const std::string &url, bool use_hw = false);

    public:
        vp_ffmpeg_src_node(std::string node_name,
                           int channel_index,
                           std::string file_ro_url_path,
                           bool use_hw = false);

    };

} // vp_nodes

#endif //VIDEO_PIPE_VP_FFMPEG_SRC_NODE_H
#endif //VP_WITH_FFMPEG
