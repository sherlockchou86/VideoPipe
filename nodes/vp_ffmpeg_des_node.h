//
// Created by lijin on 2023/8/2.
//
#ifdef VP_WITH_FFMPEG

#ifndef VIDEO_PIPE_VP_FFMPEG_DES_NODE_H
#define VIDEO_PIPE_VP_FFMPEG_DES_NODE_H


#include "vp_des_node.h"

#include "../third_party/ffmpeg/Scaler.h"
#include "../third_party/ffmpeg/Encoder.h"
#include "../third_party/ffmpeg/Enmux.h"

namespace vp_nodes {

    /*
     *  ffmpeg流输出节点，支持rtmp推流，支持硬件编码
     * */
    class vp_ffmpeg_des_node : public vp_des_node {
    private:
        std::shared_ptr<vp::Scaler> m_scaler;
        std::shared_ptr<vp::Encoder> m_encoder;
        std::shared_ptr<vp::Enmux> m_enmux;

        int m_from_width = 0;
        int m_from_height = 0;
        int m_from_format = 0;
        int m_to_width = 0;
        int m_to_height = 0;
        int m_to_format = 0;
        int m_fps = 25;
        int m_bitrate = 1024 * 1024 * 2;


    protected:
        virtual std::shared_ptr<vp_objects::vp_meta>
        handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) override;

    public:
        vp_ffmpeg_des_node(std::string node_name,
                           int channel_index,
                           std::string url,
                           int from_width,
                           int from_height,
                           int from_format,
                           int to_width,
                           int to_height,
                           int to_format,
                           int bitrate = 1024 * 1024 * 2,
                           int fps = 25,
                           bool use_hw = false);


        void set_frommat(int width, int height, int format);

        void set_tomat(int width, int height, int format);

        void set_fps(int fps);

        void set_bitrate(int bitrate);

        bool open(const std::string &url, bool useHwEncode = false);

    };

} // vp_nodes

#endif //VIDEO_PIPE_VP_FFMPEG_DES_NODE_H
#endif //VP_WITH_FFMPEG
