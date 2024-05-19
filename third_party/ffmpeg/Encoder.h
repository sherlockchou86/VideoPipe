//
// Created by jin_li on 2024/5/17.
//

#pragma once

#include "SafeAVFormat.h"

namespace vp {

    class Encoder {
    public:
        Encoder(int width, int height, int fps, int bitrate);

        Encoder(AVCodecID codec_id, int width, int height, int fps, int bitrate);

        ~Encoder();

        bool open(bool use_hw = false);

        bool send(av_frame frame);

        bool receive(av_packet &packet);

        bool encode(av_frame frame, av_packet &packet);

        void close();

        void set_gop_size(int gop_size);

        inline av_codec_context get_codec_ctx() {
            return m_codec_ctx;
        }

        inline AVRational get_time_base() const {
            return {1, m_fps};
        }

        /*!
         * @brief 创建一个编码器
         * @param width 输出视频的宽度
         * @param height 输出视频的高度
         * @param fps 输出视频的帧率
         * @param bitrate 输出视频的码率
         * @return
         */
        static std::shared_ptr<Encoder> createShared(int width, int height, int fps, int bitrate) {
            return std::make_shared<Encoder>(width, height, fps, bitrate);
        }

        /*!
         * @brief 创建一个编码器
         * @param width 输出视频的宽度
         * @param height 输出视频的高度
         * @param fps 输出视频的帧率
         * @param bitrate 输出视频的码率
         * @return
         */
        static std::shared_ptr<Encoder> createShared(AVCodecID codec_id, int width, int height, int fps, int bitrate) {
            return std::make_shared<Encoder>(codec_id, width, height, fps, bitrate);
        }


    private:
        av_codec_context m_codec_ctx;
        int m_width;
        int m_height;
        int m_fps;
        int m_bitrate;
        int m_gop_size = 50;
        AVCodecID m_codec_id = AV_CODEC_ID_H264;
    };

}  // namespace vp