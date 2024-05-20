#include "Encoder.h"
#include "NvidiaTools.h"

namespace vp {

    Encoder::Encoder(int width, int height, int fps, int bitrate) {
        m_width = width;
        m_height = height;
        m_fps = fps;
        m_bitrate = bitrate;
    }

    Encoder::Encoder(AVCodecID codec_id, int width, int height, int fps, int bitrate) {
        m_width = width;
        m_height = height;
        m_fps = fps;
        m_bitrate = bitrate;
        m_codec_id = codec_id;
    }

    Encoder::~Encoder() {
        close();
    }

    bool Encoder::open(bool use_hw) {
        const AVCodec *codec = nullptr;
        switch (m_codec_id) {
            case AV_CODEC_ID_H264: {
                if (use_hw && checkIfSupportedNvidia()) {
                    codec = avcodec_find_encoder_by_name("h264_nvenc");
                } else {
                    codec = avcodec_find_encoder(AV_CODEC_ID_H264);
                }
                break;
            }
            case AV_CODEC_ID_HEVC: {
                if (use_hw && checkIfSupportedNvidia()) {
                    codec = avcodec_find_encoder_by_name("hevc_nvenc");
                } else {
                    codec = avcodec_find_encoder(AV_CODEC_ID_HEVC);
                }
                break;
            }
            default:
                break;
        }
        if (!codec) {
            std::cout << "avcodec_find_decoder failed" << std::endl;
            return false;
        }

        m_codec_ctx = alloc_av_codec_context(codec);
        if (!m_codec_ctx) {
            std::cout << "alloc_av_codec_context failed" << std::endl;
            return false;
        }
        m_codec_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
        m_codec_ctx->codec_id = codec->id;
        m_codec_ctx->width = m_width;
        m_codec_ctx->height = m_height;
        m_codec_ctx->bit_rate = m_bitrate;
        m_codec_ctx->time_base = {1, m_fps};
        m_codec_ctx->framerate = {m_fps, 1};
        m_codec_ctx->gop_size = m_gop_size;
        m_codec_ctx->max_b_frames = 0;
        m_codec_ctx->pix_fmt = AV_PIX_FMT_YUV420P;

        AVDictionary *param = nullptr;
        av_dict_set(&param, "preset", "veryfast", 0);
        av_dict_set(&param, "tune", "zerolatency", 0);
        av_dict_set(&param, "threads", "auto", 0);

        ASSERT_FFMPEG(avcodec_open2(m_codec_ctx.get(), codec, &param));
        av_dict_free(&param);
        return true;
    }

    bool Encoder::send(av_frame frame) {
        if (!m_codec_ctx) {
            std::cout << "m_codec_ctx is nullptr" << std::endl;
            return false;
        }
        ASSERT_FFMPEG(avcodec_send_frame(m_codec_ctx.get(), frame.get()));
        return true;
    }

    bool Encoder::receive(av_packet &packet) {
        if (!m_codec_ctx) {
            std::cout << "m_codec_ctx is nullptr" << std::endl;
            return false;
        }
        ASSERT_FFMPEG(avcodec_receive_packet(m_codec_ctx.get(), packet.get()));
        return true;
    }

    bool Encoder::encode(av_frame frame, av_packet &packet) {
        if (!send(frame)) {
            return false;
        }
        if (!receive(packet)) {
            return false;
        }
        return true;
    }

    void Encoder::close() {
        if (m_codec_ctx) {
            avcodec_close(m_codec_ctx.get());
        }
    }

    void Encoder::set_gop_size(int gop_size) {
        m_gop_size = gop_size;
    }


}  // namespace vp