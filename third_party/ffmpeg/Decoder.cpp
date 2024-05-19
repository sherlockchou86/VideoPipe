#include "Decoder.h"
#include "NvidiaTools.h"

namespace vp {
Decoder::Decoder(std::shared_ptr<Demux> demux) {
    m_demux     = demux;
    m_codec_ctx = nullptr;
}

bool Decoder::open(bool use_hw) {
    AVCodecID      codec_id = m_demux->get_video_codec_id();
    const AVCodec *codec    = nullptr;
    if(codec_id == AV_CODEC_ID_H264){
        if(use_hw && checkIfSupportedNvidia()){
            codec = avcodec_find_decoder_by_name("h264_cuvid");
        } else{
            codec = avcodec_find_decoder(codec_id);
        }
    }
    if(codec_id == AV_CODEC_ID_HEVC){
        if(use_hw && checkIfSupportedNvidia()){
            codec = avcodec_find_decoder_by_name("hevc_cuvid");
        } else{
            codec = avcodec_find_decoder(codec_id);
        }
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

    // 拷贝视频信息进解码器上下文
    auto stream_info = m_demux->get_video_stream()->codecpar;
    ASSERT_FFMPEG(avcodec_parameters_to_context(m_codec_ctx.get(), stream_info));
    // 解码器参数设置
    m_codec_ctx->flags |= AV_CODEC_FLAG_LOW_DELAY;
    m_codec_ctx->flags2 |= AV_CODEC_FLAG2_FAST;
    m_codec_ctx->width  = stream_info->width;
    m_codec_ctx->height = stream_info->height;
    AVDictionary *dict = nullptr;
    av_dict_set(&dict, "threads", "auto", 0);
    av_dict_set(&dict, "zerolatency", "1", 0);
    av_dict_set(&dict, "strict", "-2", 0);
#ifdef AV_CODEC_CAP_TRUNCATED
    if(m_codec_ctx->codec->capabilities & AV_CODEC_CAP_TRUNCATED){
        m_codec_ctx->flags |= AV_CODEC_FLAG_TRUNCATED;
    }
#endif
    ASSERT_FFMPEG(avcodec_open2(m_codec_ctx.get(), codec, &dict));
    return true;
}

bool Decoder::send(av_packet packet) {
    if (!m_codec_ctx) {
        std::cout << "m_codec_ctx is nullptr" << std::endl;
        return false;
    }
    ASSERT_FFMPEG(avcodec_send_packet(m_codec_ctx.get(), packet.get()));
    return true;
}

bool Decoder::receive(av_frame &frame) {
    if (!m_codec_ctx) {
        std::cout << "m_codec_ctx is nullptr" << std::endl;
        return false;
    }
    ASSERT_FFMPEG(avcodec_receive_frame(m_codec_ctx.get(), frame.get()));
    return true;
}

void Decoder::close() {
    if (m_demux) {
        m_demux->close();
    }
}

Decoder::~Decoder() {
    close();
}

}  // namespace vp