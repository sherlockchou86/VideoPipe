#include "ImageSaver.h"
//#include <logger/Logger.h>

namespace vp {

ImageSaver::ImageSaver(int width, int height, int format, int to_width, int to_height)
    : m_from_width(width), m_from_height(height), m_format(format), m_to_width(to_width),
      m_to_height(to_height) {
    const AVCodec *codec   = avcodec_find_encoder(AV_CODEC_ID_MJPEG);
    m_codec_ctx            = alloc_av_codec_context(codec);
    m_codec_ctx->width     = m_to_width;
    m_codec_ctx->height    = m_to_height;
    m_codec_ctx->pix_fmt   = AV_PIX_FMT_YUVJ420P;  //(AVPixelFormat)m_format;
    m_codec_ctx->time_base = {1, 25};              // 帧率
    avcodec_open2(m_codec_ctx.get(), codec, nullptr);
    m_scaler = Scaler::createShare(m_from_width, m_from_height, (AVPixelFormat)m_format, m_to_width, m_to_height,
                                   AV_PIX_FMT_YUVJ420P);
}

av_packet ImageSaver::frame_to_jpeg(av_frame frame) {
    av_frame yuvj420_frame = alloc_av_frame();
    yuvj420_frame->format  = AV_PIX_FMT_YUVJ420P;
    yuvj420_frame->width   = m_to_width;
    yuvj420_frame->height  = m_to_height;
    av_frame_get_buffer(yuvj420_frame.get(), 32);
    m_scaler->scale<av_frame, av_frame>(frame, yuvj420_frame);
    av_packet packet = alloc_av_packet();
    int       re     = avcodec_send_frame(m_codec_ctx.get(), yuvj420_frame.get());
    if (re < 0) {
        std::cerr << "Error sending frame to encoder" << std::endl;
        return nullptr;
    }
    while (re >= 0) {
        auto ret = avcodec_receive_packet(m_codec_ctx.get(), packet.get());
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
            break;
        else if (ret < 0) {
            std::cerr << "Error encoding frame" << std::endl;
            return nullptr;
        } else if (ret == 0) {
            return packet;
        }
    }
    return nullptr;
}

}  // namespace vp