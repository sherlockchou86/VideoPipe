#include "Enmux.h"
#include "FileOperate.h"
namespace vp {
Enmux::Enmux(std::shared_ptr<Encoder> encoder, std::string out_url) {
    m_encoder = encoder;
    m_out_url = out_url;
    if (out_url.find("mp4") != std::string::npos) {
        m_format_name = "mp4";
    } else if (out_url.find("flv") != std::string::npos) {
        m_format_name = "flv";
    } else if (out_url.find("rtmp") != std::string::npos) {
        m_format_name = "flv";
    } else if (out_url.find("jpg") != std::string::npos) {
        m_format_name = "image2";
    } else {
        m_format_name = "mp4";
    }
}

Enmux::~Enmux() {
    m_format_ctx = nullptr;
    close();
}

bool Enmux::open() {
    if (m_format_name == "mp4")
        utils::FileOperate::rm(m_out_url);
    ASSERT_FFMPEG(avformat_alloc_output_context2(&m_format_ctx, nullptr, m_format_name.c_str(),
                                                 m_out_url.c_str()));
    avformat_new_stream(m_format_ctx, m_encoder->get_codec_ctx()->codec);
    ASSERT_FFMPEG(avcodec_parameters_from_context(get_video_stream()->codecpar,
                                                  m_encoder->get_codec_ctx().get()));
    ASSERT_FFMPEG(avio_open(&m_format_ctx->pb, m_out_url.c_str(), AVIO_FLAG_WRITE));
    ASSERT_FFMPEG(avformat_write_header(m_format_ctx, nullptr));
    return true;
}

bool Enmux::write_packet(av_packet &packet) {
    std::unique_lock<std::mutex> lock(m_mutex);
    packet->dts = packet->pts;
    packet->duration = 1;

    packet->pts =
            av_rescale_q(packet->pts, m_encoder->get_time_base(), get_time_base());
    packet->dts =
            av_rescale_q(packet->dts, m_encoder->get_time_base(), get_time_base());
    packet->duration =
            av_rescale_q(packet->duration, m_encoder->get_time_base(), get_time_base()) / 3;
    packet->pos = -1;
    ASSERT_FFMPEG(av_interleaved_write_frame(m_format_ctx, packet.get()));
    return true;
}

void Enmux::close() {
    std::unique_lock<std::mutex> lock(m_mutex);
    if (m_format_ctx) {
        write_trailer();
        avio_close(m_format_ctx->pb);
        avformat_free_context(m_format_ctx);
        m_format_ctx = nullptr;
    }
}

void Enmux::write_trailer() {
    if (m_format_name == "mp4" || m_format_name == "jpg")
        av_write_trailer(m_format_ctx);
}

}  // namespace vp