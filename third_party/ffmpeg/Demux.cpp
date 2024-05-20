#include <chrono>
#include <thread>
#include "Demux.h"

namespace vp {

bool Demux::open(const std::string &url) {
    if (url.find("mp4") != std::string::npos) {
        m_suffix = "mp4";
    }
    // 创建上下文配置
    av_dictionary_set_info av_ctx_info;
    av_ctx_info.preset         = av_ctx_info.av_dict_set_preset.ultrafast;
    av_ctx_info.rtsp_transport = av_ctx_info.av_dict_set_rtsp_transport.tcp;
    av_ctx_info.stimeout       = "3000000";
    av_ctx_info.timeout        = "3000000";
    DESERIALIZE_INFO_TO_DICT(av_ctx_info, opt);
    // 打开上下文
    ASSERT_FFMPEG(avformat_open_input(&m_format_ctx, url.c_str(), nullptr, &opt));

    ASSERT_FFMPEG(avformat_find_stream_info(m_format_ctx, nullptr));
    // 查找视频流
    for (int i = 0; i < m_format_ctx->nb_streams; i++) {
        if (m_format_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            m_video_stream_index = i;
            break;
        }
    }
    if (m_video_stream_index == -1) {
        m_format_ctx = nullptr;
        return false;
    }
    av_dump_format(m_format_ctx, 0, url.c_str(), 0);
    return true;
}

int Demux::read_packet(av_packet &packet) {
    if (m_suffix == "mp4")
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    if (!m_format_ctx)
        return -1;
    return av_read_frame(m_format_ctx, packet.get());
}

void Demux::close() {
    if (opt) {
        av_dict_free(&opt);
    }
    if (m_format_ctx) {
        avformat_close_input(&m_format_ctx);
        avformat_free_context(m_format_ctx);
    }
}

Demux::~Demux() {
    close();
}

void Demux::seek(int64_t timestamp) {
    if (!m_format_ctx)
        return;
    av_seek_frame(m_format_ctx, m_video_stream_index, timestamp, AVSEEK_FLAG_BACKWARD);
}
}  // namespace vp