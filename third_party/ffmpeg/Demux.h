//
// Created by jin_li on 2024/5/17.
//

#pragma once
#include "SafeAVFormat.h"

namespace vp {
class Demux {
public:
    Demux() = default;
    ~Demux();

    bool open(const std::string &url);
    void close();

    int read_packet(av_packet &packet);

    void seek(int64_t timestamp);

    inline static std::shared_ptr<Demux> createShare() {
        return std::make_shared<Demux>();
    }

    inline AVFormatContext *get_format_ctx() {
        return m_format_ctx;
    }
    inline int get_video_stream_index() {
        return m_video_stream_index;
    }
    inline AVStream *get_video_stream() const {
        return m_format_ctx->streams[m_video_stream_index];
    }
    inline AVCodecID get_video_codec_id() const {
        return get_video_stream()->codecpar->codec_id;
    }
    inline av_codec_parameters get_video_codec_parameters() const {
        return std::make_shared<AVCodecParameters>(*get_video_stream()->codecpar);
    }

private:
    AVDictionary    *opt = nullptr;
    AVFormatContext *m_format_ctx;
    int              m_video_stream_index = -1;
    std::string      m_suffix;
};

}  // namespace vp