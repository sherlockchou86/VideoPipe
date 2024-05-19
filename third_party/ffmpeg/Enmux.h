//
// Created by jin_li on 2024/5/17.
//

#pragma once
#include "Encoder.h"
#include "SafeAVFormat.h"
#include <mutex>

namespace vp {
class Enmux {
public:
    Enmux(std::shared_ptr<Encoder> encoder, std::string out_url);
    ~Enmux();

    bool open();

    bool write_packet(av_packet &packet);

    void write_trailer();

    void close();

    /*!
     * @brief 创建一个封装器
     * @param encoder 编码器
     * @param out_url 输出文件路径
     * @return
     */
    static std::shared_ptr<Enmux> createShared(std::shared_ptr<Encoder> encoder,
                                               std::string              out_url) {
        return std::make_shared<Enmux>(encoder, out_url);
    }

    inline AVFormatContext *get_format_ctx() {
        return m_format_ctx;
    }

    inline AVStream *get_video_stream() const {
        return m_format_ctx->streams[m_video_stream_index];
    }

    inline AVRational get_time_base() const {
        return get_video_stream()->time_base;
    }

private:
    AVFormatContext         *m_format_ctx;
    std::shared_ptr<Encoder> m_encoder;
    std::string              m_out_url;
    std::string              m_format_name;
    int                      m_video_stream_index = 0;

    std::mutex m_mutex;
};

}  // namespace vp