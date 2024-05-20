//
// Created by jin_li on 2024/5/17.
//

#pragma once
#include "Demux.h"
#include "SafeAVFormat.h"


namespace vp {
class Decoder {
public:
    Decoder(std::shared_ptr<Demux> demux);

    ~Decoder();

    bool open(bool use_hw = false);

    bool send(av_packet packet);

    bool receive(av_frame &frame);

    void close();

    static std::shared_ptr<Decoder> createShare(std::shared_ptr<Demux> demux) {
        return std::make_shared<Decoder>(demux);
    }

private:
    av_codec_context       m_codec_ctx;
    std::shared_ptr<Demux> m_demux;
};
}  // namespace vp
