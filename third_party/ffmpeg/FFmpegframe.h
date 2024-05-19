//
// Created by jin_li on 2024/5/17.
//

#pragma once

extern "C" {
#include "libavcodec/avcodec.h"
#include "libavutil/audio_fifo.h"
#include "libavutil/avutil.h"
#include "libavutil/imgutils.h"
#include "libavutil/pixdesc.h"
#include "libswresample/swresample.h"
#include "libswscale/swscale.h"
}

#include <memory>

namespace FFmpeg {

class FFmpegFrame {
public:
    typedef std::shared_ptr<FFmpegFrame> ptr;

    FFmpegFrame(std::shared_ptr<AVFrame> frame = nullptr);
    ~FFmpegFrame();

    AVFrame *get() const;
    void     fillPicture(AVPixelFormat target_format, int target_width, int target_height);

private:
    char                    *_data = nullptr;
    std::shared_ptr<AVFrame> _frame;
};

std::shared_ptr<AVPacket> alloc_av_packet();

std::shared_ptr<AVFrame> alloc_av_frame();

std::shared_ptr<AVCodecContext> alloc_av_codec_context();

}  // namespace FFmpeg
