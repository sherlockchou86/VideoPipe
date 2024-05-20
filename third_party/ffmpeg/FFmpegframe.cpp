#include "FFmpegframe.h"
#include <assert.h>

namespace FFmpeg{



    FFmpegFrame::FFmpegFrame(std::shared_ptr<AVFrame> frame)
    {
        if (frame) {
            _frame = std::move(frame);
        } else {
            _frame.reset(av_frame_alloc(), [](AVFrame *ptr) {
                av_frame_free(&ptr);
            });
        }
    }

    FFmpegFrame::~FFmpegFrame()
    {
        if (_data) {
            delete[] _data;
        }
    }


    AVFrame *FFmpegFrame::get() const {
        return _frame.get();
    }


    
    void FFmpegFrame::fillPicture(AVPixelFormat target_format, int target_width, int target_height) {
        assert(_data == nullptr);
        _data = new char[av_image_get_buffer_size(target_format, target_width, target_height, 1)];
        av_image_fill_arrays(_frame->data, _frame->linesize, (uint8_t *) _data,  target_format, target_width, target_height,1);
    }




    std::shared_ptr<AVPacket> alloc_av_packet() {
        auto pkt = std::shared_ptr<AVPacket>(av_packet_alloc(), [](AVPacket *pkt) {
            av_packet_free(&pkt);
        });
        pkt->data = NULL;    // packet data will be allocated by the encoder
        pkt->size = 0;
        return pkt;
    }

    std::shared_ptr<AVFrame> alloc_av_frame() {
        auto frame = std::shared_ptr<AVFrame>(av_frame_alloc(), [](AVFrame *frame) {
            av_frame_free(&frame);
        });
        return frame;
    }

    std::shared_ptr<AVCodecContext> alloc_av_codec_context() {
        auto codec_context = std::shared_ptr<AVCodecContext>(avcodec_alloc_context3(NULL), [](AVCodecContext *codec_context) {
            avcodec_free_context(&codec_context);
        });
        return codec_context;
    }













}