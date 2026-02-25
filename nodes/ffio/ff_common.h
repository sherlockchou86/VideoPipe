#pragma once
#ifdef VP_WITH_FFMPEG
#include <memory>
#include <thread>
#include <condition_variable>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <queue>
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/avutil.h>
#include <libavutil/pixdesc.h>
#include <libavutil/imgutils.h>
#include <libavdevice/avdevice.h>
}

#define ff_av_packet_ptr std::shared_ptr<AVPacket>
#define ff_av_frame_ptr std::shared_ptr<AVFrame>
#define ff_src_ptr std::shared_ptr<vp_nodes::ff_src>
#define ff_des_ptr std::shared_ptr<vp_nodes::ff_des>
#define ff_scaler_ptr std::shared_ptr<vp_nodes::ff_scaler>

#define alloc_ff_src(channel_index) std::make_shared<vp_nodes::ff_src>(channel_index)
#define alloc_ff_des(channel_index) std::make_shared<vp_nodes::ff_des>(channel_index)
#define alloc_ff_scaler(src_width, src_height, src_fmt, dst_width, dst_height, dst_fmt)               \
        std::make_shared<vp_nodes::ff_scaler>(src_width, src_height, src_fmt, dst_width, dst_height, dst_fmt)
        
#define alloc_ff_av_frame()                                                                           \
        std::shared_ptr<AVFrame>(av_frame_alloc(), [](AVFrame *frame) { av_frame_free(&frame); })
#define alloc_ff_av_packet()                                                                          \
        std::shared_ptr<AVPacket>(av_packet_alloc(), [](AVPacket *pkt) { av_packet_free(&pkt); })
    
namespace vp_nodes {
    /**
     * tools for color conversion & image resize using FFmpeg on CPUs.
     * 
     */
    class ff_scaler {
    private:
        SwsContext* sws_ctx = NULL;
        int m_src_width;
        int m_src_height;
        AVPixelFormat m_src_fmt;
        int m_dst_width;
        int m_dst_height;
        AVPixelFormat m_dst_fmt;
    public:
        /* disable copy and assignment operations */
        ff_scaler(const ff_scaler&) = delete;
        ff_scaler& operator=(const ff_scaler&) = delete;
        ff_scaler(int src_width, 
                 int src_height, 
                 AVPixelFormat src_fmt, 
                 int dst_width, 
                 int dst_height,
                 AVPixelFormat dst_fmt);
        ~ff_scaler();
        bool scale(ff_av_frame_ptr src, ff_av_frame_ptr& dst);
    };
}
#endif