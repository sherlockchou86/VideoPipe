
#ifdef VP_WITH_FFMPEG
#include "ff_common.h"

namespace vp_nodes {
    ff_scaler::ff_scaler(int src_width, 
            int src_height, 
            AVPixelFormat src_fmt, 
            int dst_width, 
            int dst_height,
            AVPixelFormat dst_fmt):
            m_src_width(src_width),
            m_src_height(src_height),
            m_src_fmt(src_fmt),
            m_dst_width(dst_width),
            m_dst_height(dst_height),
            m_dst_fmt(dst_fmt) {
        sws_ctx = sws_getContext(src_width, src_height, src_fmt, dst_width,dst_height, dst_fmt, SWS_FAST_BILINEAR, NULL, NULL, NULL);
    }

    ff_scaler::~ff_scaler() {
        if (!sws_ctx) {
            sws_freeContext(sws_ctx);
        }
    }

    bool ff_scaler::scale(ff_av_frame_ptr src, ff_av_frame_ptr& dst) {
        if (!sws_ctx || !src || !dst) {
            return false;
        }

        if (src->format != m_src_fmt || src->width != m_src_width || src->height != m_src_height ||
            dst->format != m_dst_fmt || dst->width != m_dst_width || dst->height != m_dst_height) {
            return false;
        }

        if (dst->format == AV_PIX_FMT_BGR24) {
            dst->linesize[0] = m_dst_width * 3;
        }

        /* allocate buffer if need for dst */
        if (dst->data[0] == NULL) {
            auto ret = av_frame_get_buffer(dst.get(), 0);
            if (ret < 0) {
                return false;
            }
        }
        
        auto ret = sws_scale(sws_ctx, src->data, src->linesize, 0, src->height, dst->data, dst->linesize);
        dst->pts = src->pts;
        dst->pkt_dts = src->pkt_dts;
        return ret >= 0;
    }

    ff_packet_queue::ff_packet_queue(int max_num): m_max_num_(max_num) {}
    
    void ff_packet_queue::push(ff_av_packet_ptr packet) {
        std::unique_lock<std::mutex> lock(mutex_);
        queue_.push(packet);
        if (queue_.size() > m_max_num_) {
            queue_.pop();  // make sure queue is not too much data
            return;
        }
        cond_var_.notify_one();
    }

    bool ff_packet_queue::try_pop(ff_av_packet_ptr& packet) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            return false;
        }
        packet = queue_.front();
        queue_.pop();
        return true;
    }

    void ff_packet_queue::wait_and_pop(ff_av_packet_ptr& packet) {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_var_.wait(lock, [this] { return !queue_.empty(); });
        packet = queue_.front();
        queue_.pop();
    }
}
#endif