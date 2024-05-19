#include "Scaler.h"

namespace vp {

Scaler::Scaler(int           srcWidth,
               int           srcHeight,
               AVPixelFormat srcFormat,
               int           dstWidth,
               int           dstHeight,
               AVPixelFormat dstFormat)
    : m_dstWidth(dstWidth),
      m_dstHeight(dstHeight),
      m_dstFormat(dstFormat),
      m_srcWidth(srcWidth),
      m_srcHeight(srcHeight),
      m_srcFormat(srcFormat),
      m_sws_ctx(std::shared_ptr<SwsContext>(sws_getContext(srcWidth,
                                                           srcHeight,
                                                           srcFormat,
                                                           dstWidth,
                                                           dstHeight,
                                                           dstFormat,
                                                           SWS_FAST_BILINEAR,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr),
                                            [](SwsContext *ctx) { sws_freeContext(ctx); })) {}

bool Scaler::scale_avframe_to_uint8(av_frame src, uint8_t *dst) {
    if (src->format != m_srcFormat || src->width != m_srcWidth || src->height != m_srcHeight) {
        return false;
    }
    av_frame t_frame = alloc_av_frame();
    t_frame->width   = m_dstWidth;
    t_frame->height  = m_dstHeight;
    t_frame->format  = m_dstFormat;
    av_frame_get_buffer(t_frame.get(), 32);
    if (!scale_avframe_to_avframe(src, t_frame)) {
        return false;
    }
    av_image_copy_to_buffer(dst, m_dstWidth * m_dstHeight * 3, t_frame->data, t_frame->linesize,
                            m_dstFormat, m_dstWidth, m_dstHeight, 1);
    return true;
}

bool Scaler::scale_uint8_to_avframe(uint8_t *src, av_frame &dst) {
    if (dst->format != m_dstFormat || dst->width != m_dstWidth || dst->height != m_dstHeight) {
        return false;
    }
    av_frame t_frame = alloc_av_frame();
    t_frame->width   = m_srcWidth;
    t_frame->height  = m_srcHeight;
    t_frame->format  = m_srcFormat;
    av_frame_get_buffer(t_frame.get(), 32);
    av_image_fill_arrays(t_frame->data, t_frame->linesize, src, m_srcFormat, m_srcWidth,
                         m_srcHeight, 1);
    if (!scale_avframe_to_avframe(t_frame, dst)) {
        return false;
    }
    return true;
}

bool Scaler::scale_uint8_to_uint8(uint8_t *src, uint8_t *dst) {
    av_frame in_frame = alloc_av_frame();
    in_frame->width   = m_srcWidth;
    in_frame->height  = m_srcHeight;
    in_frame->format  = m_srcFormat;
    av_frame_get_buffer(in_frame.get(), 32);
    av_image_fill_arrays(in_frame->data, in_frame->linesize, src, m_srcFormat, m_srcWidth,
                         m_srcHeight, 1);
    av_frame out_frame = alloc_av_frame();
    out_frame->width   = m_dstWidth;
    out_frame->height  = m_dstHeight;
    out_frame->format  = m_dstFormat;
    av_frame_get_buffer(out_frame.get(), 32);
    if (!scale_avframe_to_avframe(in_frame, out_frame)) {
        return false;
    }
    av_image_copy_to_buffer(dst, m_dstWidth * m_dstHeight * 3, out_frame->data,
                            out_frame->linesize, m_dstFormat, m_dstWidth, m_dstHeight, 1);
    return true;
}

bool Scaler::scale_avframe_to_avframe(av_frame src, av_frame &dst) {
    if (src->format != m_srcFormat || src->width != m_srcWidth || src->height != m_srcHeight) {
        std::cout << "src format error" << std::endl;
        return false;
    }
    if (dst->format != m_dstFormat || dst->width != m_dstWidth || dst->height != m_dstHeight) {
        std::cout << "dst format error" << std::endl;
        return false;
    }
    if (dst->format == AV_PIX_FMT_BGR24) {
        dst->linesize[0] = m_dstWidth * 3;
    }
    int ret = sws_scale(m_sws_ctx.get(), src->data, src->linesize, 0, src->height, dst->data,
                        dst->linesize);
    dst->pts = src->pts;
    dst->pkt_dts = src->pkt_dts;
    return ret >= 0;
}

    bool Scaler::scale_avframe_to_cvMat(std::shared_ptr<AVFrame> src, cv::Mat &dst) {
        if (m_sws_ctx){
            int  cvLinesizes[1];
            cvLinesizes[0] = dst.step1();
            int ret = sws_scale(m_sws_ctx.get(), src->data, src->linesize, 0, src->height, &dst.data,
                                cvLinesizes);
            return ret >= 0;
        }
        return false;
    }

    bool Scaler::scale_cvMat_to_avframe(cv::Mat src, std::shared_ptr<AVFrame> &dst) {
        if (m_sws_ctx){
            if(!dst) return false;
            dst->width = src.cols;
            dst->height = src.rows;
            dst->format = AV_PIX_FMT_YUV420P;
            av_frame_get_buffer(dst.get(), 32);

            uint8_t *inData[AV_NUM_DATA_POINTERS] = {0};
            inData[0] = src.data;
            int inLinesize[AV_NUM_DATA_POINTERS] = {0};
            inLinesize[0] = src.cols * src.elemSize();
            int ret = sws_scale(m_sws_ctx.get(), inData, inLinesize, 0, src.rows, dst->data,
                                dst->linesize);
            return ret >= 0;
        }
        return false;
    }

}  // namespace vp