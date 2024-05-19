//
// Created by jin_li on 2024/5/17.
//

#pragma once
#include "SafeAVFormat.h"
#include <opencv2/core/core.hpp>

namespace vp {
class Scaler {
public:
    /*!
     * @brief 画面缩放
     * @param srcWidth  源宽度
     * @param srcHeight 源高度
     * @param srcFormat 源格式
     * @param dstWidth  目标宽度
     * @param dstHeight 目标高度
     * @param dstFormat 目标格式
     */
    Scaler(int           srcWidth,
           int           srcHeight,
           AVPixelFormat srcFormat,
           int           dstWidth,
           int           dstHeight,
           AVPixelFormat dstFormat);

private:
    bool scale_avframe_to_avframe(av_frame src, av_frame &dst);

    bool scale_avframe_to_uint8(av_frame src, uint8_t *dst);

    bool scale_uint8_to_avframe(uint8_t *src, av_frame &dst);

    bool scale_uint8_to_uint8(uint8_t *src, uint8_t *dst);

    bool scale_avframe_to_cvMat(av_frame src, cv::Mat &dst);

    bool scale_cvMat_to_avframe(cv::Mat src, av_frame &dst);

public:
    static std::shared_ptr<Scaler> createShare(int           srcWidth,
                                               int           srcHeight,
                                               AVPixelFormat srcFormat,
                                               int           dstWidth,
                                               int           dstHeight,
                                               AVPixelFormat dstFormat) {
        return std::make_shared<Scaler>(srcWidth, srcHeight, srcFormat, dstWidth, dstHeight,
                                        dstFormat);
    }

    template <typename F, typename T>
    bool scale(F src, T dst) {
        if constexpr (std::is_same_v<F, av_frame> && std::is_same_v<T, av_frame>) {
            return scale_avframe_to_avframe(src, dst);
        } else if constexpr (std::is_same_v<F, av_frame> && std::is_same_v<T, uint8_t *>) {
            return scale_avframe_to_uint8(src, dst);
        } else if constexpr (std::is_same_v<F, uint8_t *> && std::is_same_v<T, av_frame>) {
            return scale_uint8_to_avframe(src, dst);
        } else if constexpr (std::is_same_v<F, uint8_t *> && std::is_same_v<T, uint8_t *>) {
            return scale_uint8_to_uint8(src, dst);
        } else if constexpr (std::is_same_v<F, av_frame> && std::is_same_v<T, cv::Mat>) {
            return scale_avframe_to_cvMat(src, dst);
        } else if constexpr (std::is_same_v<F, cv::Mat> && std::is_same_v<T, av_frame>) {
            return scale_cvMat_to_avframe(src, dst);
        } else {
            return false;
        }
    }

private:
    int            m_dstWidth;
    int            m_dstHeight;
    AVPixelFormat  m_dstFormat;
    int            m_srcWidth;
    int            m_srcHeight;
    AVPixelFormat  m_srcFormat;
    av_sws_context m_sws_ctx;
};
}  // namespace vp