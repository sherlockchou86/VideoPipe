//
// Created by jin_li on 2024/5/17.
//

#pragma once

#include "SafeAVFormat.h"
#include "Scaler.h"

namespace vp {

class ImageSaver {
private:
    int                     m_from_width;
    int                     m_from_height;
    int                     m_to_width;
    int                     m_to_height;
    int                     m_format;
    av_codec_context        m_codec_ctx;
    std::shared_ptr<Scaler> m_scaler;

public:
    ImageSaver(int input_width, int input_height, int input_format, int output_width, int output_height);

    /**
     * @brief 将帧转换为jpeg格式的图片
     * @param frame
     * @return 编码后的图片数据保存在packet中
     */
    av_packet frame_to_jpeg(av_frame frame);
};

}  // namespace vp
