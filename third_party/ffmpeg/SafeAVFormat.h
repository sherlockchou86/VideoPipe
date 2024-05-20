#pragma once
extern "C" {
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libavutil/imgutils.h"
#include "libswscale/swscale.h"
}

#include <iostream>  // for std::cout
#include <memory>

/*
 * Safe libAVFormat
 * 使用智能指针管理 libAVFormat 的变量
 * 避免内存泄漏
 */

#define av_packet std::shared_ptr<AVPacket>

#define av_format_context std::shared_ptr<AVFormatContext>

#define av_dictionary std::shared_ptr<AVDictionary>

#define av_codec_context std::shared_ptr<AVCodecContext>

#define av_frame std::shared_ptr<AVFrame>

#define av_codec_parameters std::shared_ptr<AVCodecParameters>

#define av_sws_context std::shared_ptr<SwsContext>


#define ASSERT_FFMPEG(FFMPEG_FUNC)                                                                 \
    {                                                                                              \
        int CODE = FFMPEG_FUNC;                                                                    \
        if (CODE == -11) {                                                                         \
            return false;                                                                          \
        }                                                                                          \
        if (CODE < 0) {                                                                            \
            char errbuf[1024];                                                                     \
            av_strerror(FFMPEG_FUNC, errbuf, sizeof(errbuf));                                      \
            std::cout << "\033[41m FFMPEG_FUNC error: \033[0m " << __FILE__ << ":" << __LINE__     \
                      << "\t" << errbuf << " " << CODE << std::endl;                               \
            return false;                                                                          \
        }                                                                                          \
    }

#define alloc_av_codec_parameters()                                                                \
    std::shared_ptr<AVCodecParameters>(avcodec_parameters_alloc(), [](AVCodecParameters *para) {   \
        avcodec_parameters_free(&para);                                                            \
    })

#define alloc_av_frame()                                                                           \
    std::shared_ptr<AVFrame>(av_frame_alloc(), [](AVFrame *frame) { av_frame_free(&frame); })

#define alloc_av_frame_with(AV_FRAME)                                                              \
    std::shared_ptr<AVFrame>(av_frame_clone(AV_FRAME),                                             \
                             [](AVFrame *frame) { av_frame_free(&frame); })

#define alloc_av_format_context()                                                                  \
    std::shared_ptr<AVFormatContext>(avformat_alloc_context(),                                     \
                                     [](AVFormatContext *ctx) { avformat_free_context(ctx); })

#define alloc_av_codec_context(CODEC)                                                              \
    std::shared_ptr<AVCodecContext>(avcodec_alloc_context3(CODEC),                                 \
                                    [](AVCodecContext *ctx) { avcodec_free_context(&ctx); })

#define alloc_av_packet()                                                                          \
    std::shared_ptr<AVPacket>(av_packet_alloc(), [](AVPacket *pkt) { av_packet_free(&pkt); })

#define alloc_av_packet_with(AV_PACKET)                                                            \
    std::shared_ptr<AVPacket>(av_packet_clone(AV_PACKET),                                          \
                              [](AVPacket *pkt) { av_packet_free(&pkt); })

#define DESERIALIZE_INFO_TO_DICT(INFO, DICT)                                                       \
    if (INFO.preset)                                                                               \
        av_dict_set(&DICT, "preset", INFO.preset, 0);                                              \
    if (INFO.muxdelay)                                                                             \
        av_dict_set(&DICT, "muxdelay", INFO.muxdelay, 0);                                          \
    if (INFO.tune)                                                                                 \
        av_dict_set(&DICT, "tune", INFO.tune, 0);                                                  \
    if (INFO.crf)                                                                                  \
        av_dict_set(&DICT, "crf", INFO.crf, 0);                                                    \
    if (INFO.profile)                                                                              \
        av_dict_set(&DICT, "profile", INFO.profile, 0);                                            \
    if (INFO.level)                                                                                \
        av_dict_set(&DICT, "level", INFO.level, 0);                                                \
    if (INFO.rtsp_transport)                                                                       \
        av_dict_set(&DICT, "rtsp_transport", INFO.rtsp_transport, 0);                              \
    if (INFO.stimeout)                                                                             \
        av_dict_set(&DICT, "stimeout", INFO.stimeout, 0);                                          \
    if (INFO.buffer_size)                                                                          \
        av_dict_set(&DICT, "buffer_size", INFO.buffer_size, 0);                                    \
    if (INFO.timeout)                                                                              \
        av_dict_set(&DICT, "timeout", INFO.timeout, 0);                                            \
    av_dict_set(&DICT, "fflags", "nobuffer", 0);

struct av_dictionary_set_info {
    const char *preset   = nullptr;  // 预设
    const char *muxdelay = nullptr;  // 延迟约束 单位秒
    const char *tune     = nullptr;  // 转码延迟，以牺牲视频质量减少时延
    const char *crf = nullptr;  // 指定输出视频的质量，范围0-51，默认23 数字越小质量越高
    const char *profile = nullptr;  // 四种画质级别,分别是baseline, extended, main, high
    const char *level   = nullptr;  // 越高视频质量也就越高
    const char *rtsp_transport = nullptr;  // udp、tcp、rtp
    const char *stimeout       = nullptr;  // 超时断开时间 单位是微秒
    const char *buffer_size    = nullptr;  // 缓冲区大小
    const char *timeout        = nullptr;  // 超时时间 单位是微秒

    /*
     * 预设
     */
    struct av_dict_set_preset {
        const char *ultrafast = "ultrafast";
        const char *superfast = "superfast";
        const char *veryfast  = "veryfast";
        const char *faster    = "faster";
        const char *fast      = "fast";
        const char *medium    = "medium";
        const char *slow      = "slow";
        const char *slower    = "slower";
        const char *veryslow  = "veryslow";
        const char *placebo   = "placebo";
    } av_dict_set_preset;

    /*
     * 转码延迟，以牺牲视频质量减少时延
     */
    struct av_dict_set_tune {
        const char *film        = "film";
        const char *animation   = "animation";
        const char *grain       = "grain";
        const char *stillimage  = "stillimage";
        const char *psnr        = "psnr";
        const char *ssim        = "ssim";
        const char *fastdecode  = "fastdecode";
        const char *zerolatency = "zerolatency";
    } av_dict_set_tune;

    /*
     * 四种画质级别,分别是baseline, extended, main, high
     */
    struct av_dict_set_profile {
        const char *baseline = "baseline";
        const char *extended = "extended";
        const char *main     = "main";
        const char *high     = "high";
    } av_dict_set_profile;

    /*
     * 越高视频质量也就越高
     */
    struct av_dict_set_level {
        const char *level1_0 = "1.0";
        const char *level1_1 = "1.1";
        const char *level1_2 = "1.2";
        const char *level1_3 = "1.3";
        const char *level2_0 = "2.0";
        const char *level2_1 = "2.1";
        const char *level2_2 = "2.2";
        const char *level3_0 = "3.0";
        const char *level3_1 = "3.1";
        const char *level3_2 = "3.2";
        const char *level4_0 = "4.0";
        const char *level4_1 = "4.1";
        const char *level4_2 = "4.2";
        const char *level5_0 = "5.0";
        const char *level5_1 = "5.1";
    } av_dict_set_level;

    /*
     * 传输协议
     */
    struct av_dict_set_rtsp_transport {
        const char *udp = "udp";
        const char *tcp = "tcp";
        const char *rtp = "rtp";
    } av_dict_set_rtsp_transport;
};

enum VP_ERR_CODE {
    ERROR_CODE_NONE = 0,
    INPUTTER_OPEN_FAILED,
    INPUTTER_GET_IMAGES_FAILED,
    OUTPUTTER_OPEN_FAILED,
    OUTPUTTER_WRITE_HEADER_FAILED,
    OUTPUTTER_WRITE_FRAME_FAILED,
    OUTPUTTER_WRITE_TRAILER_FAILED,
};

#define VP_ON_ERR_CB_FUNC(T) std::function<void(VP_ERR_CODE, std::string, std::shared_ptr<T>)>