#pragma once
#ifdef VP_WITH_FFMPEG
#include <string>
#include <queue>
#include <vector>
#include <mutex>
#include <algorithm>
#include <functional>
#include "ff_common.h"
#include "../../utils/vp_semaphore.h"
#include "../../utils/vp_utils.h"
#include "../../utils/logger/vp_logger.h"

namespace vp_nodes {
    /**
     * encode and enmux using FFmpeg.
     * used to encode & enmux network streams or file streams.
     */
    class ff_des: public std::enable_shared_from_this<ff_des> {
    private:
        /* core members */
        const std::vector<std::string> m_supported_files = {"mp4", "mkv", "flv", "h265", "h264"};
        const std::vector<std::string> m_supported_protocols = {"rtsp", "rtmp", "udp", "rtp"};
        AVFormatContext* m_ofmt_ctx = nullptr;
        AVCodecContext* m_enc_ctx = nullptr;
        std::queue<ff_av_packet_ptr> m_enmux_packets_q;
        std::queue<ff_av_frame_ptr> m_encode_frames_q;
        std::mutex m_enmux_packets_m;
        std::mutex m_encode_frames_m;
        int m_enmux_packets_q_max_size = 25;
        int m_encode_frames_q_max_size = 25;
        std::shared_ptr<std::thread> m_enmux_th = nullptr;
        std::shared_ptr<std::thread> m_encode_th = nullptr;
        vp_utils::vp_semaphore m_enmux_semaphore;
        vp_utils::vp_semaphore m_encode_semaphore;

        /**
         * live stream or not for output.
         */
        bool m_live_stream = false;

        /**
         * fps of output stream.
         */
        int m_fps = 0;
        
        /**
         * width of output stream in pixels.
         */
        int m_width = 0;

        /**
         * height of output stream in pixels.
         */
        int m_height = 0;

        /**
         * bitrate of output stream (kbit/s).
         */
        long m_bitrate = 0;

        /**
         * codec type name of output stream (strings like `h264/hevc/vp8/...`).
         */
        std::string m_codec_name = "";

        /**
         * pixel format of output stream (strings like `YUV420P/NV12/...`). 
         */
        std::string m_pixel_format = "";

        /**
         * max B frames for encode.
         */
        int m_max_b_frames = 0;

        /**
         * channel index of output.
         */
        int m_channel_index = -1;

        /**
         * type name of hardware used for encode (`none` if no hardware used).
         */
        std::string m_hw_type_name = "";

        /**
         * encoder name used for encoding.
         */
        std::string m_encoder_name = "";

        /**
         * uri for output (rtmp/file/...).
         */
        std::string m_uri = "";

        /* inner flags */
        int m_inner_stream_index = -1;
        bool m_enmux_running = false;
        bool m_encode_running = false;

        /* inner methods */
        void enmux_run();
        void encode_run();
        void inner_close();
        void inner_exit_signal();
        std::string print_summary();
        bool inner_open(const std::string& uri,
                        int width, int height, int fps, int bitrate, int max_b_frames,
                        const std::string& encoder_name = "", 
                        AVPixelFormat sw_pix_fmt = AVPixelFormat::AV_PIX_FMT_YUV420P,
                        AVHWDeviceType hw_type = AVHWDeviceType::AV_HWDEVICE_TYPE_NONE,
                        AVPixelFormat hw_pix_fmt = AVPixelFormat::AV_PIX_FMT_NONE);
        int hw_encoder_init(AVCodecContext* enc_ctx, const enum AVHWDeviceType hw_type, const enum AVPixelFormat sw_pix_fmt);

    public:
        /**
         * create ff_des instance using initial parameters.
         * 
         * @param channel_index specify the channel index for output stream.
         */
        ff_des(int channel_index);
        ~ff_des();

        /* disable copy and assignment operations */
        ff_des(const ff_des&) = delete;
        ff_des& operator=(const ff_des&) = delete;

        /**
         * try to open ff_des, not thread-safe.
         * 
         * @param 
         */
        bool open(const std::string& uri,
                  int width, int height, int fps, int bitrate, int max_b_frames = 0,
                  const std::string& encoder_name = "", 
                  AVPixelFormat sw_pix_fmt = AVPixelFormat::AV_PIX_FMT_YUV420P,
                  AVHWDeviceType hw_type = AVHWDeviceType::AV_HWDEVICE_TYPE_NONE,
                  AVPixelFormat hw_pix_fmt = AVPixelFormat::AV_PIX_FMT_NONE);
        
        /**
         * close ff_des.
         */
        void close();

        /**
         * if working or not.
         * 
         * @return
         * true if it is working.
         */
        bool is_opened() const;

        /**
         * write the next frame to ff_des, keep as same as cv::VideoWriter.
         * 
         * @param frame frame to be written.
         * 
         * @return
         * true if write successfully.
         * 
         * @note
         * return false means:
         * 1. write too quickly no space prepared, please try to write again.
         * 2. ff_des not opened yet or not working.
         */
        bool write(const ff_av_frame_ptr& frame);

        /**
         * write the next frame to ff_des, support for `<<` operator.
         * 
         * @param frame frame to be written.
         * 
         * @return
         * reference for ff_des.
         */
        ff_des& operator<<(const ff_av_frame_ptr& frame);

        /**
         * get fps of output stream in pixels.
         */
        int get_video_fps() const;

        /**
         * get width of output stream in pixels.
         */
        int get_video_width() const;

        /**
         * get height of output stream in pixels.
         */
        int get_video_height() const;

        /**
         * get bitrate of output stream (kbit/s).
         */
        long get_video_bitrate() const;

        /**
         * get codec type name of output stream (strings like `h264/hevc/vp8/...`).
         */
        std::string get_video_codec_name() const;

        /**
         * get pixel format of output stream (strings like `YUV420P/NV12/...`). 
         */
        std::string get_video_pixel_format_name() const;

        /**
         * check is live stream or not.
         */
        bool is_live_stream() const;

        /**
         * get type name of hardware used for encoding (strings like 'cuda/vaapi', 'none' if no hardware used).
         */
        std::string get_hw_type_name() const;

        /**
         * get encoder name used for encoding.
         */
        std::string get_encoder_name() const;

        /**
         * get uri of output.
         */
        std::string get_uri() const;

        /**
         * get channel index of output.
         */
        int get_channel_index() const;

        /**
         * get pointer of encode context from FFmpeg.
         * used to allocate hardware AVFrame outside of ff_des.
         */
        const AVCodecContext* get_encode_ctx() const;
    };
}
#endif