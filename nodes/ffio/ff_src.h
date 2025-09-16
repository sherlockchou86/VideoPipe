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
    class ff_src;
    typedef std::function<void(ff_src_ptr, const std::string&)> ff_src_opened_hooker;

    /**
     * demux and decode using FFmpeg.
     * used to demux & decode network streams or file streams.
     */
    class ff_src: public std::enable_shared_from_this<ff_src> {
    private:
        /* core members */
        const std::vector<std::string> m_supported_files = {"mp4", "mkv", "flv", "avi", "h264"};
        const std::vector<std::string> m_supported_protocols = {"rtsp", "rtmp", "http", "rtp"};
        AVFormatContext* m_ifmt_ctx = nullptr;
        AVCodecContext* m_dec_ctx = nullptr;
        std::queue<ff_av_packet_ptr> m_demux_packets_q;
        std::queue<ff_av_frame_ptr> m_decode_frames_q;
        std::mutex m_demux_packets_m;
        std::mutex m_decode_frames_m;        
        int m_demux_packets_q_max_size = 25;
        int m_decode_frames_q_max_size = 25;
        std::shared_ptr<std::thread> m_demux_th = nullptr;
        std::shared_ptr<std::thread> m_decode_th = nullptr;
        vp_utils::vp_semaphore m_demux_semaphore;

        /**
         * live stream or not for input.
         */
        bool m_live_stream = false;
        
        /**
         * fps of input stream.
         */
        int m_fps = 0;
        
        /**
         * width of input stream in pixels.
         */
        int m_width = 0;

        /**
         * height of input stream in pixels.
         */
        int m_height = 0;

        /**
         * bitrate of input stream (kbit/s).
         */
        long m_bitrate = 0;

        /**
         * codec type name of input stream (strings like `h264/hevc/vp8/...`).
         */
        std::string m_codec_name = "";

        /**
         * pixel format of input stream (strings like `YUV420P/NV12/...`). 
         */
        std::string m_pixel_format = "";

        /**
         * duration of input stream (seconds), only for file stream.
         */
        double m_duration = 0.0;

        /**
         * channel index of input.
         */
        int m_channel_index = -1;

        /**
         * type name of hardware used for decoding (`none` if no hardware used).
         */
        std::string m_hw_type_name = "";

        /**
         * decoder name used for decoding.
         */
        std::string m_decoder_name = "";

        /**
         * uri of input (rtsp/file/...).
         */
        std::string m_uri = "";

        /* inner flags */
        int m_inner_stream_index = -1;
        bool m_demux_running = false;
        bool m_decode_running = false;

        /* inner methods */
        void demux_run();
        void decode_run();
        void inner_close();
        std::string print_summary();
        bool inner_open(const std::string& uri, const std::string& decoder_name, AVHWDeviceType hw_type);
        int hw_decoder_init(AVCodecContext* dec_ctx, const enum AVHWDeviceType type);

        /* callbacks */
        ff_src_opened_hooker m_src_opened_hooker;
    public:
        /**
         * create ff_src instance using initial parameters.
         * 
         * @param channel_index specify the channel index for input stream.
         */
        ff_src(int channel_index);
        ~ff_src();

        /* disable copy and assignment operations */
        ff_src(const ff_src&) = delete;
        ff_src& operator=(const ff_src&) = delete;

        /**
         * try to open ff_src, not thread-safe.
         * 
         * @param uri uri to open, url for network streams or file path for file streams.
         * @param decoder_name specify the decoder name used for decoding, MUST supported by FFmpeg.
         * @param hw_type type of hardware for decoding, MUST supported by FFmpeg.
         * 
         * @note
         * if `decoder_name` not specified, ff_src will choose the default decoder accordding to the codec type of input stream.
         */
        bool open(const std::string& uri, 
                  const std::string& decoder_name = "", 
                  AVHWDeviceType hw_type = AVHWDeviceType::AV_HWDEVICE_TYPE_NONE);
        /**
         * close ff_src.
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
         * read the next frame from ff_src, keep as same as cv::VideoCapture.
         * 
         * @param frame frame to be returned (return nullptr if read failed).
         * 
         * @return
         * true if read successfully.
         * 
         * @note
         * return false means:
         * 1. read too quickly no data prepared, please try to read again.
         * 2. ff_src not opened yet or not working.
         */
        bool read(ff_av_frame_ptr& frame);

        /**
         * read the next frame from ff_src, support for `>>` operator.
         * 
         * @param frame frame to be returned (return nullptr if read failed).
         * 
         * @return
         * reference for ff_src.
         */
        ff_src& operator>>(ff_av_frame_ptr& frame);

        /**
         * get fps of input stream.
         */
        int get_video_fps() const;

        /**
         * get width of input stream in pixels.
         */
        int get_video_width() const;

        /**
         * get height of input stream in pixels.
         */
        int get_video_height() const;

        /**
         * get bitrate of input stream (kbit/s).
         */
        long get_video_bitrate() const;

        /**
         * get codec type name of input stream (strings like `h264/hevc/vp8/...`).
         */
        std::string get_video_codec_name() const;
        
        /**
         * get pixel format of input stream (strings like `YUV420P/NV12/...`). 
         */
        std::string get_video_pixel_format_name() const;
        
        /**
         * get duration of input stream (seconds), only for file stream.
         */
        double get_video_duration() const;
        
        /**
         * check is live stream or not.
         */
        bool is_live_stream() const;
        
        /**
         * get type name of hardware used for decoding (strings like 'cuda/vaapi', 'none' if no hardware used).
         */
        std::string get_hw_type_name() const;
        
        /**
         * get decoder name used for decoding.
         */
        std::string get_decoder_name() const;
        
        /**
         * get uri of input.
         */
        std::string get_uri() const;
        
        /**
         * get channel index of input.
         */
        int get_channel_index() const;

        /**
         * set callback for opened event. would be activated every time ff_src opened.
         * 
         * @param src_opened_hooker callback activated when ff_src opened.
         */
        void set_src_opened_hooker(ff_src_opened_hooker src_opened_hooker);
    };
}
#endif