#ifdef VP_WITH_FFMPEG
#include <iostream>
#include <sstream>
#include "ff_src.h"

namespace vp_nodes {
    ff_src::ff_src(int channel_index): m_channel_index(channel_index) {
    }
    
    ff_src::~ff_src() {
        inner_close();
    }

    void ff_src::demux_run() {
        auto start_time = std::chrono::system_clock::now();
        auto total_bytes = 0;
        while (m_demux_running) {
            auto demux_start_t = std::chrono::system_clock::now();
            auto ff_packet = alloc_ff_av_packet();
            auto ret = 0;

            if ((ret = av_read_frame(m_ifmt_ctx, ff_packet.get())) < 0) {
                VP_ERROR(vp_utils::string_format("[ffio/ff_src][%d][demux_run] av_read_frame failed. ret: %d", 
                                                 get_channel_index(), 
                                                 ret));
                break;
            }

            // only demux the right video stream
            if (ff_packet->stream_index != m_inner_stream_index) {
                continue;
            }
/*
            // analyse
            vp_media::stream_analyser analyser(ff_packet->data, ff_packet->size, true);
            std::vector<vp_media::nal_unit> nal_units;
            analyser.analyse(nal_units);
            std::ostringstream oss;
            for (auto& nal: nal_units) {
                oss << std::setw(8) << std::setfill(' ') << nal.index 
                    << std::setw(16) << std::setfill(' ') << nal.offset
                    << std::setw(8) << std::setfill(' ') << nal.nal_length 
                    << std::setw(16) << std::setfill(' ') << vp_media::stream_analyser::to_hex(nal.start_bytes) << vp_media::stream_analyser::to_hex(nal.head_bytes, false) 
                    << std::setw(4) << std::setfill(' ') <<  nal.nal_type
                    << std::setw(24) << std::setfill(' ') << nal.nal_type_name << std::endl;
            }
            std::cout << oss.str();
*/
            // need calculate bitrate manually
            if (m_bitrate <= 0) {
                total_bytes += ff_packet->size;
                auto current_time = std::chrono::system_clock::now();
                auto elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time);
                if (elapsed_seconds.count() > 5.0) {
                    m_bitrate = (total_bytes * 8) / elapsed_seconds.count() / 1024;   // convert to kbit/s 
                    total_bytes = 0;
                    start_time = std::chrono::system_clock::now();
                }
            }
            
            bool notify = true;
            {
                std::lock_guard<std::mutex> g(m_demux_packets_m);
                m_demux_packets_q.push(ff_packet);
                if (m_demux_packets_q.size() > m_demux_packets_q_max_size) {
                    m_demux_packets_q.pop();
                    notify = false;
                    VP_WARN(vp_utils::string_format("[ffio/ff_src][%d][demux_run] exceed m_demux_packets_q_max_size(%d), discard the front in queue.", 
                                                    get_channel_index(), 
                                                    m_demux_packets_q_max_size));
                }
            }
            if (notify) {
                m_demux_semaphore.signal();
            }

            /* 1. wait for a while for video file
             * 2. demux as soon as possible for live stream
            */
            if (!m_live_stream) {
                auto cost_t = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - demux_start_t);
                auto duration_t = std::chrono::milliseconds(int(1000.0 / m_fps));
                if (cost_t < duration_t) {
                    std::this_thread::sleep_for(duration_t - cost_t);
                }
            }
        }

        /* set demux running flag to false in case of abnormal exit */
        m_demux_running = false;
        {
            // send exit flag to notify decode thread
            std::lock_guard<std::mutex> g(m_demux_packets_m);
            m_demux_packets_q.push(nullptr);
            VP_INFO(vp_utils::string_format("[ffio/ff_src][%d][demux_run] send exit flag to decode thread.", 
                                            get_channel_index()));
        }
        m_demux_semaphore.signal();
        VP_INFO(vp_utils::string_format("[ffio/ff_src][%d][demux_run] demux thread exits.", 
                                        get_channel_index()));
    }

    void ff_src::decode_run() {
        while (m_decode_running) {
            m_demux_semaphore.wait();
            ff_av_packet_ptr ff_packet = nullptr;
            {
                std::lock_guard<std::mutex> g(m_demux_packets_m);
                ff_packet = m_demux_packets_q.front();
                m_demux_packets_q.pop();
            }

            auto ret = 0;
            /* get exit flag */
            if (!ff_packet) {
                VP_INFO(vp_utils::string_format("[ffio/ff_src][%d][decode_run] get exit flag, go to flush decoder.", 
                                                get_channel_index()));
                ret = avcodec_send_packet(m_dec_ctx, NULL);  // flush decoder
            } else {
                ret = avcodec_send_packet(m_dec_ctx, ff_packet.get());
            }

            if (ret < 0) {
                VP_ERROR(vp_utils::string_format("[ffio/ff_src][%d][decode_run] avcodec_send_packet failed. ret: %d", 
                                                 get_channel_index(), ret));
                break;
            }
            
            while (ret >= 0) {
                auto ff_frame = alloc_ff_av_frame();
                ret = avcodec_receive_frame(m_dec_ctx, ff_frame.get());
                if (ret == AVERROR(EAGAIN)) {
                    break;
                } else if (ret < 0) {
                    VP_ERROR(vp_utils::string_format("[ffio/ff_src][%d][decode_run] avcodec_receive_frame failed. ret: %d", 
                                                     get_channel_index(), 
                                                     ret));
                    ff_packet = nullptr;
                    break;
                }
                {
                    std::lock_guard<std::mutex> g(m_decode_frames_m);
                    m_decode_frames_q.push(ff_frame);
                    if (m_decode_frames_q.size() > m_decode_frames_q_max_size) {
                        m_decode_frames_q.pop();
                        VP_WARN(vp_utils::string_format("[ffio/ff_src][%d][decode_run] exceed m_decode_frames_q_max_size(%d), discard the front in queue.", 
                                                        get_channel_index(), 
                                                        m_decode_frames_q_max_size));
                    }
                }
            }

            if (!ff_packet) {
                break;
            }
        }

        /* set decode running flag to false in case of abnormal exit */
        m_decode_running = false;
        VP_INFO(vp_utils::string_format("[ffio/ff_src][%d][decode_run] decode thread exits.", 
                                        get_channel_index()));
    }   

    bool ff_src::open(const std::string& uri, const std::string& decoder_name, AVHWDeviceType hw_type) {
        inner_close();
        auto uri_valid = false;
        if (!uri_valid) {
            auto uri_parts = vp_utils::string_split(uri, '.');  // file
            if (std::find(m_supported_files.begin(), m_supported_files.end(), uri_parts[uri_parts.size() - 1]) != m_supported_files.end()) {
                uri_valid = true;
                m_live_stream = false;
            }
        }
        if (!uri_valid) {
            auto uri_parts = vp_utils::string_split(uri, ':');  // live stream
            if (std::find(m_supported_protocols.begin(), m_supported_protocols.end(), uri_parts[0]) != m_supported_protocols.end()) {
                uri_valid = true;
                m_live_stream = true;
            }
        }
        if (!uri_valid) {
            VP_ERROR(vp_utils::string_format("[ffio/ff_src][%d][open] uri invalid! uri MUST start with `rtsp/rtmp/..`(network streams) or end with `mp4/mkv/..`(file streams).", 
                                             get_channel_index()));
            return false;
        }
        
        return inner_open(uri, decoder_name, hw_type);
    }

    bool ff_src::inner_open(const std::string& uri, const std::string& decoder_name, AVHWDeviceType hw_type) {
        auto ret = 0;
        AVDictionary *options = NULL;
        // Set options
        av_dict_set(&options, "max_delay", "100000", 0);       // 设置最大延迟为100ms
        av_dict_set(&options, "buffer_size", "2000000", 0);    // 设置缓冲区大小为2MB
        av_dict_set(&options, "timeout", "3000000", 0);        // 设置超时时间为3秒
        av_dict_set(&options, "rtsp_transport", "tcp", 0);     // 使用TCP传输
        av_dict_set(&options, "max_interleave_delta", "1000000", 0);  // 设置最大间隔时间为1秒

        /* open input uri(file/stream), and allocate format context */
        if (avformat_open_input(&m_ifmt_ctx, uri.c_str(), NULL, NULL) < 0) {
            VP_ERROR(vp_utils::string_format("[ffio/ff_src][%d][inner_open] could not open input uri.", 
                                             get_channel_index()));
            return false;
        }
        /* retrieve stream information */
        if (avformat_find_stream_info(m_ifmt_ctx, NULL) < 0) {
            VP_ERROR(vp_utils::string_format("[ffio/ff_src][%d][inner_open] could not find stream information.", 
                                             get_channel_index()));
            avformat_close_input(&m_ifmt_ctx);
            return false;
        }
        /* find video stream */
        if ((ret = av_find_best_stream(m_ifmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0)) < 0) {
            VP_ERROR(vp_utils::string_format("[ffio/ff_src][%d][inner_open] could not find proper VIDEO stream.", 
                                             get_channel_index()));
            avformat_close_input(&m_ifmt_ctx);
            return false;
        }
        m_inner_stream_index = ret;
        auto in_stream = m_ifmt_ctx->streams[m_inner_stream_index];
        
        /* initialize video properties */
        m_fps = int(av_q2d(in_stream->r_frame_rate));
        m_width = in_stream->codecpar->width;
        m_height = in_stream->codecpar->height;
        m_bitrate = in_stream->codecpar->bit_rate / 1024;   // kb/s
        m_codec_name = std::string(avcodec_get_name(in_stream->codecpar->codec_id));
        m_duration = av_q2d(in_stream->time_base) * in_stream->duration;
        m_duration = m_duration < 0 ? 0 : m_duration;
        m_pixel_format = std::string(av_get_pix_fmt_name(static_cast<AVPixelFormat>(in_stream->codecpar->format)));
        
        /* find decoder for the input video stream */
        const AVCodec* dec = nullptr;
        if (decoder_name.empty()) {
            dec = avcodec_find_decoder(in_stream->codecpar->codec_id);   // get default decoder by AVCodecID
        } else {
            dec = avcodec_find_decoder_by_name(decoder_name.c_str());    // get by decoder name
        }
        if (!dec) {
            VP_ERROR(vp_utils::string_format("[ffio/ff_src][%d][inner_open] could not find the proper decoder for input stream.", 
                                             get_channel_index()));
            avformat_close_input(&m_ifmt_ctx);
            return false;
        }
        /* allocate codec context for the decoder */
        m_dec_ctx = avcodec_alloc_context3(dec);
        if (!m_dec_ctx) {
            VP_ERROR(vp_utils::string_format("[ffio/ff_src][%d][inner_open] could not allocate context for decoder(%s).", 
                                             get_channel_index(), dec->name));
            avformat_close_input(&m_ifmt_ctx);
            return false;
        }
        /* copy codec parameters from input stream to codec context */
        if ((ret = avcodec_parameters_to_context(m_dec_ctx, in_stream->codecpar)) < 0) {
            VP_ERROR(vp_utils::string_format("[ffio/ff_src][%d][inner_open] could not copy codec parameters to decode context. ret: %d", 
                                             get_channel_index(), ret));
            avcodec_free_context(&m_dec_ctx);
            avformat_close_input(&m_ifmt_ctx);
            return false;
        }
        /* check if need init hwaccels context for decoder */
        if (hw_type != AVHWDeviceType::AV_HWDEVICE_TYPE_NONE 
            && hw_decoder_init(m_dec_ctx, hw_type) < 0) {
            VP_ERROR(vp_utils::string_format("[ffio/ff_src][%d][inner_open] failed to create specified(%s) HW device context for decoder.", 
                                             get_channel_index(), av_hwdevice_get_type_name(hw_type)));
            avcodec_free_context(&m_dec_ctx);
            avformat_close_input(&m_ifmt_ctx);
            return false;
        }
        /* open the decoder */
        if ((ret = avcodec_open2(m_dec_ctx, dec, NULL)) < 0) {
            VP_ERROR(vp_utils::string_format("[ffio/ff_src][%d][inner_open] could not open decoder. ret: %d", 
                                             get_channel_index(), ret));
            avcodec_free_context(&m_dec_ctx);
            avformat_close_input(&m_ifmt_ctx);
            return false;
        }
        /* initialize node properties */
        m_decoder_name = decoder_name.empty() ? std::string(m_dec_ctx->codec->name) : decoder_name;
        m_uri = uri;
        m_hw_type_name = hw_type != AV_HWDEVICE_TYPE_NONE ? std::string(av_hwdevice_get_type_name(hw_type)) : "none";

        /* collect summary notify caller */
        auto summary = print_summary();
        if (m_src_opened_hooker) {
            m_src_opened_hooker(shared_from_this(), summary);
        }
        
        /* go! */
        m_demux_running = true;
        m_decode_running = true;
        m_demux_th = std::make_shared<std::thread>(&ff_src::demux_run, this);
        m_decode_th = std::make_shared<std::thread>(&ff_src::decode_run, this);
        VP_INFO(vp_utils::string_format("[ffio/ff_src][%d][inner_open] open successfully.", 
                                         get_channel_index()));
        return true;   
    }

    void ff_src::close() {
        inner_close();
    }

    void ff_src::inner_close() {
        /* set running flags to false */
        m_demux_running = false;
        m_decode_running = false;

        /* waiting for threads exist */
        if (m_demux_th != nullptr && m_demux_th->joinable()) {
            m_demux_th->join();
        }
        if (m_decode_th != nullptr && m_decode_th->joinable()) {
            m_decode_th->join();
        }

        /* free format & codec context */
        if (m_ifmt_ctx) {
            avformat_close_input(&m_ifmt_ctx);
        }
        if (m_dec_ctx) {
            avcodec_free_context(&m_dec_ctx);
        }

        /* clear queue */
        {
            // clear m_demux_packets_q
            std::lock_guard<std::mutex> g(m_demux_packets_m);
            m_demux_packets_q = {};
            m_demux_semaphore.reset();
        } 
        {
            // clear m_decode_frames_q
            std::lock_guard<std::mutex> g(m_decode_frames_m);
            m_decode_frames_q = {};
        }
        VP_INFO(vp_utils::string_format("[ffio/ff_src][%d][inner_close] close successfully.", 
                                         get_channel_index()));
    }

    bool ff_src::is_opened() const {
        return m_demux_running && m_decode_running; 
    }

    int ff_src::get_video_fps() const {
        // no check is_opened or not
        return m_fps;
    }

    int ff_src::get_video_width() const {
        // no check is_opened or not
        return m_width;
    }

    int ff_src::get_video_height() const {
        // no check is opened or not
        return m_height;
    }

    long ff_src::get_video_bitrate() const {
        // no check is opened or not
        return m_bitrate;
    }

    std::string ff_src::get_video_codec_name() const {
        // no check is_opened or not
        return m_codec_name;
    }

    double ff_src::get_video_duration() const {
        // no check is opened or not
        return m_duration;    
    }

    bool ff_src::is_live_stream() const {
        // no check is opened or not
        return m_live_stream;
    }

    std::string ff_src::get_hw_type_name() const {
        // no check is opened or not
        return m_hw_type_name;
    }

    std::string ff_src::get_decoder_name() const {
        // no check is opened or not
        return m_decoder_name;
    }

    std::string ff_src::get_uri() const {
        // no check is opened or not
        return m_uri;
    }

    int ff_src::get_channel_index() const {
        // no check is opened or not
        return m_channel_index;
    }

    std::string ff_src::get_video_pixel_format_name() const {
        // no check is opened or not
        return m_pixel_format;
    }

    std::string ff_src::print_summary() {
        std::ostringstream s_stream;
        s_stream << std::endl;
        s_stream << "################# summary for [ff_src] #################" << std::endl;
        s_stream << "|channel_index     |: " << get_channel_index() << std::endl;
        s_stream << "|uri               |: " << get_uri() << std::endl;
        s_stream << "|is_live_stream    |: " << (is_live_stream() ? std::string("YES") : std::string("NO")) << std::endl;
        s_stream << "|decoder_name      |: " << get_decoder_name() << std::endl;
        s_stream << "|hw_type_name      |: " << get_hw_type_name() << std::endl;
        s_stream << "|------------------|  " << std::endl;
        s_stream << "|video_codec       |: " << get_video_codec_name() << std::endl;
        s_stream << "|video_pic_format  |: " << get_video_pixel_format_name() << std::endl;
        s_stream << "|video_fps         |: " << get_video_fps() << std::endl;
        s_stream << "|video_width       |: " << get_video_width() << std::endl;
        s_stream << "|video_height      |: " << get_video_height() << std::endl;
        s_stream << "|video_bitrate     |: " << get_video_bitrate() << " [kbit/s]" << std::endl;
        s_stream << "|video_duration    |: " << get_video_duration() << " [seconds]" << std::endl;
        s_stream << "|------------------|  " << std::endl;
        s_stream << "################# summary for [ff_src] #################" << std::endl;

        auto summary = s_stream.str();
        VP_INFO(summary);
        return summary;
    }

    bool ff_src::read(ff_av_frame_ptr& frame) {
        auto ret = false;
        {
            std::lock_guard<std::mutex> g(m_decode_frames_m);
            if (!m_decode_frames_q.empty()) {
                frame = m_decode_frames_q.front();
                m_decode_frames_q.pop();
                ret = true;
            } else {
                frame = nullptr;
            }
        }
        if (!ret) {
            // switch control of CPUs if no frame returned, avoid of occupying CPUs for a long time by caller outside
            //std::this_thread::sleep_for(std::chrono::milliseconds(1));
            //VP_DEBUG(vp_utils::string_format("[ffio/ff_src][%d][read] read frame failed, sleep for 1 millisecond.", 
            //                                 get_channel_index()));
        }

        /* false means no frame returned (not opened or read too quickly so data not prepared) */
        return ret;
    }

    ff_src& ff_src::operator>>(ff_av_frame_ptr& frame) {
        read(frame);
        return *this;
    }

    int ff_src::hw_decoder_init(AVCodecContext* dec_ctx, const enum AVHWDeviceType type) {
        int err = 0;
        AVBufferRef* hw_device_ctx = nullptr;
        if ((err = av_hwdevice_ctx_create(&hw_device_ctx, type,
                                        NULL, NULL, 0)) < 0) {
            return err;
        }
        // update using AVHWDeviceContext*(hw_device_ctx->data)
        dec_ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);
        av_buffer_unref(&hw_device_ctx);
        return err;
    }

    void ff_src::set_src_opened_hooker(ff_src_opened_hooker src_opened_hooker) {
        m_src_opened_hooker = src_opened_hooker;
    }
}
#endif