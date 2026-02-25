#ifdef VP_WITH_FFMPEG
#include <iostream>
#include <sstream>
#include "ff_des.h"

namespace vp_nodes {
    ff_des::ff_des(int channel_index): m_channel_index(channel_index) {
    }

    ff_des::~ff_des() {
        inner_close();
    }

    void ff_des::enmux_run() {
        auto ret = 0;
        while (m_enmux_running) {
            ff_av_packet_ptr ff_packet = nullptr;
            m_enmux_semaphore.wait();
            {
                std::lock_guard<std::mutex> g(m_enmux_packets_m);
                ff_packet = m_enmux_packets_q.front();
                m_enmux_packets_q.pop();
            }

            if (!ff_packet) {
                VP_INFO(vp_utils::string_format("[ffio/ff_des][%d][enmux_run] get exit flag.", 
                                                get_channel_index()));
                break;
            } else {
                /* set parameters for packets */
                av_packet_rescale_ts(ff_packet.get(), m_enc_ctx->time_base, m_ofmt_ctx->streams[m_inner_stream_index]->time_base);
                ff_packet->stream_index = m_inner_stream_index;
                ret = av_interleaved_write_frame(m_ofmt_ctx, ff_packet.get());
            }

            if (ret < 0) {
                VP_ERROR(vp_utils::string_format("[ffio/ff_des][%d][enmux_run] av_interleaved_write_frame failed. ret: %d", 
                                                 get_channel_index(), ret));
                break;
            }
        }
        /* write the trailer for output stream */
        ret = av_write_trailer(m_ofmt_ctx);
        if (ret != 0) {
            VP_ERROR(vp_utils::string_format("[ffio/ff_des][%d][enmux_run] av_write_trailer failed. ret: %d", 
                                             get_channel_index(), ret));
        }
        
        /* set enmux running flag to false in case of abnormal exit */
        m_enmux_running = false;
        VP_INFO(vp_utils::string_format("[ffio/ff_des][%d][enmux_run] enmux thread exits.", 
                                        get_channel_index()));
    }

    void ff_des::encode_run() {
        while (m_encode_running) {
            ff_av_frame_ptr ff_frame;
            m_encode_semaphore.wait();
            {
                std::lock_guard<std::mutex> g(m_encode_frames_m);
                ff_frame = m_encode_frames_q.front();
                m_encode_frames_q.pop();
            }

            auto ret = 0;
            if (!ff_frame) {
                VP_INFO(vp_utils::string_format("[ffio/ff_des][%d][encode_run] get exit flag.", 
                                                get_channel_index()));
                break;
            }
            ff_frame->pts = m_enc_ctx->frame_num;
            ret = avcodec_send_frame(m_enc_ctx, ff_frame.get());
            if (ret < 0) {
                VP_ERROR(vp_utils::string_format("[ffio/ff_des][%d][encode_run] avcodec_send_frame failed. ret: %d", 
                                                 get_channel_index(), ret));
                break;
            }
            
            while (ret >= 0) {
                auto ff_packet = alloc_ff_av_packet();
                ret = avcodec_receive_packet(m_enc_ctx, ff_packet.get());
                if (ret == AVERROR(EAGAIN)) {
                    break;
                } else if (ret < 0) {
                    VP_ERROR(vp_utils::string_format("[ffio/ff_des][%d][encode_run] avcodec_receive_packet failed. ret: %d", 
                                                     get_channel_index(), ret));
                    ff_frame = nullptr;
                    break;
                }

                bool notify = true;
                {
                    std::lock_guard<std::mutex> g(m_enmux_packets_m);
                    m_enmux_packets_q.push(ff_packet);
                    if (m_enmux_packets_q.size() > m_enmux_packets_q_max_size) {
                        m_enmux_packets_q.pop();
                        notify = false;
                        VP_WARN(vp_utils::string_format("[ffio/ff_des][%d][encode_run] exceed m_enmux_packets_q_max_size(%d), discard the front in queue.", 
                                                        get_channel_index(), m_enmux_packets_q_max_size));
                    }
                }
                if (notify) {
                    m_enmux_semaphore.signal();
                }
            }
            
            if (!ff_frame) {
                break;
            }  
        }
        /* set encode running flag to false in case of abnormal exit */
        m_encode_running = false;
        {
            // send exit flag to notify enmux thread
            std::lock_guard<std::mutex> g(m_enmux_packets_m);
            m_enmux_packets_q.push(nullptr);
            VP_INFO(vp_utils::string_format("[ffio/ff_des][%d][encode_run] send exit flag to enmux thread.", 
                                            get_channel_index()));
        }
        m_enmux_semaphore.signal();
        VP_INFO(vp_utils::string_format("[ffio/ff_des][%d][encode_run] encode thread exits.", 
                                        get_channel_index()));
    }

    bool ff_des::
    open(const std::string& uri,
         int width, int height, int fps, int bitrate, int max_b_frames,
         const std::string& encoder_name, 
         AVPixelFormat sw_pix_fmt,
         AVHWDeviceType hw_type,
         AVPixelFormat hw_pix_fmt) {
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
            VP_ERROR(vp_utils::string_format("[ffio/ff_des][%d][open] uri invalid! uri MUST start with `rtsp/rtmp/..`(network streams) or end with `mp4/mkv/..`(file streams).", 
                                             get_channel_index()));
            return false;
        }

        return inner_open(uri, width, height, fps, bitrate, max_b_frames, encoder_name, sw_pix_fmt, hw_type, hw_pix_fmt);
    }

    void ff_des::close() {
        inner_close();
    }

    bool ff_des::is_opened() const {
        return m_encode_running && m_enmux_running;
    }

    void ff_des::inner_close() {
        /* set running flags to false */
        m_encode_running = false;
        m_enmux_running = false;

        inner_exit_signal();
        /* waiting for threads exist */
        if (m_encode_th != nullptr && m_encode_th->joinable()) {
            m_encode_th->join();
        }
        if (m_enmux_th != nullptr && m_enmux_th->joinable()) {
            m_enmux_th->join();
        }

        /* free format & codec context */
        if (m_ofmt_ctx) {
            if (m_ofmt_ctx->pb) {
                avio_closep(&m_ofmt_ctx->pb);
            }
            avformat_free_context(m_ofmt_ctx);
        }
        if (m_enc_ctx) {
            avcodec_free_context(&m_enc_ctx);
        }

        /* clear queue */
        {
            // clear enmux_packets_q
            std::lock_guard<std::mutex> g(m_enmux_packets_m);
            m_enmux_packets_q = {};
            m_enmux_semaphore.reset();
        }
        {
            // clear encode_frames_q
            std::lock_guard<std::mutex> g(m_encode_frames_m);
            m_encode_frames_q = {};
            m_encode_semaphore.reset();
        }    
        VP_INFO(vp_utils::string_format("[ffio/ff_des][%d][inner_close] close successfully.", 
                                        get_channel_index()));
    }

    std::string ff_des::print_summary() {
        return "";
    }

    bool ff_des::
    inner_open(const std::string& uri,
               int width, int height, int fps, int bitrate, int max_b_frames,
               const std::string& encoder_name, 
               AVPixelFormat sw_pix_fmt,
               AVHWDeviceType hw_type,
               AVPixelFormat hw_pix_fmt) {
        auto ret = 0;
        /* allocate the output media context */
        ret = avformat_alloc_output_context2(&m_ofmt_ctx, NULL, NULL, uri.c_str());
        if (ret < 0) {
            VP_WARN(vp_utils::string_format("[ffio/ff_des][%d][inner_open] could not deduce output format from uri, try to use FLV.", 
                                             get_channel_index()));
            ret = avformat_alloc_output_context2(&m_ofmt_ctx, NULL, "flv", uri.c_str());
        }
        if (ret < 0) {
            VP_ERROR(vp_utils::string_format("[ffio/ff_des][%d][inner_open] could not deduce output format from uri(used FLV).", 
                                             get_channel_index()));
            return false;
        }
        auto ofmt = m_ofmt_ctx->oformat;   
        
        /* find encoder for the output stream */
        const AVCodec* enc = nullptr;
        if (encoder_name.empty()) {
            enc = avcodec_find_encoder(ofmt->video_codec);             // get default encoder by AVCodecID
        } else {
            enc = avcodec_find_encoder_by_name(encoder_name.c_str());  // get by encoder name
        }
        if (!enc) {
            VP_ERROR(vp_utils::string_format("[ffio/ff_des][%d][inner_open] could not find proper encoder for output stream.", 
                                             get_channel_index()));
            avformat_free_context(m_ofmt_ctx);
            m_ofmt_ctx = NULL;
            return false;
        }
        /* allocate a codec context for the encoder */
        m_enc_ctx = avcodec_alloc_context3(enc);
        if (!m_enc_ctx) {
            VP_ERROR(vp_utils::string_format("[ffio/ff_des][%d][inner_open] could not allocate context for encoder(%s).", 
                                             get_channel_index(), enc->name));
            avformat_free_context(m_ofmt_ctx);
            m_ofmt_ctx = NULL;
            return false;
        }
        /* set parameters for encoder */
        m_enc_ctx->framerate = av_make_q(fps,1);
        m_enc_ctx->time_base = av_inv_q(m_enc_ctx->framerate);
        m_enc_ctx->pix_fmt   = sw_pix_fmt;
        m_enc_ctx->width     = width;
        m_enc_ctx->height    = height;
        m_enc_ctx->bit_rate  = bitrate * 1024;
        m_enc_ctx->max_b_frames = max_b_frames;
        /* check if need init hwaccels context for encoder */
        if (hw_type != AVHWDeviceType::AV_HWDEVICE_TYPE_NONE) {
            m_enc_ctx->pix_fmt   = hw_pix_fmt;
            if ((ret = hw_encoder_init(m_enc_ctx, hw_type, sw_pix_fmt)) < 0) {
                VP_ERROR(vp_utils::string_format("[ffio/ff_des][%d][inner_open] hw_encoder_init failed. ret: %d", 
                                                 get_channel_index(), ret));
                avcodec_free_context(&m_enc_ctx);
                avformat_free_context(m_ofmt_ctx);
                m_ofmt_ctx = NULL;
                return false;
            }
        }
        /* open the encoder */
        if ((ret = avcodec_open2(m_enc_ctx, enc, NULL)) < 0) {
            VP_ERROR(vp_utils::string_format("[ffio/ff_des][%d][inner_open] could not open encoder. ret: %d", 
                                             get_channel_index(), ret));
            avcodec_free_context(&m_enc_ctx);
            avformat_free_context(m_ofmt_ctx);
            m_ofmt_ctx = NULL;
            return false;
        }
        /* create new stream for output */
        auto out_stream = avformat_new_stream(m_ofmt_ctx, NULL);
        if (!out_stream) {
            VP_ERROR(vp_utils::string_format("[ffio/ff_des][%d][inner_open] could not create new stream for output uri.", 
                                             get_channel_index()));
            avcodec_free_context(&m_enc_ctx);
            avformat_free_context(m_ofmt_ctx);
            m_ofmt_ctx = NULL;
            return false;
        }
        m_inner_stream_index = out_stream->index;
        out_stream->time_base = m_enc_ctx->time_base;
        ret = avcodec_parameters_from_context(out_stream->codecpar, m_enc_ctx);
        if (ret < 0) {
            VP_ERROR(vp_utils::string_format("[ffio/ff_des][%d][inner_open] failed to copy codec parameters from encode context. ret: %d", 
                                             get_channel_index(), ret));
            avcodec_free_context(&m_enc_ctx);
            avformat_free_context(m_ofmt_ctx);
            m_ofmt_ctx = NULL;
            return false;
        }
        if (!(m_ofmt_ctx->flags & AVFMT_NOFILE)) {
            ret = avio_open(&m_ofmt_ctx->pb, uri.c_str(), AVIO_FLAG_WRITE);
            if (ret < 0) {
                VP_ERROR(vp_utils::string_format("[ffio/ff_des][%d][inner_open] avio_open failed on output uri(%s). ret: %d", 
                                                 get_channel_index(), uri.c_str(), ret));
                avcodec_free_context(&m_enc_ctx);
                avformat_free_context(m_ofmt_ctx);
                m_ofmt_ctx = NULL;
                return false;
            }
        }

        /* write the stream header */
        if ((ret = avformat_write_header(m_ofmt_ctx, NULL)) < 0) {
            VP_ERROR(vp_utils::string_format("[ffio/ff_des][%d][inner_open] avformat_write_header failed. ret: %d", 
                                             get_channel_index(), ret));
            avcodec_free_context(&m_enc_ctx);
            avformat_free_context(m_ofmt_ctx);
            m_ofmt_ctx = NULL;
            return false;
        }
        
        /* initialize video & ff_des properties */
        m_hw_type_name = hw_type != AV_HWDEVICE_TYPE_NONE ? std::string(av_hwdevice_get_type_name(hw_type)) : "none";
        m_encoder_name = encoder_name.empty() ? std::string(m_enc_ctx->codec->name) : encoder_name;
        m_uri = uri;
        m_fps = fps;
        m_width = width;
        m_height = height;
        m_bitrate = bitrate;
        m_codec_name = std::string(avcodec_get_name(out_stream->codecpar->codec_id));;
        m_pixel_format = std::string(av_get_pix_fmt_name(static_cast<AVPixelFormat>(out_stream->codecpar->format)));
        m_max_b_frames = max_b_frames;

        /* collect summary notify caller */
        auto summary = print_summary();

        /* go! */
        m_enmux_running = true;
        m_encode_running = true;
        m_enmux_th = std::make_shared<std::thread>(&ff_des::enmux_run, this);
        m_encode_th = std::make_shared<std::thread>(&ff_des::encode_run, this);
        VP_INFO(vp_utils::string_format("[ffio/ff_des][%d][inner_open] open successfully.", 
                                        get_channel_index()));
        return true;   
    }

    bool ff_des::write(const ff_av_frame_ptr& frame) {
        if (!is_opened()) {
            return false;
        }
        
        bool notify = true;
        {
            std::lock_guard<std::mutex> g(m_encode_frames_m);
            m_encode_frames_q.push(frame);
            if (m_encode_frames_q.size() > m_encode_frames_q_max_size) {
                m_encode_frames_q.pop();
                notify = false;
                VP_WARN(vp_utils::string_format("[ffio/ff_des][%d][write] exceed m_encode_frames_q_max_size(%d), discard the front in queue.", 
                                                get_channel_index(), m_encode_frames_q_max_size));
            }
        }
        if (notify) {
            m_encode_semaphore.signal();
        }
        return true;
    }

    ff_des& ff_des::operator<<(const ff_av_frame_ptr& frame) {
        write(frame);
        return *this;
    }

    void ff_des::inner_exit_signal() {
        {
            // send exit flag to notify encode thread
            std::lock_guard<std::mutex> g(m_encode_frames_m);
            m_encode_frames_q.push(nullptr);
        }
        m_encode_semaphore.signal();
    }

    int ff_des::get_video_fps() const {
        return m_fps;
    }

    int ff_des::get_video_width() const {
        return m_width;
    }

    int ff_des::get_video_height() const {
        return m_height;    
    }

    long ff_des::get_video_bitrate() const {
        return m_bitrate;
    }

    std::string ff_des::get_video_codec_name() const {
        return m_codec_name;
    }

    std::string ff_des::get_video_pixel_format_name() const {
        return m_pixel_format;
    }

    bool ff_des::is_live_stream() const {
        return m_live_stream;
    }

    std::string ff_des::get_hw_type_name() const {
        return m_hw_type_name;
    }

    std::string ff_des::get_encoder_name() const {
        return m_encoder_name;
    }

    std::string ff_des::get_uri() const {
        return m_uri;
    }

    int ff_des::get_channel_index() const {
        return m_channel_index;   
    }

    const AVCodecContext* ff_des::get_encode_ctx() const {
        return m_enc_ctx;
    }

    int ff_des::
    hw_encoder_init(AVCodecContext* enc_ctx, 
                    const enum AVHWDeviceType hw_type, 
                    const enum AVPixelFormat sw_pix_fmt) {
        AVBufferRef* hw_device_ctx = nullptr;
        AVBufferRef* hw_frames_ref = nullptr;
        AVHWFramesContext* frames_ctx = nullptr;
        int err = 0;
        if ((err = av_hwdevice_ctx_create(&hw_device_ctx, hw_type,
                                        NULL, NULL, 0)) < 0) {
            VP_ERROR(vp_utils::string_format("[ffio/ff_des][%d][hw_encoder_init] failed to create specified(%s) HW device context for encoder.", 
                                             get_channel_index(), av_hwdevice_get_type_name(hw_type)));
            return err;
        }

        if (!(hw_frames_ref = av_hwframe_ctx_alloc(hw_device_ctx))) {
            VP_ERROR(vp_utils::string_format("[ffio/ff_des][%d][hw_encoder_init] failed to create specified(%s) HW frame context for encoder.", 
                                             get_channel_index(), av_hwdevice_get_type_name(hw_type)));
            return err;
        }
        frames_ctx = (AVHWFramesContext* )(hw_frames_ref->data);
        frames_ctx->format    = enc_ctx->pix_fmt;
        frames_ctx->sw_format = sw_pix_fmt;
        frames_ctx->width     = enc_ctx->width;
        frames_ctx->height    = enc_ctx->height;
        frames_ctx->initial_pool_size = 20;
        if ((err = av_hwframe_ctx_init(hw_frames_ref)) < 0) {
            VP_ERROR(vp_utils::string_format("[ffio/ff_des][%d][hw_encoder_init] failed to initialize specified(%s) HW frame context for encoder.", 
                                             get_channel_index(), av_hwdevice_get_type_name(hw_type)));
            av_buffer_unref(&hw_frames_ref);
            av_buffer_unref(&hw_device_ctx);
            return err;
        }
        enc_ctx->hw_frames_ctx = av_buffer_ref(hw_frames_ref);
        if (!enc_ctx->hw_frames_ctx)
            err = AVERROR(ENOMEM);

        av_buffer_unref(&hw_frames_ref);
        av_buffer_unref(&hw_device_ctx);
        return err;
    }
}
#endif