#include "vp_rtsp_src_node.h"
#include "../utils/vp_utils.h"
#include <thread>
#include <chrono>
#include <iostream>

namespace vp_nodes {
        
    vp_rtsp_src_node::vp_rtsp_src_node(std::string node_name, 
                                        int channel_index, 
                                        std::string rtsp_url, 
                                        float resize_ratio,
                                        std::string gst_decoder_name,
                                        int skip_interval): 
                                        vp_src_node(node_name, channel_index, resize_ratio),
                                        rtsp_url(rtsp_url), gst_decoder_name(gst_decoder_name), 
                                        skip_interval(skip_interval), rtsp_url_(rtsp_url) {
        assert(skip_interval >= 0 && skip_interval <= 9);
        
        VP_INFO(vp_utils::string_format("[%s] [%s]", node_name.c_str(), rtsp_url.c_str()));
        this->initialized();
    }
    
    vp_rtsp_src_node::~vp_rtsp_src_node() {
        is_running_ = false;
        if (ffmpeg_thread_.joinable()) {
            ffmpeg_thread_.join();
        }
        cleanup_ffmpeg();
        deinitialized();
    }
    
    bool vp_rtsp_src_node::init_ffmpeg(bool use_tcp) {
        cleanup_ffmpeg();
        
        // 初始化FFmpeg
        av_register_all();
        avformat_network_init();
        
        VP_INFO(vp_utils::string_format("[%s] 尝试初始化RTSP流: %s", node_name.c_str(), rtsp_url_.c_str()));
        
        // 打开RTSP流
        format_context_ = avformat_alloc_context();
        AVDictionary *options = nullptr;
        
        // 设置传输协议
        if (use_tcp) {
            av_dict_set(&options, "rtsp_transport", "tcp", 0);
        }
        
        // 设置超时时间
        char timeout_str[20];
        snprintf(timeout_str, sizeof(timeout_str), "%d", timeout_);
        av_dict_set(&options, "stimeout", timeout_str, 0);
        
        // 设置缓冲区大小
        char buffer_size_str[20];
        snprintf(buffer_size_str, sizeof(buffer_size_str), "%d", buffer_size_);
        av_dict_set(&options, "buffer_size", buffer_size_str, 0);
        
        // 打开输入流
        int ret = avformat_open_input(&format_context_, rtsp_url_.c_str(), nullptr, &options);
        av_dict_free(&options);
        
        if (ret != 0) {
            VP_ERROR(vp_utils::string_format("[%s] 无法打开RTSP流", node_name.c_str()));
            return false;
        }
        
        // 获取流信息
        ret = avformat_find_stream_info(format_context_, nullptr);
        if (ret < 0) {
            VP_ERROR(vp_utils::string_format("[%s] 无法获取流信息", node_name.c_str()));
            return false;
        }
        
        // 查找视频流
        video_stream_index_ = -1;
        for (unsigned int i = 0; i < format_context_->nb_streams; i++) {
            if (format_context_->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
                video_stream_index_ = i;
                break;
            }
        }
        
        if (video_stream_index_ == -1) {
            VP_ERROR(vp_utils::string_format("[%s] 未找到视频流", node_name.c_str()));
            return false;
        }
        
        // 获取解码器参数
        AVCodecParameters* codec_params = format_context_->streams[video_stream_index_]->codecpar;
        
        // 获取解码器
        const AVCodec* codec = avcodec_find_decoder(codec_params->codec_id);
        if (!codec) {
            VP_ERROR(vp_utils::string_format("[%s] 无法找到解码器", node_name.c_str()));
            return false;
        }
        
        // 初始化解码器上下文
        codec_context_ = avcodec_alloc_context3(codec);
        if (avcodec_parameters_to_context(codec_context_, codec_params) < 0) {
            VP_ERROR(vp_utils::string_format("[%s] 无法初始化解码器上下文", node_name.c_str()));
            return false;
        }
        
        // 打开解码器
        ret = avcodec_open2(codec_context_, codec, nullptr);
        if (ret < 0) {
            VP_ERROR(vp_utils::string_format("[%s] 无法打开解码器", node_name.c_str()));
            return false;
        }
        
        // 初始化帧
        frame_ = av_frame_alloc();
        frame_rgb_ = av_frame_alloc();
        
        // 分配RGB帧缓冲区
        int num_bytes = av_image_get_buffer_size(AV_PIX_FMT_BGR24, target_width_, target_height_, 1);
        buffer_ = (uint8_t*)av_malloc(num_bytes * sizeof(uint8_t));
        av_image_fill_arrays(frame_rgb_->data, frame_rgb_->linesize, buffer_, 
                            AV_PIX_FMT_BGR24, target_width_, target_height_, 1);
        
        // 初始化图像转换上下文
        sws_context_ = sws_getContext(
            codec_context_->width, codec_context_->height, codec_context_->pix_fmt,
            target_width_, target_height_, AV_PIX_FMT_BGR24,
            SWS_BICUBIC, nullptr, nullptr, nullptr
        );
        
        if (!sws_context_) {
            VP_ERROR(vp_utils::string_format("[%s] 无法初始化图像转换上下文", node_name.c_str()));
            return false;
        }
        
        is_initialized_ = true;
        reconnect_attempts_ = 0;
        VP_INFO(vp_utils::string_format("[%s] RTSP流初始化成功", node_name.c_str()));
        return true;
    }
    
    bool vp_rtsp_src_node::reconnect_ffmpeg() {
        if (reconnect_attempts_ >= max_reconnect_attempts_) {
            VP_ERROR(vp_utils::string_format("[%s] 已达到最大重连次数，停止尝试", node_name.c_str()));
            return false;
        }
        
        reconnect_attempts_++;
        VP_WARN(vp_utils::string_format("[%s] 尝试重连(%d/%d)...", node_name.c_str(), 
                                       reconnect_attempts_, max_reconnect_attempts_));
        
        // 等待一段时间再重连
        std::this_thread::sleep_for(std::chrono::seconds(2));
        
        // 尝试重新初始化
        return init_ffmpeg(true);
    }
    
    void vp_rtsp_src_node::process_ffmpeg_stream() {
        AVPacket packet;
        av_init_packet(&packet);
        
        while (is_running_) {
            int ret = av_read_frame(format_context_, &packet);
            if (ret < 0) {
                VP_WARN(vp_utils::string_format("[%s] 读取帧失败，尝试重连", node_name.c_str()));
                
                // 尝试重连
                if (!reconnect_ffmpeg()) {
                    break;
                }
                continue;
            }
            
            // 重置重连计数器
            reconnect_attempts_ = 0;
            
            if (packet.stream_index == video_stream_index_) {
                // 发送数据包到解码器
                ret = avcodec_send_packet(codec_context_, &packet);
                if (ret != 0) {
                    VP_WARN(vp_utils::string_format("[%s] 发送数据包到解码器失败", node_name.c_str()));
                    av_packet_unref(&packet);
                    continue;
                }
                
                // 接收解码后的帧
                while (avcodec_receive_frame(codec_context_, frame_) == 0) {
                    // 转换为RGB格式
                    sws_scale(sws_context_, frame_->data, frame_->linesize, 0, 
                             codec_context_->height, frame_rgb_->data, frame_rgb_->linesize);
                    
                    // 转换为OpenCV的Mat
                    cv::Mat frame(target_height_, target_width_, CV_8UC3, frame_rgb_->data[0], frame_rgb_->linesize[0]);
                    
                    // 更新当前帧
                    {
                        std::lock_guard<std::mutex> lock(frame_mutex_);
                        current_frame_ = frame.clone();
                    }
                    
                    // 标记流已初始化
                    if (!is_stream_initialized) {
                        is_stream_initialized = true;
                        VP_INFO(vp_utils::string_format("[%s] 流初始化成功", node_name.c_str()));
                    }
                }
            }
            
            av_packet_unref(&packet);
        }
        
        is_running_ = false;
    }
    
    void vp_rtsp_src_node::cleanup_ffmpeg() {
        if (buffer_) {
            av_free(buffer_);
            buffer_ = nullptr;
        }
        if (frame_rgb_) {
            av_frame_free(&frame_rgb_);
            frame_rgb_ = nullptr;
        }
        if (frame_) {
            av_frame_free(&frame_);
            frame_ = nullptr;
        }
        if (codec_context_) {
            avcodec_free_context(&codec_context_);
            codec_context_ = nullptr;
        }
        if (format_context_) {
            avformat_close_input(&format_context_);
            format_context_ = nullptr;
        }
        if (sws_context_) {
            sws_freeContext(sws_context_);
            sws_context_ = nullptr;
        }
        
        is_initialized_ = false;
    }
    
    cv::Mat vp_rtsp_src_node::get_current_frame() {
        std::lock_guard<std::mutex> lock(frame_mutex_);
        return current_frame_.clone();
    }
    
    // 使用FFmpeg实现的RTSP拉流
    void vp_rtsp_src_node::handle_run() {
        int video_width = 0;
        int video_height = 0;
        int fps = 25;  // 默认FPS
        int skip = 0;
        
        // 初始化FFmpeg
        bool init_success = init_ffmpeg(true);
        if (!init_success) {
            VP_WARN(vp_utils::string_format("[%s] TCP初始化失败，尝试UDP传输...", node_name.c_str()));
            init_success = init_ffmpeg(false);
        }
        
        if (!init_success) {
            VP_ERROR(vp_utils::string_format("[%s] 初始化RTSP播放器失败", node_name.c_str()));
            return;
        }
        
        is_running_ = true;
        
        // 启动FFmpeg处理线程
        ffmpeg_thread_ = std::thread(&vp_rtsp_src_node::process_ffmpeg_stream, this);
        
        while(alive) {
            gate.knock();
            
            if (!is_running_ || !is_initialized_) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            
            // 获取当前帧
            cv::Mat frame = get_current_frame();
            
            if(frame.empty()) {
                VP_WARN(vp_utils::string_format("[%s] reading frame empty, total frame==>%d", node_name.c_str(), frame_index));
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }
            
            if (skip < skip_interval) {
                skip++;
                continue;
            }
            skip = 0;

            // 更新视频信息
            if (video_width == 0 || video_height == 0) {
                video_width = frame.cols;
                video_height = frame.rows;
                original_width = video_width;
                original_height = video_height;
                original_fps = fps;
            }
            
            cv::Mat resize_frame;
            if (this->resize_ratio != 1.0f) {                 
                cv::resize(frame, resize_frame, cv::Size(), resize_ratio, resize_ratio);
                video_width = resize_frame.cols;
                video_height = resize_frame.rows;
            }
            else {
                resize_frame = frame.clone(); // clone!
            }
            
            this->frame_index++;
            auto out_meta = 
                std::make_shared<vp_objects::vp_frame_meta>(resize_frame, this->frame_index, this->channel_index, video_width, video_height, fps);

            if (out_meta != nullptr) {
                this->out_queue.push(out_meta);
                if (this->meta_handled_hooker) {
                    meta_handled_hooker(node_name, out_queue.size(), out_meta);
                }
                this->out_queue_semaphore.signal();
                VP_DEBUG(vp_utils::string_format("[%s] after handling meta, out_queue.size()==>%d", node_name.c_str(), out_queue.size()));
            }
            
            // 控制帧率，避免CPU占用过高
            std::this_thread::sleep_for(std::chrono::milliseconds(1000 / fps));
        }

        // 停止处理
        is_running_ = false;
        if (ffmpeg_thread_.joinable()) {
            ffmpeg_thread_.join();
        }
        cleanup_ffmpeg();
        
        this->out_queue.push(nullptr);
        this->out_queue_semaphore.signal();    
    }

    // return stream url
    std::string vp_rtsp_src_node::to_string() {
        return rtsp_url;
    }
}