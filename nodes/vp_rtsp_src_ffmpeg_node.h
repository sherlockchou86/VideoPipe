#pragma once

#include <string>
#include <memory>
#include <atomic>
#include <thread>
#include <mutex>
#include <opencv2/opencv.hpp>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
#include <libavutil/time.h>
#include <libavutil/opt.h>
}

#include "vp_src_node.h"
#include "../utils/vp_utils.h"

namespace vp_nodes {
    // rtsp source node, receive video stream via rtsp protocal.
    // example:
    // rtsp://admin:admin12345@192.168.77.110:554/
    class vp_rtsp_src_node: public vp_src_node {
    private:
        /* FFmpeg相关变量 */
        AVFormatContext* format_context_ = nullptr;
        AVCodecContext* codec_context_ = nullptr;
        AVFrame* frame_ = nullptr;
        AVFrame* frame_rgb_ = nullptr;
        SwsContext* sws_context_ = nullptr;
        uint8_t* buffer_ = nullptr;
        int video_stream_index_ = -1;
        
        std::string rtsp_url_;
        int target_width_ = 1280;
        int target_height_ = 720;
        int buffer_size_ = 10 * 1024 * 1024;  // 10MB缓冲区
        int timeout_ = 5000000;               // 5秒超时
        
        std::atomic<bool> is_running_{false};
        std::atomic<bool> is_initialized_{false};
        std::thread ffmpeg_thread_;
        std::mutex frame_mutex_;
        cv::Mat current_frame_;
        
        int reconnect_attempts_ = 0;
        const int max_reconnect_attempts_ = 5;

        // 状态标志，标记是否成功拉取视频流
        std::atomic<bool> is_stream_initialized{false};

    protected:
        virtual void handle_run() override;

    public:
        vp_rtsp_src_node(std::string node_name, 
                        int channel_index, 
                        std::string rtsp_url, 
                        float resize_ratio = 1.0,
                        std::string gst_decoder_name = "avdec_h264",
                        int skip_interval = 0);
        ~vp_rtsp_src_node();

        virtual std::string to_string() override;

        std::string rtsp_url;
        std::string gst_decoder_name = "avdec_h264";
        int skip_interval = 0;

        // 获取状态标志的方法
        bool is_initialized() const { return is_stream_initialized.load(); }
        
    private:
        // FFmpeg相关方法
        bool init_ffmpeg(bool use_tcp = true);
        bool reconnect_ffmpeg();
        void process_ffmpeg_stream();
        void cleanup_ffmpeg();
        cv::Mat get_current_frame();
    };
}