//
// Created by lijin on 2023/8/2.
//
#ifdef VP_WITH_FFMPEG

#include "vp_ffmpeg_src_node.h"

#include <utility>

namespace vp_nodes {

    void vp_ffmpeg_src_node::handle_run() {
        while (alive) {
            // 阻塞线程直至头节点启动
            gate.knock();

            auto pkt = alloc_av_packet();
            int re = m_demux->read_packet(pkt);
            if (re == EXIT_SUCCESS) {
                if (pkt->stream_index != m_demux->get_video_stream_index()) {
                    continue; // 忽略非视频帧
                }
                m_decoder->send(pkt);
                auto frame = alloc_av_frame();
                if (!m_decoder->receive(frame)) {
                    continue; // 编码器前几帧的缓存可能接收不到
                }
                cv::Mat image(m_demux->get_video_codec_parameters()->height,
                              m_demux->get_video_codec_parameters()->width,
                              CV_8UC3);
                if (!m_scaler->scale<av_frame, cv::Mat>(frame, image)) {
                    std::cout << "scale failed" << std::endl;
                    continue;
                }
                this->frame_index++;
                auto out_meta = std::make_shared<vp_objects::vp_frame_meta>(image, this->frame_index,
                                                                            this->channel_index, frame->width,
                                                                            frame->height);
                if (out_meta != nullptr) {
                    this->out_queue.push(out_meta);
                    this->out_queue_semaphore.signal();
                }
                // 输入是本地视频下的模拟延迟
                std::this_thread::sleep_for(std::chrono::milliseconds(30));
            } else if (re == AVERROR_EOF) {
                std::cout << "read eof" << std::endl;
                stop();
                break;
            } else {
                std::cout << "read error" << std::endl;
                break;

            }
        }

    }

    vp_ffmpeg_src_node::vp_ffmpeg_src_node(std::string node_name, int channel_index, std::string file_ro_url_path,
                                           bool use_hw) : vp_src_node(std::move(node_name), channel_index) {
        assert(open(file_ro_url_path, use_hw));
        initialized();
    }

    bool vp_ffmpeg_src_node::open(const string &url, bool use_hw) {
        if (!m_demux) {
            m_demux = vp::Demux::createShare();
        }
        if (!(m_demux->open(url))) {
            m_demux = nullptr;
            return false;
        }
        if (!m_scaler) {
            if (use_hw) {
                m_scaler = vp::Scaler::createShare(m_demux->get_video_codec_parameters()->width,
                                                   m_demux->get_video_codec_parameters()->height,
                                                   AV_PIX_FMT_NV12,
                                                   m_demux->get_video_codec_parameters()->width,
                                                   m_demux->get_video_codec_parameters()->height,
                                                   AV_PIX_FMT_BGR24);
            } else {
                m_scaler = vp::Scaler::createShare(m_demux->get_video_codec_parameters()->width,
                                                   m_demux->get_video_codec_parameters()->height,
                                                   (AVPixelFormat) m_demux->get_video_codec_parameters()->format,
                                                   m_demux->get_video_codec_parameters()->width,
                                                   m_demux->get_video_codec_parameters()->height,
                                                   AV_PIX_FMT_BGR24);
            }
        }
        if (!m_decoder) {
            m_decoder = vp::Decoder::createShare(m_demux);
        }
        if (!(m_decoder->open(use_hw))) {
            return false;
        }
        return true;
    }


} // vp_nodes
#endif //VP_WITH_FFMPEG