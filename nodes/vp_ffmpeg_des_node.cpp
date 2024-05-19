//
// Created by lijin on 2023/8/2.
//
#ifdef VP_WITH_FFMPEG
#include "vp_ffmpeg_des_node.h"

namespace vp_nodes {

    bool vp_ffmpeg_des_node::open(const string &url, bool useHwEncode) {
        if (m_from_width == 0 || m_to_width == 0) {
            std::cout << "from_width or to_width is 0";
            return false;
        }
        if (!m_scaler) {
            m_scaler = vp::Scaler::createShare(m_from_width, m_from_height, (AVPixelFormat) m_from_format, m_to_width,
                                               m_to_height, (AVPixelFormat) m_to_format);
        }
        if (!m_encoder) {
            m_encoder = vp::Encoder::createShared(m_to_width, m_to_height, m_fps, m_bitrate);
            if (!m_encoder->open(useHwEncode)) {
                std::cout << "encoder open failed" << std::endl;
                return false;
            }
        }
        if (!m_enmux) {
            m_enmux = vp::Enmux::createShared(m_encoder, url);
            if (!m_enmux->open()) {
                std::cout << "mux open failed" << std::endl;
                return false;
            }
        }
        return true;
    }

    void vp_ffmpeg_des_node::set_bitrate(int bitrate) {
        m_bitrate = bitrate;
    }

    void vp_ffmpeg_des_node::set_fps(int fps) {
        m_fps = fps;

    }

    void vp_ffmpeg_des_node::set_tomat(int width, int height, int format) {
        m_to_width = width;
        m_to_height = height;
        m_to_format = format;
    }

    void vp_ffmpeg_des_node::set_frommat(int width, int height, int format) {
        m_from_width = width;
        m_from_height = height;
        m_from_format = format;

    }

    vp_ffmpeg_des_node::vp_ffmpeg_des_node(std::string node_name,
                                           int channel_index,
                                           std::string url,
                                           int from_width,
                                           int from_height,
                                           int from_format,
                                           int to_width,
                                           int to_height,
                                           int to_format,
                                           int bitrate,
                                           int fps,
                                           bool use_hw)
            : vp_des_node(node_name, channel_index) {
        set_frommat(from_width, from_height, from_format);
        set_tomat(to_width, to_height, to_format);
        set_bitrate(bitrate);
        set_fps(fps);
        assert(open(url, use_hw));
        initialized();
    }

    std::shared_ptr<vp_objects::vp_meta>
    vp_ffmpeg_des_node::handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) {
        auto frame = alloc_av_frame();
        if(meta->osd_frame.empty()){
            if (!m_scaler->scale<cv::Mat, av_frame >(meta->frame, frame)) {
                std::cout << "scale failed" << std::endl;
                return nullptr;
            }
        } else {
            if (!m_scaler->scale<cv::Mat, av_frame >(meta->osd_frame, frame)) {
                std::cout << "scale failed" << std::endl;
                return nullptr;
            }
        }
        auto pkt = alloc_av_packet();
        if (!m_encoder->encode(frame, pkt)) {
            std::cout << "encode failed" << std::endl;
            return nullptr;
        }
        if (!m_enmux->write_packet(pkt)) {
            std::cout << "write packet failed" << std::endl;
            return nullptr;
        }
        return vp_des_node::handle_frame_meta(meta);
    }

} // vp_nodes

#endif //VP_WITH_FFMPEG