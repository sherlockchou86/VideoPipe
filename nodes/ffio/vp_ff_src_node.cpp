#ifdef VP_WITH_FFMPEG
#include "vp_ff_src_node.h"

namespace vp_nodes {
    
    vp_ff_src_node::
    vp_ff_src_node(const std::string& node_name, 
                          int channel_index,
                          const std::string& uri,
                          const std::string& decoder_name,
                          float resize_ratio,
                          int skip_interval):
                          vp_src_node(node_name, channel_index, resize_ratio),
                          m_uri(uri),
                          m_decoder_name(decoder_name),
                          m_skip_interval(skip_interval) {
        assert(skip_interval >= 0 && skip_interval <= 9);
        m_ff_src = alloc_ff_src(channel_index);
        VP_INFO(vp_utils::string_format("[%s] [%s]", node_name.c_str(), uri.c_str()));
        this->initialized();
    }
    
    vp_ff_src_node::~vp_ff_src_node() {
        deinitialized();
    }

    void vp_ff_src_node::handle_run() {
        SwsContext* sws_ctx = NULL;
        auto free_sws_ctx = [&]() {
            /* free swsContext. */
            if (sws_ctx) {
                sws_freeContext(sws_ctx);
                sws_ctx = NULL;
            }
        };
        auto reopen_wait = 10;
        auto reopen_times = 0;
        ff_av_frame_ptr src_frame;
        int video_width = 0;
        int video_height = 0;
        int fps = 0;
        int skip = 0;
        while (alive) {
            /* wait for data coming. */
            gate.knock();
            if (!m_ff_src->is_opened()) {
                if(!m_ff_src->open(m_uri, m_decoder_name)) {
                    reopen_times++;
                    if (reopen_times < 5) {
                        VP_WARN(vp_utils::string_format("[%s] open uri failed, try again right now...", node_name.c_str()));
                    }
                    else {
                        VP_WARN(vp_utils::string_format("[%s] open uri failed too many times, wait for %d seconds then try again...", node_name.c_str(), reopen_wait));
                        std::this_thread::sleep_for(std::chrono::milliseconds(1000 * reopen_wait));
                        reopen_times = 0;
                    }
                    continue;
                }
                free_sws_ctx();
            }

            // video properties
            if (video_width == 0 || video_height == 0 || fps == 0) {
                video_width = m_ff_src->get_video_width();
                video_height = m_ff_src->get_video_height();
                fps = m_ff_src->get_video_fps();
                
                original_fps = fps;
                original_width = video_width;
                original_height = video_height;

                // set true fps because skip some frames
                fps = fps / (m_skip_interval + 1);
            }
            // stream_info_hooker activated if need
            vp_stream_info stream_info {channel_index, original_fps, original_width, original_height, to_string()};
            invoke_stream_info_hooker(node_name, stream_info);

            /* try to read next frame from ff_src. */
            if (!m_ff_src->read(src_frame)) {
                //VP_WARN(vp_utils::string_format("[%s] reading frame failed, total frame==>%d", node_name.c_str(), frame_index));
                continue;
            }

            // need skip
            if (skip < m_skip_interval) {
                skip++;
                continue;
            }
            skip = 0;

            /* AVFrame -> cv::Mat. */
            /* resize and convert to BGR24. */
            auto n_width = int(src_frame->width * resize_ratio);
            auto n_height = int(src_frame->height * resize_ratio);
            if (!sws_ctx) {
                sws_ctx = sws_getContext(src_frame->width, 
                                         src_frame->height, 
                                         AVPixelFormat(src_frame->format), 
                                         n_width, n_height, 
                                         AV_PIX_FMT_BGR24, 
                                         0, NULL, NULL, NULL);
            }
            if (!sws_ctx) {
                VP_WARN(vp_utils::string_format("[%s] could not initialize sws_ctx.", node_name.c_str()));
                continue;
            }
            auto buffer_size = n_width * n_height * 3;
            uchar* bgr24 = new uchar[buffer_size];
            int linesize[1] = {n_width * 3};
            sws_scale(sws_ctx, src_frame->data, src_frame->linesize, 0, src_frame->height, &bgr24, linesize);

            cv::Mat frame(n_height, n_width, CV_8UC3, bgr24);
            auto c_frame = frame.clone();
            delete[] bgr24;
           // set true size because resize
            video_width = c_frame.cols;
            video_height = c_frame.rows;
            
            this->frame_index++;
            // create frame meta
            auto out_meta = 
                std::make_shared<vp_objects::vp_frame_meta>(c_frame, this->frame_index, this->channel_index, video_width, video_height, fps);

            if (out_meta != nullptr) {
                this->out_queue.push(out_meta);
                
                // handled hooker activated if need
                if (this->meta_handled_hooker) {
                    meta_handled_hooker(node_name, out_queue.size(), out_meta);
                }

                // important! notify consumer of out_queue in case it is waiting.
                this->out_queue_semaphore.signal();
                VP_DEBUG(vp_utils::string_format("[%s] after handling meta, out_queue.size()==>%d", node_name.c_str(), out_queue.size()));
            } 
        }
        
        free_sws_ctx();
        // send dead flag for dispatch_thread
        this->out_queue.push(nullptr);
        this->out_queue_semaphore.signal();    
    }

    std::string vp_ff_src_node::to_string() {
        return m_uri;
    }
}
#endif