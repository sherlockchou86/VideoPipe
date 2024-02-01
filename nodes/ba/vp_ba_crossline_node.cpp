

#include "vp_ba_crossline_node.h"

namespace vp_nodes {
    
    vp_ba_crossline_node::vp_ba_crossline_node(std::string node_name, 
                                                std::map<int, vp_objects::vp_line> lines,
                                                bool need_record_image,
                                                bool need_record_video):
                                                vp_node(node_name), all_lines(lines), need_record_image(need_record_image), need_record_video(need_record_video) {
        VP_INFO(vp_utils::string_format("[%s] %s", node_name.c_str(), to_string().c_str()));
        this->initialized();
    }
    
    vp_ba_crossline_node::~vp_ba_crossline_node() {

    }
    
    std::string vp_ba_crossline_node::to_string() {
        /*
        * return 2 points of all lines
        * [channel 0: x1,y1 x2,y2][channel 1: x1,y1 x2,y2]...
        */
        std::stringstream ss;
        for(auto& p: all_lines) {
            ss << "[channel" << p.first << ": " << p.second.start.x << "," << p.second.start.y << " " << p.second.end.x << "," << p.second.end.y << "]"; 
        }
        return ss.str();
    }

    std::shared_ptr<vp_objects::vp_meta> vp_ba_crossline_node::handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) {
        // if need applied on current channel or not
        if (all_lines.count(meta->channel_index) == 0) {
            return meta;
        }

        // for current channel
        auto& total_crossline = all_total_crossline[meta->channel_index];
        auto& line = all_lines[meta->channel_index];

        // for vp_frame_target only
        for (auto& target : meta->targets) {
            auto len = target->tracks.size();
            std::vector<int> involve_targets;
            if (len > 1 && target->track_id >= 0) {
                // check the last 2 points in tracks
                auto p1 = target->tracks[len - 1].track_point();
                auto p2 = target->tracks[len - 2].track_point();

                auto check1 = at_1_side_of_line(p1, line);
                auto check2 = at_1_side_of_line(p2, line);

                // `true and false`  or `false and true`
                // means target passed the line in current frame
                if (check1 ^ check2) {
                    total_crossline++;
                    involve_targets.push_back(target->track_id);

                    // not empty need fill ba result back to frame meta
                    if (involve_targets.size() > 0) {
                        // send record image and record video signal, recording actions would occur if record nodes exist in pipeline
                        std::string image_file_name_without_ext = "";    // empty means no recording image
                        std::string video_file_name_without_ext = "";    // empty means no recording video

                        // send image record control meta
                        if (need_record_image) {
                            image_file_name_without_ext = vp_utils::time_format(NOW, "crossline_image__<year><mon><day><hour><min><sec><mili>");
                            auto image_record_control_meta = std::make_shared<vp_objects::vp_image_record_control_meta>(meta->channel_index, image_file_name_without_ext, true);
                            pendding_meta(image_record_control_meta);
                        }
                        // send video record control meta
                        if (need_record_video) {
                            video_file_name_without_ext = vp_utils::time_format(NOW, "crossline_video__<year><mon><day><hour><min><sec><mili>");        
                            auto video_record_control_meta = std::make_shared<vp_objects::vp_video_record_control_meta>(meta->channel_index, video_file_name_without_ext);
                            pendding_meta(video_record_control_meta);
                        }

                        std::vector<vp_objects::vp_point> involve_region {line.start, line.end};
                        auto ba_result = std::make_shared<vp_objects::vp_ba_result>(vp_objects::vp_ba_type::CROSSLINE, 
                                                                                    meta->channel_index, 
                                                                                    meta->frame_index, 
                                                                                    involve_targets, 
                                                                                    involve_region,
                                                                                    "cross line",   // meaningful label
                                                                                    image_file_name_without_ext, 
                                                                                    video_file_name_without_ext);
                        // fill back to frame meta
                        meta->ba_results.push_back(ba_result);
                        // info log
                        VP_INFO(vp_utils::string_format("[%s] [channel %d] has found target cross line, total number of crossline: [%d]", node_name.c_str(), meta->channel_index, total_crossline));
                        if (need_record_image || need_record_video) {
                            VP_INFO(vp_utils::string_format("[%s] [channel %d] image & video record file names are: [%s & %s]", node_name.c_str(), meta->channel_index, image_file_name_without_ext.c_str(), video_file_name_without_ext.c_str()));
                        }
                    }
                }
            }
        }

        return meta;
    }

    bool vp_ba_crossline_node::at_1_side_of_line(vp_objects::vp_point p, vp_objects::vp_line line) {
        auto p1 = line.start;
        auto p2 = line.end;

        if (p1.x == p2.x) {
            return p.x < p1.x;
        }

        if (p1.y == p2.y) {
            return p.y < p1.y;
        }

        if (p2.x < p1.x) {
            auto tmp = p2;
            p2 = p1;
            p1 = tmp;
        }

        int ret = (p2.y - p.y) * (p2.x - p1.x) - (p2.y - p1.y) * (p2.x - p.x);
        return ret < 0;
    }
}