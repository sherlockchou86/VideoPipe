

#include "vp_ba_jam_node.h"

namespace vp_nodes {
    
    vp_ba_jam_node::vp_ba_jam_node(std::string node_name, 
                                    std::map<int, std::vector<vp_objects::vp_point>> jam_regions,
                                    bool need_record_image,
                                    bool need_record_video):
                                    vp_node(node_name), all_jam_regions(jam_regions), need_record_image(need_record_image), need_record_video(need_record_video) {
        VP_INFO(vp_utils::string_format("[%s] %s", node_name.c_str(), to_string().c_str()));
        this->initialized();
    }
    
    vp_ba_jam_node::~vp_ba_jam_node() {
        deinitialized();
    }

    std::string vp_ba_jam_node::to_string() {
        /*
        * return vertexs of all jam regions
        * [channel0: x1,y1 x2,y2 ...][channel1: x1,y1 x2,y2 ...]...
        */
        std::stringstream ss;
        for(auto& r: all_jam_regions) {
            ss << "[" << r.first << ":";
            for(auto& p: r.second) {
                ss << " " << p.x << "," << p.y;
            }
            ss << "]";
        }
        return ss.str();
    }

    bool vp_ba_jam_node::point_in_poly(vp_objects::vp_point p, std::vector<vp_objects::vp_point> region) {
        int i, j, c = 0;
        int nvert = region.size();

        for (i = 0, j = nvert-1; i < nvert; j = i++) {
            if (((region[i].y > p.y) != (region[j].y > p.y)) &&
            (p.x < (region[j].x - region[i].x) * (p.y - region[i].y) 
                / (region[j].y - region[i].y) + region[i].x))
            c = !c;
        }
        return c;
    }

    std::shared_ptr<vp_objects::vp_meta> vp_ba_jam_node::handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) {
        // if need applied on current channel or not
        if (all_jam_regions.count(meta->channel_index) == 0) {
            return meta;
        }
        
        // for current channel
        auto& jam_region = all_jam_regions[meta->channel_index];
        auto& stop_checking_status = all_stop_checking_status[meta->channel_index];
        auto& jam_result = all_jam_results[meta->channel_index];
        auto& last_notify = all_last_notifys[meta->channel_index];

        // for vp_frame_target only
        std::vector<int> hit_traget_ids;
        for (auto& target : meta->targets) {
            auto len = target->tracks.size();
            auto loc = target->get_rect().track_point();

            // target has been tracked AND tracked enough frames
            if (len < check_interval_frames || target->track_id < 0) {
                continue;
            }
            // if target inside of stop region or not
            if (!point_in_poly(loc, jam_region)) {
                continue;
            }

            auto pre_loc = target->tracks[len - check_interval_frames].track_point();
            if (pre_loc.distance_with(loc) <= check_max_distance) {
                stop_checking_status[target->track_id]++;
                hit_traget_ids.push_back(target->track_id);
            }
        }

        // count for stop targets
        auto stops = 0;
        std::vector<int> involve_targets;
        for (auto i = stop_checking_status.begin(); i != stop_checking_status.end();) {
            if (std::find(hit_traget_ids.begin(), hit_traget_ids.end(), i->first) == hit_traget_ids.end()) {
                // remove since it not satisfy stop condition
                i = stop_checking_status.erase(i);
                continue;
            }
            if (i->second >= check_min_hit_frames) {
                involve_targets.push_back(i->first);
                stops++;
            }
            i++;
        }
            
        // enter jam status
        if (stops >= check_min_stops && !jam_result && (meta->frame_index - last_notify) > (check_notify_interval * meta->fps)) {
            jam_result = true;
            last_notify = meta->frame_index;

            // send record image and record video signal, recording actions would occur if record nodes exist in pipeline
            std::string image_file_name_without_ext = "";    // empty means no recording image
            std::string video_file_name_without_ext = "";    // empty means no recording video

            // send image record control meta
            if (need_record_image) {
                image_file_name_without_ext = vp_utils::time_format(NOW, "jam_image__<year><mon><day><hour><min><sec><mili>");
                auto image_record_control_meta = std::make_shared<vp_objects::vp_image_record_control_meta>(meta->channel_index, image_file_name_without_ext, true);
                pendding_meta(image_record_control_meta);
            }
            // send video record control meta
            if (need_record_video) {
                video_file_name_without_ext = vp_utils::time_format(NOW, "jam_video__<year><mon><day><hour><min><sec><mili>");        
                auto video_record_control_meta = std::make_shared<vp_objects::vp_video_record_control_meta>(meta->channel_index, video_file_name_without_ext);
                pendding_meta(video_record_control_meta);
            }

            std::vector<vp_objects::vp_point> involve_region = jam_region;
            auto ba_result = std::make_shared<vp_objects::vp_ba_result>(vp_objects::vp_ba_type::JAM, 
                                                                        meta->channel_index, 
                                                                        meta->frame_index, 
                                                                        involve_targets, 
                                                                        involve_region,
                                                                        "jam",   // meaningful label
                                                                        image_file_name_without_ext, 
                                                                        video_file_name_without_ext);
            // fill back to frame meta
            meta->ba_results.push_back(ba_result);
            // info log
            VP_INFO(vp_utils::string_format("[%s] [channel %d] has found jam", node_name.c_str(), meta->channel_index));
            if (need_record_image || need_record_video) {
                VP_INFO(vp_utils::string_format("[%s] [channel %d] image & video record file names are: [%s & %s]", node_name.c_str(), meta->channel_index, image_file_name_without_ext.c_str(), video_file_name_without_ext.c_str()));
            }
        }
        
        // leave jam status
        if (stops < check_min_stops && jam_result) {
            jam_result = false;
            last_notify = meta->frame_index;

            // send record image and record video signal, recording actions would occur if record nodes exist in pipeline
            std::string image_file_name_without_ext = "";    // empty means no recording image
            std::string video_file_name_without_ext = "";    // empty means no recording video

            // send image record control meta
            if (need_record_image) {
                image_file_name_without_ext = vp_utils::time_format(NOW, "unjam_image__<year><mon><day><hour><min><sec><mili>");
                auto image_record_control_meta = std::make_shared<vp_objects::vp_image_record_control_meta>(meta->channel_index, image_file_name_without_ext, true);
                pendding_meta(image_record_control_meta);
            }
            // send video record control meta
            if (need_record_video) {
                video_file_name_without_ext = vp_utils::time_format(NOW, "unjam_video__<year><mon><day><hour><min><sec><mili>");        
                auto video_record_control_meta = std::make_shared<vp_objects::vp_video_record_control_meta>(meta->channel_index, video_file_name_without_ext);
                pendding_meta(video_record_control_meta);
            }

            std::vector<vp_objects::vp_point> involve_region = jam_region;
            auto ba_result = std::make_shared<vp_objects::vp_ba_result>(vp_objects::vp_ba_type::UNJAM, 
                                                                        meta->channel_index, 
                                                                        meta->frame_index, 
                                                                        involve_targets, 
                                                                        involve_region,
                                                                        "unjam",   // meaningful label
                                                                        image_file_name_without_ext, 
                                                                        video_file_name_without_ext);
            // fill back to frame meta
            meta->ba_results.push_back(ba_result);
            // info log
            VP_INFO(vp_utils::string_format("[%s] [channel %d] has found unjam", node_name.c_str(), meta->channel_index));
            if (need_record_image || need_record_video) {
                VP_INFO(vp_utils::string_format("[%s] [channel %d] image & video record file names are: [%s & %s]", node_name.c_str(), meta->channel_index, image_file_name_without_ext.c_str(), video_file_name_without_ext.c_str()));
            }
        }
        return meta;
    }
}