#pragma once

#include <map>
#include "../vp_node.h"
#include "../../objects/shapes/vp_point.h"
#include "../../objects/shapes/vp_line.h"
#include "../../objects/vp_image_record_control_meta.h"
#include "../../objects/vp_video_record_control_meta.h"

namespace vp_nodes {
    // behaviour analysis node for stop (support multi channels)
    class vp_ba_jam_node: public vp_node 
    {
    private:
        // channel -> vertexs of region, 1 channel supports only 1 region at most (can be 0, which means no jam check on this channel)
        std::map<int, std::vector<vp_objects::vp_point>> all_jam_regions;

        // channel -> status of targets (id -> num of hit frames)
        std::map<int, std::map<int, int>> all_stop_checking_status;

        // channel -> jam status of channel (jam or not)
        std::map<int, bool> all_jam_results;

        // channel -> last frame index to notify jam
        std::map<int, int> all_last_notifys;

        // record params
        bool need_record_image;
        bool need_record_video;

        // check if point inside of polygon
        bool point_in_poly(vp_objects::vp_point p, std::vector<vp_objects::vp_point> region);

        // jam checking logic parameters which may be configed by constructor passed in by user
        const int check_interval_frames = 20;
        const int check_min_hit_frames = 25 * 2;  // 25 fps * 2 seconds
        const int check_max_distance = 8;
        const int check_min_stops = 8;
        const int check_notify_interval = 10;  // interval time (seconds) to notify (ensure not frequently)
    protected:
        virtual std::shared_ptr<vp_objects::vp_meta> handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) override;
    public:
        vp_ba_jam_node(std::string node_name, 
                            std::map<int, std::vector<vp_objects::vp_point>> jam_regions,
                            bool need_record_image = true,
                            bool need_record_video = true);
        ~vp_ba_jam_node();
        std::string to_string() override;
    };
}