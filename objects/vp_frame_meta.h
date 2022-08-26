#pragma once

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio.hpp>

#include "vp_meta.h"
#include "vp_frame_target.h"
#include "vp_frame_pose_target.h"
#include "vp_frame_face_target.h"
#include "elements/vp_frame_element.h"
#include "../ba/vp_ba_analyser.h"

/*
* ##########################################
* how does frame meta work?
* ##########################################
* frame meta, holding all data(targets/elements/...) of current frame in the video scene. frame meta are independent and don't know about each other, neither its previous frames nor next frames.
* the data in frame meta is just telling us what **current frame** is so we can not get something like 'state-switch' from a single frame meta. 
* if you need know when the 'state-switch' happen, for example, you want to notify to cloud via restful api if state changed(ignore if it's keeping), 
* you need cache previous frame meta(maybe partial data) in your custom node first and then compare with each other to figure out if it has changed.
* 
* frame meta works like our eyes, by taking a glance at the frame in video we can see what the picture is and how many targets are there.
* but if you want to  know something like state-switch, for example, a person was walking and then stop or it stop for a while and then start to walk, you have to see(cache) more frames.
* 
* see more implementation of 'vp_track_node' and 'vp_message_broker_node' which saved history frame meta data and then work based on them.
* 1. vp_track_node          : save previous locations of targets and then do tracking based on them, we need see more frames to track targets in video.
* 2. vp_message_broker_node : save previous ba_flags and then do notifying based on them, we need see more frames to check if state-switch has happened.
* ##########################################
*/ 
namespace vp_objects {
    // frame meta, which contains frame-related data. it is kind of important meta in pipeline.
    class vp_frame_meta: public vp_meta {
    private:
        /* data */
    public:
        vp_frame_meta(cv::Mat frame, int frame_index = -1, int channel_index = -1, int original_width = 0, int original_height = 0, int fps = 0);
        ~vp_frame_meta();

        // define copy constructor since we need deep copy operation.
        vp_frame_meta(const vp_frame_meta& meta);

        // frame the meta belongs to, filled by src nodes.
        int frame_index;

        // fps for current video.
        int fps;

        // orignal frame width, fiiled by src nodes.
        int original_width;
        // original frame height, filled by src nodes.
        int original_height;

        // image data the meta holds, filled by src nodes.
        // deep copy needed here for this member.
        cv::Mat frame;

        // osd image data the meta holds, filled by osd node if exists.
        // deep copy needed here for this member.
        cv::Mat osd_frame;

        // targets created/appended by primary infer nodes, and then updated by secondary infer nodes if exist.
        // it is shared_ptr<...> type just to keep same as elements.
        // deep copy needed here for this member.
        std::vector<std::shared_ptr<vp_objects::vp_frame_target>> targets;

        // frame elements created by vp_ba_node if exists, it is shared_ptr<...> type since we do not know what specific elements are here.
        // deep copy needed here for this member.
        std::vector<std::shared_ptr<vp_objects::vp_frame_element>> elements;

        // pose targets created/appened by primary infer nodes.
        std::vector<std::shared_ptr<vp_objects::vp_frame_pose_target>> pose_targets;

        // face targets created/appened by primary infer nodes.
        std::vector<std::shared_ptr<vp_objects::vp_frame_face_target>> face_targets;

        // ba results filled by vp_ba_node if exists, it is a map relationship of element, target and ba_flag.
        // 1. element   : where
        // 2. target    : who
        // 3. ba_flag   : what
        // it is a cache for ba results(n*n) of current frame, we can check if state-switch has happened latter based on this value. 
        // deep copy needed here for this member.
        std::vector<std::tuple<std::shared_ptr<vp_frame_element>, std::shared_ptr<vp_frame_target>, vp_ba::vp_ba_flag>> ba_flags_map;

        // copy myself
        virtual std::shared_ptr<vp_meta> clone() override;
    };

}