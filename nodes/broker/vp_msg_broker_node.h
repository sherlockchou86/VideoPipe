
#pragma once

#include "../vp_node.h"

namespace vp_nodes {
    // broke for what type of data (vp_frame_target, vp_frame_face_target or others)
    enum class vp_broke_for {
        NORMAL,  // vp_frame_target
        FACE,    // vp_frame_face_target
        TEXT,    // vp_frame_text_target
        POSE     // vp_frame_pose_target
                 // others to extend
    };

    // base node for message brokers, 
    // used to serialize objects (inside vp_frame_meta) to structured data and then push them to external modules like kafka, file or sockets. 
    // note: 
    // 1. this node works asynchronously which would not block pipeline.
    // 2. this class can not be initialized directly.
    class vp_msg_broker_node: public vp_node
    {
    private:
        // warning if cache size greater than threshold
        int broking_cache_warn_threshold = 50;
        bool broking_cache_warned = false;

        // ignore if cache size greater than threshold (skip directly)
        int broking_cache_ignore_threshold = 200;

        // cache frames to be broked
        std::queue<std::shared_ptr<vp_objects::vp_frame_meta>> frames_to_broke;
        vp_utils::vp_semaphore broking_cache_semaphore;

        // broking thread
        std::thread broking_th;
        // broking function
        void broking_run();
    protected:
        virtual std::shared_ptr<vp_objects::vp_meta> handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) override final;
        virtual std::shared_ptr<vp_objects::vp_meta> handle_control_meta(std::shared_ptr<vp_objects::vp_control_meta> meta) override final;

        // serialize objects to message which SHOULD be implemented in child class.
        virtual void format_msg(const std::shared_ptr<vp_objects::vp_frame_meta>& meta, std::string& msg) = 0;
        // broke message to external modules which SHOULD be implemented in child class.
        virtual void broke_msg(const std::string& msg) = 0;

        // node applied for what type of target
        vp_broke_for broke_for = vp_broke_for::NORMAL;

        // string for broke_for
        std::map<vp_broke_for, std::string> broke_fors = {{vp_broke_for::NORMAL, "normal"}, 
                                                        {vp_broke_for::FACE, "face"}, 
                                                        {vp_broke_for::TEXT, "text"}, 
                                                        {vp_broke_for::POSE, "pose"}};
    public:
        vp_msg_broker_node(std::string node_name, 
                        vp_broke_for broke_for = vp_broke_for::NORMAL, 
                        int broking_cache_warn_threshold = 50, 
                        int broking_cache_ignore_threshold = 200);
        virtual ~vp_msg_broker_node();
    };
}