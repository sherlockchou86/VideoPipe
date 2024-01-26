
#pragma once

#include <map>
#include <assert.h>
#include "../vp_node.h"

namespace vp_nodes {
    // track node applied to which type of target (vp_frame_target, vp_frame_face_target or others)
    enum class vp_track_for {
        NORMAL = 1,    // vp_frame_target
        FACE = 2       // vp_frame_face_target
                       // others to extend
    };

    // base class for tracking, can not be initialized directly.
    // note that a track node can work on different channels at the same time
    class vp_track_node: public vp_node {
    private:
        // track for
        vp_track_for track_for = vp_track_for::NORMAL;
        
        // cache tracks at previous frames
        // std::map<int, std::vector<vp_objects::vp_rect>> tracks_by_id;
        std::map<int, std::map<int, std::vector<vp_objects::vp_rect>>> all_tracks_by_id;

        // stamp
        // std::map<int, int> last_tracked_frame_indexes;
        std::map<int, std::map<int, int>> all_last_tracked_frame_indexes;

        // remove cache tracks if it has been long time since last tracked.
        const int max_allowed_disappear_frames = 25;
    protected:
        virtual std::shared_ptr<vp_objects::vp_meta> handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) override final;
        virtual std::shared_ptr<vp_objects::vp_meta> handle_control_meta(std::shared_ptr<vp_objects::vp_control_meta> meta) override final;

        // prepare data according to `track_for`
        void preprocess(std::shared_ptr<vp_objects::vp_frame_meta> frame_meta, 
                        std::vector<vp_objects::vp_rect>& target_rects, 
                        std::vector<std::vector<float>>& target_embeddings);
        
        // track api
        // it is a pure virtual function which should be implemented by derived class.
        // In:  rects & embeddings whose size() can be zero
        // Out: track ids
        virtual void track(int channel_index, const std::vector<vp_objects::vp_rect>& target_rects, 
                        const std::vector<std::vector<float>>& target_embeddings, 
                        std::vector<int>& track_ids) = 0;

        // write track_ids back to frame meta
        // we can also cache history rects for each target, and then push them back to tracks field (like vp_frame_target::tracks)
        void postprocess(std::shared_ptr<vp_objects::vp_frame_meta> frame_meta, 
                        const std::vector<vp_objects::vp_rect>& target_rects, 
                        const std::vector<std::vector<float>>& target_embeddings, 
                        const std::vector<int>& track_ids);
    public:
        vp_track_node(std::string node_name, vp_track_for track_for = vp_track_for::NORMAL);
        virtual ~vp_track_node();
    };
}