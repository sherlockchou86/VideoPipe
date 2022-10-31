
#include "vp_track_node.h"
//#include "../objects/shapes/vp_rect.h"

namespace vp_nodes {
        
    vp_track_node::vp_track_node(std::string node_name, 
                                vp_track_for track_for): 
                                vp_node(node_name), 
                                track_for(track_for) {
    }
    
    vp_track_node::~vp_track_node()
    {
    }

    std::shared_ptr<vp_objects::vp_meta> vp_track_node::handle_control_meta(std::shared_ptr<vp_objects::vp_control_meta> meta) {
        return meta;
    }

    std::shared_ptr<vp_objects::vp_meta> vp_track_node::handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) {
        // track for only 1 channel at the same time
        if (channel_index == -1) {
            channel_index = meta->channel_index;
        }
        assert(channel_index == meta->channel_index);
        
        // data used for tracking
        std::vector<vp_objects::vp_rect> rects;      // rects of targets
        std::vector<std::vector<float>> embeddings;  // embeddings of targets
        std::vector<int> track_ids;                  // track ids of targets

        // step 1, collect data
        preprocess(meta, rects, embeddings);

        // step 2, track
        track(rects, embeddings, track_ids);

        // step 3, postprocess
        postprocess(meta, rects, embeddings, track_ids);

        return meta;
    }

    void vp_track_node::preprocess(std::shared_ptr<vp_objects::vp_frame_meta> frame_meta, 
                                std::vector<vp_objects::vp_rect>& target_rects, 
                                std::vector<std::vector<float>>& target_embeddings) {
        if (track_for == vp_track_for::NORMAL) {
            for(auto& i: frame_meta->targets) {
                target_rects.push_back(i->get_rect());      // rect fo target (via i variable)
                target_embeddings.push_back(i->embeddings); // embeddings of target (via i variable)
            }
        }

        if (track_for == vp_track_for::FACE) {
            for(auto& i: frame_meta->face_targets) {
                target_rects.push_back(i->get_rect());       // rect of face target (via i variable)
                target_embeddings.push_back(i->embeddings);  // embeddings of face target (via i variable)
            }
        }
        // ... extend for more track for...
    }

    // write track_ids back to frame meta
    // we can also cache history rects for each target, and then push them back to tracks field (such as vp_frame_target::tracks)
    void vp_track_node::postprocess(std::shared_ptr<vp_objects::vp_frame_meta> frame_meta, 
                    const std::vector<vp_objects::vp_rect>& target_rects, 
                    const std::vector<std::vector<float>>& target_embeddings, 
                    const std::vector<int>& track_ids) {

        if(track_ids.empty()){
            return;
        }
        // assert for length of vectors since they are generated by step1 & step2 separately
        //assert(target_rects.size() == target_embeddings.size());
        assert(target_rects.size() == track_ids.size());

        if (track_for == vp_track_for::NORMAL) {
            //assert(target_rects.size() == frame_meta->targets.size());
            frame_meta->targets.resize(target_rects.size());
            for(int i = 0; i < frame_meta->targets.size(); i++) {
                auto& target = frame_meta->targets[i];
                auto& rect = target_rects[i];
                auto& track_id = track_ids[i];
                target->x = rect.x;
                target->y = rect.y;
                target->width = rect.width;
                target->height = rect.height;
                tracks_by_id[track_id].push_back(rect);    // cache
                target->track_id = track_id;               // write track_id back to target
                target->tracks = tracks_by_id[track_id];   // write tracks back to target
            }
        }

        if (track_for == vp_track_for::FACE){
            //assert(target_rects.size() == frame_meta->face_targets.size());
            // Use the tracking results to update the detection results of the current frame
            //TODO: 跟踪结果和检测结果不一致时的情况分析。
            frame_meta->face_targets.resize(target_rects.size()); //TODO 这里的resize 感觉不合适！  理解后重新写
            for (int i = 0; i < frame_meta->face_targets.size(); i++) {
                /* code */
                auto& face = frame_meta->face_targets[i];
                auto& rect = target_rects[i];
                auto& track_id = track_ids[i];
                face->x = rect.x;
                face->y = rect.y;
                face->width = rect.width;
                face->height = rect.height;
                // no cache needed since no tracks field for vp_frame_face_target
                face->track_id = track_id;  // write track_id back to face target
            }
        }

        // ... extend for more track for...

        // TO-DO
        // remove cache tracks if has been long time since last updated (maybe it disappeared already).
    }
}