
#include "vp_sync_node.h"
#include <iostream>

namespace vp_nodes {
        
    vp_sync_node::vp_sync_node(std::string node_name, vp_sync_mode mode, int timeout):
        vp_node(node_name), 
        mode(mode),
        timeout(timeout) {
            assert(timeout > 0);
            this->initialized();
    }
    
    vp_sync_node::~vp_sync_node() {
        deinitialized();
    }
    
    std::shared_ptr<vp_objects::vp_meta> vp_sync_node::handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) {
        if (all_indexs_last_syned.count(meta->channel_index) == 0) {
            all_indexs_last_syned[meta->channel_index] = -1; // initialize by -1
        }
        auto& index_last_synced = all_indexs_last_syned[meta->channel_index];
        auto& meta_waiting_for_sync = all_meta_waiting_for_sync[meta->channel_index];

        // discard because it has been already synced or timeout before
        if (meta->frame_index <= index_last_synced) {
            // warn log for discard
            VP_WARN(vp_utils::string_format("[%s][channel%d] discard for frame_index:[%d], because it has been synced or timeout before...", node_name.c_str(), meta->channel_index, meta->frame_index));
            return nullptr;
        }
        
        bool need_wait = true;
        bool start_of_queue = true;
        for (auto i = meta_waiting_for_sync.begin(); i != meta_waiting_for_sync.end();) {
            auto meta_type = (*i)->meta_type;

            // it is control meta AND at the start of queue, push to downstream directly
            if (meta_type == vp_objects::vp_meta_type::CONTROL) {
                if (start_of_queue) {
                    auto waiting_control_meta = dynamic_pointer_cast<vp_objects::vp_control_meta>((*i));
                    pendding_meta(waiting_control_meta);
                    i = meta_waiting_for_sync.erase(i);
                    VP_DEBUG(vp_utils::string_format("[%s][channel%d] sync control meta for control_uid:[%s], now cache size of sync is:[%d]", node_name.c_str(), meta->channel_index, waiting_control_meta->control_uid, meta_waiting_for_sync.size()));
                }
                else {
                    // skip
                    i++;
                }
            }
            else {
                // it is frame meta
                auto waiting_meta = dynamic_pointer_cast<vp_objects::vp_frame_meta>((*i));
                auto frame_index = waiting_meta->frame_index;

                auto frame_period = 1000.0 / meta->fps;
                auto delta = (meta->frame_index - frame_index) * frame_period;

                // timeout logic
                if (delta >= timeout) {
                    pendding_meta(waiting_meta);
                    index_last_synced = frame_index;
                    i = meta_waiting_for_sync.erase(i);

                    // warn log for timeout
                    VP_WARN(vp_utils::string_format("[%s][channel%d] timeout for frame_index:[%d], push to downstream directly...", node_name.c_str(), meta->channel_index, waiting_meta->frame_index));
                }
                else {
                    // sync logic
                    if (frame_index == meta->frame_index) {
                        sync(waiting_meta, meta);
                        pendding_meta(waiting_meta);
                        index_last_synced = frame_index;
                        i = meta_waiting_for_sync.erase(i);
                        // do not need wait for sync
                        need_wait = false;
                        VP_DEBUG(vp_utils::string_format("[%s][channel%d] sync frame meta for frame_index:[%d], now cache size of sync is:[%d]", node_name.c_str(), meta->channel_index, waiting_meta->frame_index, meta_waiting_for_sync.size()));
                        break;
                    }
                    // skip
                    i++;
                    start_of_queue = false;
                }
            }
        }

        if (need_wait) {
            meta_waiting_for_sync.push_back(meta);
        }
        return nullptr; // important
    }

    std::shared_ptr<vp_objects::vp_meta> vp_sync_node::handle_control_meta(std::shared_ptr<vp_objects::vp_control_meta> meta) {
        if (all_control_uids_last_synced.count(meta->channel_index) == 0) {
            all_control_uids_last_synced[meta->channel_index] = "";  // initialize by empty
        }
        auto& control_uid_last_synced = all_control_uids_last_synced[meta->channel_index];
        auto& meta_waiting_for_sync = all_meta_waiting_for_sync[meta->channel_index];

        if (control_uid_last_synced != meta->control_uid) {
            meta_waiting_for_sync.push_back(meta);
            control_uid_last_synced = meta->control_uid;
        }

        return nullptr;  // important
    }

    void vp_sync_node::sync(std::shared_ptr<vp_objects::vp_frame_meta> des, std::shared_ptr<vp_objects::vp_frame_meta> src) {
        // point to the same vp_frame_meta object
        if (des == src) {
            return;
        }

        /*
        * normal data to sync:
        * 1. osd frame
        * 2. mask
        */
        if (des->osd_frame.empty() && !src->osd_frame.empty()) {
            des->osd_frame = src->osd_frame;
        }
        if (des->mask.empty() && !src->mask.empty()) {
            des->mask = src->mask;
        }
        
        /*
        * infer data to sync:
        * 1. targets 
        * 2. face targets
        * 3. text targets
        * 4. pose targets
        */
        if (mode == vp_sync_mode::MERGE) {
            // merge target lists between 2 vp_frame_meta objects
            // insert pointers directly
            des->targets.insert(des->targets.end(), src->targets.begin(), src->targets.end());
            des->face_targets.insert(des->face_targets.end(), src->face_targets.begin(), src->face_targets.end());
            des->text_targets.insert(des->text_targets.end(), src->text_targets.begin(), src->text_targets.end());
            des->pose_targets.insert(des->pose_targets.end(), src->pose_targets.begin(), src->pose_targets.end());
        }
        else {
            // update properties of targets
            // first make sure the size of targets are equal
            assert(des->targets.size() == src->targets.size());
            assert(des->face_targets.size() == src->face_targets.size());
            assert(des->text_targets.size() == src->text_targets.size());
            assert(des->pose_targets.size() == src->pose_targets.size());

            for (int i = 0; i < src->targets.size(); i++) {
                auto& des_target = des->targets[i];
                auto& src_target = src->targets[i];

                if (des_target->track_id == -1 && src_target->track_id != -1) {
                    des_target->track_id = src_target->track_id;
                }
                if (des_target->tracks.size() == 0 && src_target->tracks.size() != 0) {
                    des_target->tracks = src_target->tracks;
                }
                if (des_target->mask.empty() && !src_target->mask.empty()) {
                    des_target->mask = src_target->mask;
                }
                if (des_target->embeddings.size() == 0 && src_target->embeddings.size() != 0) {
                    des_target->embeddings = src_target->embeddings;
                }

                des_target->secondary_class_ids.insert(des_target->secondary_class_ids.end(), src_target->secondary_class_ids.begin(), src_target->secondary_class_ids.end());
                des_target->secondary_labels.insert(des_target->secondary_labels.end(), src_target->secondary_labels.begin(), src_target->secondary_labels.end());
                des_target->secondary_scores.insert(des_target->secondary_scores.end(), src_target->secondary_scores.begin(), src_target->secondary_scores.end());

                des_target->sub_targets.insert(des_target->sub_targets.end(), src_target->sub_targets.begin(), src_target->sub_targets.end());
            }
            
            for (int i = 0; i < src->face_targets.size(); i++) {
                auto& des_face_target = des->face_targets[i];
                auto& src_face_target = src->face_targets[i];

                if (des_face_target->track_id == -1 && src_face_target->track_id != -1) {
                    des_face_target->track_id = src_face_target->track_id;
                }
                if (des_face_target->tracks.size() == 0 && src_face_target->tracks.size() != 0) {
                    des_face_target->tracks = src_face_target->tracks;
                }
                if (des_face_target->embeddings.size() == 0 && src_face_target->embeddings.size() != 0) {
                    des_face_target->embeddings = src_face_target->embeddings;
                }         
            } 
        }
    }
}