#pragma once

/*
* define EXTERNAL archive functions for objects which need to be serialized by cereal library in VideoPipe.
* refer to `https://uscilab.github.io/cereal/serialization_functions.html` for more details.
*/

// object types
#include "../../../objects/vp_frame_target.h"
#include "../../../objects/vp_frame_face_target.h"
#include "../../../objects/vp_frame_text_target.h"
#include "../../../objects/vp_frame_pose_target.h"
#include "../../../objects/vp_sub_target.h"
/* extend for more types of objects in VideoPipe. */

// headers from cereal
#include "../../../third_party/cereal/cereal.hpp"
#include "../../../third_party/cereal/types/vector.hpp"
#include "../../../third_party/cereal/types/memory.hpp"
#include "../../../third_party/cereal/types/string.hpp"
#include "../../../third_party/cereal/types/utility.hpp"
#include "../../../third_party/cereal/archives/json.hpp"
#include "../../../third_party/cereal/archives/xml.hpp"

/* same namespace as object types */
namespace vp_objects {
    /* vp_frame_target */
    template<typename Archive>
    void serialize(Archive& archive, vp_frame_target& target) {
        // define the form of structured data for vp_frame_target
        archive(cereal::make_nvp("x", target.x),
                cereal::make_nvp("y", target.y),
                cereal::make_nvp("width", target.width),
                cereal::make_nvp("height", target.height),
                cereal::make_nvp("primary_class_id", target.primary_class_id),
                cereal::make_nvp("primary_score", target.primary_score),
                cereal::make_nvp("primary_label", target.primary_label),
                cereal::make_nvp("channel_index", target.channel_index),
                cereal::make_nvp("frame_index", target.frame_index),
                cereal::make_nvp("track_id", target.track_id),
                cereal::make_nvp("secondary_class_ids", target.secondary_class_ids),
                cereal::make_nvp("secondary_scores", target.secondary_scores),
                cereal::make_nvp("secondary_labels", target.secondary_labels),
                cereal::make_nvp("sub_targets", target.sub_targets),
                cereal::make_nvp("embeddings", target.embeddings));
    }

    template<typename Archive>
    void serialize(Archive& archive, vp_sub_target& target) {
        // define the form of structured data for vp_sub_target
        archive(cereal::make_nvp("x", target.x),
                cereal::make_nvp("y", target.y),
                cereal::make_nvp("width", target.width),
                cereal::make_nvp("height", target.height),
                cereal::make_nvp("class_id", target.class_id),
                cereal::make_nvp("score", target.score),
                cereal::make_nvp("label", target.label),
                cereal::make_nvp("frame_index", target.frame_index),
                cereal::make_nvp("channel_index", target.channel_index),
                cereal::make_nvp("attachments", target.attachments));
    }
    /* END OF vp_frame_target */


    /* vp_frame_face_target */
    template<typename Archive>
    void serialize(Archive& archive, vp_frame_face_target& target) {
        // define the form of structured data for vp_frame_face_target
        archive(cereal::make_nvp("x", target.x),
                cereal::make_nvp("y", target.y),
                cereal::make_nvp("width", target.width),
                cereal::make_nvp("height", target.height),
                cereal::make_nvp("score", target.score),
                cereal::make_nvp("embeddings", target.embeddings),
                cereal::make_nvp("key_points", target.key_points),
                cereal::make_nvp("track_id", target.track_id));
    }
    /* END OF vp_frame_face_target */


    /* vp_frame_text_target */
    template<typename Archive>
    void serialize(Archive& archive, vp_frame_text_target& target) {
        // define the form of structured data for vp_frame_text_target
        archive(cereal::make_nvp("text", target.text),
                cereal::make_nvp("score", target.score),
                cereal::make_nvp("region", target.region_vertexes));
    }
    /* END OF vp_frame_text_target */


    /* vp_frame_pose_target */
    template<typename Archive>
    void serialize(Archive& archive, vp_frame_pose_target& target) {
        // define the form of structured data for vp_frame_pose_target

    }
    /* END OF vp_frame_pose_target */
}