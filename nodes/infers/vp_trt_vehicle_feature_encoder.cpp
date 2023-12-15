#include "vp_trt_vehicle_feature_encoder.h"

namespace vp_nodes {
    vp_trt_vehicle_feature_encoder::vp_trt_vehicle_feature_encoder(std::string node_name, 
                                                                    std::string vehicle_feature_model_path, 
                                                                    std::vector<int> p_class_ids_applied_to):
                                                                    vp_secondary_infer_node(node_name, "", "", "", 1, 1, 1, p_class_ids_applied_to) {
        vehicle_feature_encoder = std::make_shared<trt_vehicle::VehicleFeatureEncoder>(vehicle_feature_model_path);
        this->initialized();
    }
    
    vp_trt_vehicle_feature_encoder::~vp_trt_vehicle_feature_encoder() {

    }

    void vp_trt_vehicle_feature_encoder::run_infer_combinations(const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) {
        assert(frame_meta_with_batch.size() == 1);
        std::vector<cv::Mat> mats_to_infer;

        // start
        auto start_time = std::chrono::system_clock::now();

        // prepare data, as same as base class
        vp_secondary_infer_node::prepare(frame_meta_with_batch, mats_to_infer);
        auto prepare_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time);

        // infer using trt_vehicle library
        start_time = std::chrono::system_clock::now();
        std::vector<std::vector<float>> vehicles_feature;
        vehicle_feature_encoder->encode(mats_to_infer, vehicles_feature);

        auto& frame_meta = frame_meta_with_batch[0];
        auto index = 0;
        for (int i = 0; i < vehicles_feature.size(); i++) {
            for (int j = index; j < frame_meta->targets.size(); j++) {
                // need apply or not?
                if (!need_apply(frame_meta->targets[j]->primary_class_id)) {
                    // continue as its primary_class_id is not in p_class_ids_applied_to
                    continue;
                }

                // update back to frame meta
                auto& features = vehicles_feature[i];
                for(int k = 0; k < features.size(); k++) {
                    frame_meta->targets[j]->embeddings.push_back(features[k]);
                }

                // break as we found the right target!
                index = j + 1;
                break;
            }
        }
        auto infer_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time);

        // can not calculate preprocess time and postprocess time, set 0 by default.
        vp_infer_node::infer_combinations_time_cost(mats_to_infer.size(), prepare_time.count(), 0, infer_time.count(), 0);
    }

    void vp_trt_vehicle_feature_encoder::postprocess(const std::vector<cv::Mat>& raw_outputs, const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) {

    }
}