
#include "vp_trt_vehicle_detector.h"

namespace vp_nodes {
    
    vp_trt_vehicle_detector::vp_trt_vehicle_detector(std::string node_name, 
                                                    std::string vehicle_det_model_path):
                                                    vp_primary_infer_node(node_name, "") {
        vehicle_detector = std::make_shared<trt_vehicle::VehicleDetector>(vehicle_det_model_path);
        this->initialized();
    }
    
    vp_trt_vehicle_detector::~vp_trt_vehicle_detector() {

    }

    // please refer to vp_infer_node::run_infer_combinations
    void vp_trt_vehicle_detector::run_infer_combinations(const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) {
        assert(frame_meta_with_batch.size() == 1);
        std::vector<cv::Mat> mats_to_infer;

        // start
        auto start_time = std::chrono::system_clock::now();

        // prepare data, as same as base class
        vp_primary_infer_node::prepare(frame_meta_with_batch, mats_to_infer);
        auto prepare_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time);

        start_time = std::chrono::system_clock::now();
        std::vector<std::vector<trt_vehicle::ObjBox>> vehicles;
        vehicle_detector->detect(mats_to_infer, vehicles);

        assert(vehicles.size() == 1);
        auto& vehicle_list = vehicles[0];
        auto& frame_meta = frame_meta_with_batch[0];

        for (int i = 0; i < vehicle_list.size(); i++) {
            auto& objbox = vehicle_list[i];

            auto target = std::make_shared<vp_objects::vp_frame_target>(objbox.x, objbox.y, objbox.width, objbox.height, 
                                                                        objbox.class_, objbox.score, frame_meta->frame_index, frame_meta->channel_index, objbox.label);
            // create target and update back into frame meta
            frame_meta->targets.push_back(target);
        }
        auto infer_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time);

        // can not calculate preprocess time and postprocess time, set 0 by default.
        vp_infer_node::infer_combinations_time_cost(mats_to_infer.size(), prepare_time.count(), 0, infer_time.count(), 0);
    }

    void vp_trt_vehicle_detector::postprocess(const std::vector<cv::Mat>& raw_outputs, const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) {

    }
}