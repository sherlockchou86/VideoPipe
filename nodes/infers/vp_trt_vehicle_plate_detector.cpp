
#include "vp_trt_vehicle_plate_detector.h"

namespace vp_nodes {
    
    vp_trt_vehicle_plate_detector::vp_trt_vehicle_plate_detector(std::string node_name, 
                                                                std::string plate_det_model_path, 
                                                                std::string char_rec_model_path):
                                                                vp_secondary_infer_node(node_name, "") {
        plate_detector = std::make_shared<trt_vehicle::VehiclePlateDetector>(plate_det_model_path, char_rec_model_path);
        this->initialized();
    }
    
    vp_trt_vehicle_plate_detector::~vp_trt_vehicle_plate_detector() {

    }

    void vp_trt_vehicle_plate_detector::postprocess(const std::vector<cv::Mat>& raw_outputs, const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) {

    }

    // please refer to vp_infer_node::run_infer_combinations
    void vp_trt_vehicle_plate_detector::run_infer_combinations(const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) {
        assert(frame_meta_with_batch.size() == 1);
        std::vector<cv::Mat> mats_to_infer;

        // start
        auto start_time = std::chrono::system_clock::now();

        // prepare data, as same as base class
        vp_secondary_infer_node::prepare(frame_meta_with_batch, mats_to_infer);
        auto prepare_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time);

        start_time = std::chrono::system_clock::now();
        // here we detect vehicle plate for all target(like car/bus) in frame
        std::vector<std::vector<trt_vehicle::Plate>> plates;
        plate_detector->detect(mats_to_infer, plates, true);  // detect at most 1 plate for each image

        auto& frame_meta = frame_meta_with_batch[0];
        for (int i = 0; i < plates.size(); i++) {
            for (int j = 0; j < plates[i].size(); j++) {
                auto& plate = plates[i][j];

                // create sub target and update back into frame meta
                // we treat vehicle plate as sub target of those in vp_frame_meta.targets
                auto sub_target = std::make_shared<vp_objects::vp_sub_target>(plate.x, plate.y, plate.width, plate.height, 
                                                    -1, 0, "", frame_meta->frame_index, frame_meta->channel_index);
                sub_target->attachments.push_back(plate.color);
                sub_target->attachments.push_back(plate.text);
                frame_meta->targets[i]->sub_targets.push_back(sub_target);
            }
        }
        auto infer_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time);

        // can not calculate preprocess time and postprocess time, set 0 by default.
        vp_infer_node::infer_combinations_time_cost(mats_to_infer.size(), prepare_time.count(), 0, infer_time.count(), 0);
    }
}