
#include "vp_trt_vehicle_plate_detector_v2.h"

namespace vp_nodes {
    
    vp_trt_vehicle_plate_detector_v2::vp_trt_vehicle_plate_detector_v2(std::string node_name, 
                                                                std::string plate_det_model_path, 
                                                                std::string char_rec_model_path):
                                                                vp_primary_infer_node(node_name, "") {
        plate_detector = std::make_shared<trt_vehicle::VehiclePlateDetector>(plate_det_model_path, char_rec_model_path);
        this->initialized();
    }
    
    vp_trt_vehicle_plate_detector_v2::~vp_trt_vehicle_plate_detector_v2() {

    }

    void vp_trt_vehicle_plate_detector_v2::postprocess(const std::vector<cv::Mat>& raw_outputs, const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) {

    }

    // please refer to vp_infer_node::run_infer_combinations
    void vp_trt_vehicle_plate_detector_v2::run_infer_combinations(const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) {
        std::vector<cv::Mat> mats_to_infer;

        // start
        auto start_time = std::chrono::system_clock::now();

        // prepare data, as same as base class
        vp_primary_infer_node::prepare(frame_meta_with_batch, mats_to_infer);
        auto prepare_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time);

        start_time = std::chrono::system_clock::now();
        // here we detect vehicle plate on the whole big frame
        std::vector<std::vector<trt_vehicle::Plate>> plates;
        plate_detector->detect(mats_to_infer, plates);

        for (int i = 0; i < plates.size(); i++) {
            auto& frame_meta = frame_meta_with_batch[i];
            for (int j = 0; j < plates[i].size(); j++) {
                auto& plate = plates[i][j];

                // only plate detection but no recognition result
                if (plate.text.empty()) {
                    continue;
                }
                
                // check value range
                auto x = std::max(0, plate.x);
                auto y = std::max(0, plate.y);
                auto w = std::min(plate.width, frame_meta->frame.cols - x);
                auto h = std::min(plate.height, frame_meta->frame.rows - y);
                if (w <= 0 || h <=0) {
                    continue;
                }
                
                // create target and update back into frame meta
                /* 
                  we treat vehicle plate as target in vp_frame_meta.targets,
                  1. x,y,width,height in vp_frame_target stand for location of plate in frame
                  2. primary_label in vp_frame_target stands for plate color and plate text, which combined with '_'
                */ 
                auto target = std::make_shared<vp_objects::vp_frame_target>(x, y, w, h, 
                                                    -1, 0, frame_meta->frame_index, frame_meta->channel_index, plate.color + "_" + plate.text);
                frame_meta->targets.push_back(target);
                VP_INFO("plate-color:" + plate.color + "    plate-text:" + plate.text);
            }
        }
        auto infer_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time);

        // can not calculate preprocess time and postprocess time, set 0 by default.
        vp_infer_node::infer_combinations_time_cost(mats_to_infer.size(), prepare_time.count(), 0, infer_time.count(), 0);
    }
}