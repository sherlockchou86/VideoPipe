
#ifdef VP_WITH_TRT
#include "vp_trt_yolov8_pose_detector.h"

namespace vp_nodes {
    
    vp_trt_yolov8_pose_detector::vp_trt_yolov8_pose_detector(std::string node_name, 
                                                    std::string model_path):
                                                    vp_primary_infer_node(node_name, "") {
        yolov8_pose_detector = std::make_shared<trt_yolov8::trt_yolov8_pose_detector>(model_path);
        this->initialized();
    }
    
    vp_trt_yolov8_pose_detector::~vp_trt_yolov8_pose_detector() {
        deinitialized();
    }

    // please refer to vp_infer_node::run_infer_combinations
    void vp_trt_yolov8_pose_detector::run_infer_combinations(const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) {
        assert(frame_meta_with_batch.size() == 1);
        std::vector<cv::Mat> mats_to_infer;

        // start
        auto start_time = std::chrono::system_clock::now();

        // prepare data, as same as base class
        vp_primary_infer_node::prepare(frame_meta_with_batch, mats_to_infer);
        auto prepare_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time);

        start_time = std::chrono::system_clock::now();
        std::vector<std::vector<Detection>> detections;
        yolov8_pose_detector->detect(mats_to_infer, detections);

        assert(detections.size() == 1);
        auto& detection_list = detections[0];
        auto& frame_meta = frame_meta_with_batch[0];

        for (int i = 0; i < detection_list.size(); i++) {
            auto& objbox = detection_list[i];
            auto rect = get_rect_adapt_landmark(frame_meta->frame, objbox.bbox, objbox.keypoints);
            
            std::vector<vp_objects::vp_pose_keypoint> kps;    
            for (int j = 0; j < 51; j += 3) {
                kps.push_back(vp_objects::vp_pose_keypoint {j, int(objbox.keypoints[j]), int(objbox.keypoints[j + 1]), objbox.keypoints[j + 2]});
            }
                        
            auto pose_target = std::make_shared<vp_objects::vp_frame_pose_target>(vp_objects::vp_pose_type::yolov8_pose_17, kps);
            frame_meta->pose_targets.push_back(pose_target);            
        }
        auto infer_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time);

        // can not calculate preprocess time and postprocess time, set 0 by default.
        vp_infer_node::infer_combinations_time_cost(mats_to_infer.size(), prepare_time.count(), 0, infer_time.count(), 0);
    }

    void vp_trt_yolov8_pose_detector::postprocess(const std::vector<cv::Mat>& raw_outputs, const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) {

    }
}
#endif