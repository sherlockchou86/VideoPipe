
#ifdef VP_WITH_TRT
#include "vp_trt_yolov8_seg_detector.h"

namespace vp_nodes {
    
    vp_trt_yolov8_seg_detector::vp_trt_yolov8_seg_detector(std::string node_name, 
                                                    std::string model_path,
                                                    std::string labels_path):
                                                    vp_primary_infer_node(node_name, "", "", labels_path) {
        yolov8_seg_detector = std::make_shared<trt_yolov8::trt_yolov8_seg_detector>(model_path);
        this->initialized();
    }
    
    vp_trt_yolov8_seg_detector::~vp_trt_yolov8_seg_detector() {
        deinitialized();
    }

    // please refer to vp_infer_node::run_infer_combinations
    void vp_trt_yolov8_seg_detector::run_infer_combinations(const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) {
        assert(frame_meta_with_batch.size() == 1);
        std::vector<cv::Mat> mats_to_infer;

        // start
        auto start_time = std::chrono::system_clock::now();

        // prepare data, as same as base class
        vp_primary_infer_node::prepare(frame_meta_with_batch, mats_to_infer);
        auto prepare_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time);

        start_time = std::chrono::system_clock::now();
        std::vector<std::vector<Detection>> detections;
        std::vector<std::vector<cv::Mat>> masks;
        yolov8_seg_detector->detect(mats_to_infer, detections, masks);

        assert(detections.size() == 1);
        assert(masks.size() == 1);
        auto& detection_list = detections[0];
        auto& mask_list = masks[0];
        assert(detection_list.size() == mask_list.size());
        auto& frame_meta = frame_meta_with_batch[0];

        for (int i = 0; i < detection_list.size(); i++) {
            auto& objbox = detection_list[i];
            auto& mask = mask_list[i];
            auto scaled_mask = scale_mask(mask, frame_meta->frame);

            // objbox.bbox: center_x center_y width height
            auto rect = get_rect(frame_meta->frame, objbox.bbox);  // convert to: x, y, width,height
            // check value range
            rect.x = std::max(rect.x, 0);
            rect.y = std::max(rect.y, 0);
            rect.width = std::min(rect.width, frame_meta->frame.cols - rect.x);
            rect.height = std::min(rect.height, frame_meta->frame.rows - rect.y);
            if (rect.width <= 0 || rect.height <= 0) {
                continue;
            }

            auto label = labels.size() == 0 ? "" : labels[objbox.class_id];
            auto target = std::make_shared<vp_objects::vp_frame_target>(rect.x, rect.y, rect.width, rect.height, 
                                                                        objbox.class_id, objbox.conf, frame_meta->frame_index, frame_meta->channel_index, label);
            auto rect_mask = scaled_mask(rect);
            target->mask = rect_mask;

            // create target and update back into frame meta
            frame_meta->targets.push_back(target);
        }
        auto infer_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time);

        // can not calculate preprocess time and postprocess time, set 0 by default.
        vp_infer_node::infer_combinations_time_cost(mats_to_infer.size(), prepare_time.count(), 0, infer_time.count(), 0);
    }

    void vp_trt_yolov8_seg_detector::postprocess(const std::vector<cv::Mat>& raw_outputs, const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) {

    }
}
#endif