#pragma once

#ifdef VP_WITH_TRT
#include "../vp_primary_infer_node.h"
#include "../../third_party/trt_yolov8/trt_yolov8_pose_detector.h"

namespace vp_nodes {
    // universal yolov8 pose detector based on tensorrt using third_party/trt_yolov8 library
    class vp_trt_yolov8_pose_detector: public vp_primary_infer_node
    {
    private:
        std::shared_ptr<trt_yolov8::trt_yolov8_pose_detector> yolov8_pose_detector = nullptr;
    protected:
        // we need a totally new logic for the whole infer combinations
        // no separate step pre-defined needed in base class
        virtual void run_infer_combinations(const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) override;
        // override pure virtual method, for compile pass
        virtual void postprocess(const std::vector<cv::Mat>& raw_outputs, const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) override;
    public:
        vp_trt_yolov8_pose_detector(std::string node_name, std::string model_path);
        ~vp_trt_yolov8_pose_detector();
    };
}
#endif