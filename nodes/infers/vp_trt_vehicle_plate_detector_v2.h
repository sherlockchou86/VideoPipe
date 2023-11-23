#pragma once

#include "../vp_primary_infer_node.h"
#include "../../third_party/trt_vehicle/models/vehicle_plate_detector.h"

namespace vp_nodes {
    // vehicle plate detector based on tensorrt
    // source code: ../../third_party/trt_vehicle
    // note: derived from vp_primary_infer_node since it detects plates on the whole big frame, which is different from vp_trt_vehicle_plate_detector class
    // this class is not based on opencv::dnn module but tensorrt, a few data members declared in base class are not usable any more(just ignore), such as vp_infer_node::net.
    class vp_trt_vehicle_plate_detector_v2: public vp_primary_infer_node
    {
    private:
        /* data */
        std::shared_ptr<trt_vehicle::VehiclePlateDetector> plate_detector = nullptr;
    protected:
        // we need a totally new logic for the whole infer combinations
        // no separate step pre-defined needed in base class
        virtual void run_infer_combinations(const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) override;
        // override pure virtual method, for compile pass
        virtual void postprocess(const std::vector<cv::Mat>& raw_outputs, const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) override;
    public:
        vp_trt_vehicle_plate_detector_v2(std::string node_name, std::string plate_det_model_path = "", std::string char_rec_model_path = "");
        ~vp_trt_vehicle_plate_detector_v2();
    };

}