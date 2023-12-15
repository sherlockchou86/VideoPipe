#pragma once

#include "../vp_primary_infer_node.h"
#include "../../third_party/trt_vehicle/models/vehicle_scanner.h"

namespace vp_nodes {
    // vehicle scanner based on tensorrt using trt_vehicle library
    class vp_trt_vehicle_scanner: public vp_primary_infer_node
    {
    private:
        std::shared_ptr<trt_vehicle::VehicleScanner> vehicle_scanner = nullptr;
    protected:
        // we need a totally new logic for the whole infer combinations
        // no separate step pre-defined needed in base class
        virtual void run_infer_combinations(const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) override;
        // override pure virtual method, for compile pass
        virtual void postprocess(const std::vector<cv::Mat>& raw_outputs, const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) override;
    public:
        vp_trt_vehicle_scanner(std::string node_name, std::string vehicle_scan_model_path = "");
        ~vp_trt_vehicle_scanner();
    };

}