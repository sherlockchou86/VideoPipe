#pragma once

#ifdef VP_WITH_PADDLE
#include "../vp_primary_infer_node.h"
#include "../../third_party/paddle_ocr/include/paddleocr.h"

namespace vp_nodes {
    // ocr based on paddle ocr
    // paddle ocr project(official): https://github.com/PaddlePaddle/PaddleOCR
    // source code(modified based on official): ../../third_party/paddle_ocr
    // note:
    // this class is not based on opencv::dnn module but paddle, a few data members declared in base class are not usable any more(just ignore), such as vp_infer_node::net.
    class vp_ppocr_text_detector_node: public vp_primary_infer_node
    {
    private:
        // paddle ocr instance
        std::shared_ptr<PaddleOCR::PPOCR> ocr;
    protected:
        // we need a totally new logic for the whole infer combinations
        // no separate step pre-defined needed in base class
        virtual void run_infer_combinations(const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) override;
        // override pure virtual method, for compile pass
        virtual void postprocess(const std::vector<cv::Mat>& raw_outputs, const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) override;
    public:
        vp_ppocr_text_detector_node(std::string node_name, 
                                    std::string det_model_dir = "", 
                                    std::string cls_model_dir = "", 
                                    std::string rec_model_dir = "", 
                                    std::string rec_char_dict_path = "");
        ~vp_ppocr_text_detector_node();
    };
}
#endif