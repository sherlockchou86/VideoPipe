
#include "vp_ppocr_text_detector_node.h"
#include "../../objects/vp_frame_text_target.h"

namespace vp_nodes {
        
    vp_ppocr_text_detector_node::vp_ppocr_text_detector_node(std::string node_name, 
                                                            std::string det_model_dir, 
                                                            std::string cls_model_dir, 
                                                            std::string rec_model_dir, 
                                                            std::string rec_char_dict_path):
                                                            vp_primary_infer_node(node_name, "") {
        // to make the code simpler, paddle ocr has no more config other than model path
        // we need modify source code at ../../third_party/paddle_ocr/ if we need tune the parameters 
        ocr = std::make_shared<PaddleOCR::PPOCR>(det_model_dir, cls_model_dir, rec_model_dir, rec_char_dict_path);
        this->initialized();
    }
    
    vp_ppocr_text_detector_node::~vp_ppocr_text_detector_node() {

    }
    
    void vp_ppocr_text_detector_node::postprocess(const std::vector<cv::Mat>& raw_outputs, const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) {

    }
    // please refer to vp_infer_node::run_infer_combinations
    void vp_ppocr_text_detector_node::run_infer_combinations(const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) {
        assert(frame_meta_with_batch.size() == 1);
        std::vector<cv::Mat> mats_to_infer;

        // start
        auto start_time = std::chrono::system_clock::now();

        // prepare data, as same as base class
        vp_primary_infer_node::prepare(frame_meta_with_batch, mats_to_infer);
        auto prepare_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time);

        start_time = std::chrono::system_clock::now();
        // call paddle ocr
        auto ocr_results = ocr->ocr(mats_to_infer);
        assert(ocr_results.size() == 1);

        auto& ocr_result = ocr_results[0];
        auto& frame_meta = frame_meta_with_batch[0];

        // scan text detected in frame
        for (int i = 0; i < ocr_result.size(); i++) {
            /* code */
            auto& text = ocr_result[i];
            std::vector<std::pair<int, int>> region_vertexes = {{text.box[0][0], text.box[0][1]}, 
                                                                {text.box[1][0], text.box[1][1]}, 
                                                                {text.box[2][0], text.box[2][1]}, 
                                                                {text.box[3][0], text.box[3][1]}};
            // create text target and update back into frame meta
            auto text_target = std::make_shared<vp_objects::vp_frame_text_target>(region_vertexes, text.text, text.score);
            frame_meta->text_targets.push_back(text_target);
        }
        auto infer_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time);

        // can not calculate preprocess time and postprocess time, set 0 by default.
        vp_infer_node::infer_combinations_time_cost(frame_meta_with_batch.size(), prepare_time.count(), 0, infer_time.count(), 0);
    }
}