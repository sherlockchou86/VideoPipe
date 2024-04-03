
#include "vp_restoration_node.h"

namespace vp_nodes {
    
    vp_restoration_node::vp_restoration_node(std::string node_name, 
                            std::string realesrgan_bg_restoration_model,
                            std::string face_restoration_model,
                            bool restoration_to_osd):
                            vp_primary_infer_node(node_name, ""),
                            restoration_to_osd(restoration_to_osd) {        
        /* init net*/
        restoration_net = cv::dnn::readNetFromONNX(realesrgan_bg_restoration_model);
        /* to-do, load face restoration model*/
        #ifdef VP_WITH_CUDA
        //restoration_net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        //restoration_net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        #endif
        this->initialized();
    }
    
    vp_restoration_node::~vp_restoration_node() {
        deinitialized();
    }

    // please refer to vp_infer_node::run_infer_combinations
    void vp_restoration_node::run_infer_combinations(const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) {
        assert(frame_meta_with_batch.size() == 1);
        auto& frame_meta = frame_meta_with_batch[0];

        // 4x larger for RealESRGAN_x4plus model
        cv::Mat target_blob = cv::dnn::blobFromImage(frame_meta->frame, 1 / 255.0, cv::Size(frame_meta->frame.cols, frame_meta->frame.rows), (0, 0, 0), true);
        std::vector<cv::Mat> target_outputs;
        restoration_net.setInput(target_blob);
        restoration_net.forward(target_outputs, restoration_net.getUnconnectedOutLayersNames());

        // parse to image
        auto& output = target_outputs[0];
        cv::Mat output_channel_last;
        cv::transposeND(output, {0,2,3,1}, output_channel_last);
        cv::Mat img_result(output_channel_last.size[1], output_channel_last.size[2], CV_32FC3, output_channel_last.data);
        img_result.convertTo(img_result, CV_8U, 255);
        
        // update back to frame meta
        auto& bg = restoration_to_osd ? frame_meta->osd_frame : frame_meta->frame;
        cv::cvtColor(img_result, bg, cv::COLOR_RGB2BGR);
    }

    void vp_restoration_node::postprocess(const std::vector<cv::Mat>& raw_outputs, const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) {

    }
}