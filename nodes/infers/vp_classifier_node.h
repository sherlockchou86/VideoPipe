

#pragma once

#include "../vp_secondary_infer_node.h"

namespace vp_nodes {
    // common classifier for image classification task.
    // used for image classification, update secondary_class_ids/secondary_labels/secondary_scores of vp_frame_target.
    class vp_classifier_node: public vp_secondary_infer_node
    {
    private:
        // softmax logic applied on output or not
        bool need_softmax;
    protected:
        virtual void postprocess(const std::vector<cv::Mat>& raw_outputs, const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) override;
    public:
        vp_classifier_node(std::string node_name, 
                            std::string model_path, 
                            std::string model_config_path = "", 
                            std::string labels_path = "", 
                            int input_width = 128, 
                            int input_height = 128, 
                            int batch_size = 1,
                            std::vector<int> p_class_ids_applied_to = std::vector<int>(),
                            int min_width_applied_to = 0,
                            int min_height_applied_to = 0,
                            int crop_padding = 10,
                            bool need_softmax = true,  // imagenet dataset
                            float scale = 1 / 255.0,
                            cv::Scalar mean = cv::Scalar(123.675, 116.28, 103.53),  // imagenet dataset
                            cv::Scalar std = cv::Scalar(0.229, 0.224, 0.225),
                            bool swap_rb = true,
                            bool swap_chn = false);
        ~vp_classifier_node();
    };

}