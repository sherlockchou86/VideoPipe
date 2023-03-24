

#include "vp_classifier_node.h"

namespace vp_nodes {
    
    vp_classifier_node::vp_classifier_node(std::string node_name, 
                                            std::string model_path, 
                                            std::string model_config_path, 
                                            std::string labels_path, 
                                            int input_width, 
                                            int input_height, 
                                            int batch_size,
                                            std::vector<int> p_class_ids_applied_to,
                                            int crop_padding,
                                            bool need_softmax,
                                            float scale,
                                            cv::Scalar mean,
                                            cv::Scalar std,
                                            bool swap_rb,
                                            bool swap_chn):
                                            vp_secondary_infer_node(node_name,
                                                                    model_path,
                                                                    model_config_path,
                                                                    labels_path,
                                                                    input_width, 
                                                                    input_height,
                                                                    batch_size,
                                                                    p_class_ids_applied_to,
                                                                    crop_padding,
                                                                    scale,
                                                                    mean,
                                                                    std,
                                                                    swap_rb,
                                                                    swap_chn), need_softmax(need_softmax){
        this->initialized();
    }
    
    vp_classifier_node::~vp_classifier_node() {

    }

    void vp_classifier_node::postprocess(const std::vector<cv::Mat>& raw_outputs, const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) {
        // make sure heads of output are not zero
        assert(raw_outputs.size() > 0);
        assert(frame_meta_with_batch.size() == 1);

        // just one head of output
        auto& output = raw_outputs[0];
        assert(output.dims == 2);

        auto count = output.rows;
        auto index = 0;

        auto& frame_meta = frame_meta_with_batch[0];
        cv::Point class_id_point;
        int class_id;
        double score;

        for (int i = 0; i < count; i++) {
            for (int j = index; j < frame_meta->targets.size(); j++)
            {
                if (!need_apply(frame_meta->targets[j]->primary_class_id)) {
                    // continue as its primary_class_id is not in p_class_ids_applied_to
                    continue;
                }
                
                auto prob = output.row(i);
                cv::minMaxLoc(prob, 0, &score, 0, &class_id_point);

                if (need_softmax) {
                    float maxProb = 0.0;
                    float sum = 0.0;
                    cv::Mat softmaxProb;
                    maxProb = *std::max_element(prob.begin<float>(), prob.end<float>());
                    cv::exp(prob - maxProb, softmaxProb);
                    sum = (float)cv::sum(softmaxProb)[0];
                    softmaxProb /= sum;
                    minMaxLoc(softmaxProb.reshape(1, 1), 0, &score, 0, &class_id_point);
                }
                class_id = class_id_point.x;
                auto label = (labels.size() < class_id + 1) ? "" : labels[class_id];

                // update back to frame meta
                frame_meta->targets[j]->secondary_class_ids.push_back(class_id);
                frame_meta->targets[j]->secondary_scores.push_back(score);
                frame_meta->targets[j]->secondary_labels.push_back(label);

                // break as we found the right target!
                index = j + 1;
                break;
            }
        }
    }
}