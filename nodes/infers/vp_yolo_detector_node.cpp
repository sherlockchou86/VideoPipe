
#include "vp_yolo_detector_node.h"

namespace vp_nodes {
    
    vp_yolo_detector_node::vp_yolo_detector_node(std::string node_name, 
                            std::string model_path, 
                            std::string model_config_path, 
                            std::string labels_path, 
                            int input_width, 
                            int input_height, 
                            int batch_size,
                            int class_id_offset,
                            float score_threshold,
                            float confidence_threshold,
                            float nms_threshold,
                            float scale,
                            cv::Scalar mean,
                            cv::Scalar std,
                            bool swap_rb):
                            vp_primary_infer_node(node_name, model_path, model_config_path,labels_path,input_width, input_height, batch_size, class_id_offset, scale, mean, std, swap_rb),
                            score_threshold(score_threshold),
                            confidence_threshold(confidence_threshold),
                            nms_threshold(nms_threshold) {
        this->initialized();
    }
    
    vp_yolo_detector_node::~vp_yolo_detector_node() {
        deinitialized();
    }
    
    void vp_yolo_detector_node::postprocess(const std::vector<cv::Mat>& raw_outputs, const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) {
        // make sure heads of output are not zero
        assert(raw_outputs.size() > 0);

        // check dims of each output
        auto dim_offset = raw_outputs[0].dims == 2 ? 0 : 1;
        auto batch = dim_offset == 0 ? 1 : raw_outputs[0].size[0];

        std::vector<int> class_ids;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;

        for (int k = 0; k < batch; k++) {  // scan batch
            auto& frame_meta = frame_meta_with_batch[k];
            for (int i = 0; i < raw_outputs.size(); ++i) {  // scan heads of output
                cv::Mat output = cv::Mat(raw_outputs[i].size[0 + dim_offset], raw_outputs[i].size[1 + dim_offset], CV_32F, const_cast<uchar*>(raw_outputs[i].ptr(k)));

                auto data = (float*)output.data;
                for (int j = 0; j < output.rows; ++j, data += output.cols) {
                    
                    float confidence = data[4];
                    // check confidence threshold
                    if (confidence < confidence_threshold) {
                        continue;
                    }

                    cv::Mat scores = output.row(j).colRange(5, output.cols);
                    cv::Point class_id;
                    double max_score;
                    // Get the value and location of the maximum score
                    cv::minMaxLoc(scores, 0, &max_score, 0, &class_id);

                    // check score threshold
                    if (max_score >= score_threshold) {
                        int centerX = (int)(data[0] * frame_meta->frame.cols);
                        int centerY = (int)(data[1] * frame_meta->frame.rows);
                        int width = (int)(data[2] * frame_meta->frame.cols);
                        int height = (int)(data[3] * frame_meta->frame.rows);
                        int left = centerX - width / 2;
                        int top = centerY - height / 2;

                        class_ids.push_back(class_id.x);
                        confidences.push_back((float)confidence);
                        boxes.push_back(cv::Rect(left, top, width, height));
                    }
                }
            }

            // nms
            std::vector<int> indices;
            cv::dnn::NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold, indices);

            // create target
            for (int i = 0; i < indices.size(); ++i) {
                int idx = indices[i];
                auto box = boxes[idx];
                auto confidence = confidences[idx];
                auto class_id = class_ids[idx];
                auto label = (labels.size() < class_id + 1) ? "" : labels[class_id];

                // check value range
                box.x = std::max(box.x, 0);
                box.y = std::max(box.y, 0);
                box.width = std::min(box.width, frame_meta->frame.cols - box.x);
                box.height = std::min(box.height, frame_meta->frame.rows - box.y);
                if (box.width <= 0 || box.height <= 0) {
                    continue;
                }
                
                // apply offset to class id since multi detectors can exist at front of this one.
                // later we MUST use the new class id (applied offset) instead of orignal one.
                class_id += class_id_offset;

                auto target = std::make_shared<vp_objects::vp_frame_target>(box.x, box.y, box.width, box.height, class_id, confidence, frame_meta->frame_index, frame_meta->channel_index, label);

                // insert target back to frame meta
                frame_meta->targets.push_back(target);
            }

            class_ids.clear();
            confidences.clear();
            boxes.clear();
        }
    }
}