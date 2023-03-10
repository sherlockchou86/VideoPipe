
#include "vp_mask_rcnn_detector_node.h"


namespace vp_nodes {
    vp_mask_rcnn_detector_node::vp_mask_rcnn_detector_node(std::string node_name, 
                            std::string model_path, 
                            std::string model_config_path, 
                            std::string labels_path, 
                            int input_width, 
                            int input_height, 
                            int batch_size,
                            int class_id_offset,
                            float score_threshold,
                            float scale,
                            cv::Scalar mean,
                            cv::Scalar std,
                            bool swap_rb):
                            vp_primary_infer_node(node_name, model_path, model_config_path, labels_path, input_width, input_height, batch_size, class_id_offset, scale, mean, std, swap_rb),
                            score_threshold(score_threshold) {
        this->initialized();
    }
    
    vp_mask_rcnn_detector_node::~vp_mask_rcnn_detector_node() {

    }


    void vp_mask_rcnn_detector_node::infer(const cv::Mat& blob_to_infer, std::vector<cv::Mat>& raw_outputs) {
        // blob_to_infer is a 4D matrix
        // the first dim is number of batch, MUST be 1
        assert(blob_to_infer.dims == 4);
        assert(blob_to_infer.size[0] == 1);
        assert(!net.empty());

        net.setInput(blob_to_infer);
        net.forward(raw_outputs, out_names);
    }


    void vp_mask_rcnn_detector_node::preprocess(const std::vector<cv::Mat>& mats_to_infer, cv::Mat& blob_to_infer) {
        // ignore preprocess logic in base class
        cv::dnn::blobFromImages(mats_to_infer, blob_to_infer, 1.0, cv::Size(), cv::Scalar(), true, false);
    }

    void vp_mask_rcnn_detector_node::postprocess(const std::vector<cv::Mat>& raw_outputs, const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) {
        auto& frame_meta = frame_meta_with_batch[0];
        
        auto outDetections = raw_outputs[0];
        auto outMasks = raw_outputs[1];
    
        // Output size of masks is NxCxHxW where
        // N - number of detected boxes
        // C - number of classes (excluding background)
        // HxW - segmentation shape
        const int numDetections = outDetections.size[2];
        const int numClasses = outMasks.size[1];
    
        auto outDetections_ = outDetections.reshape(1, outDetections.total() / 7);
        auto& frame = frame_meta->frame;
        for (int i = 0; i < numDetections; ++i) {
            float score = outDetections_.at<float>(i, 2);
            if (score > score_threshold) {
                // Extract the bounding box
                int classId = static_cast<int>(outDetections_.at<float>(i, 1));
                int left = static_cast<int>(frame.cols * outDetections_.at<float>(i, 3));
                int top = static_cast<int>(frame.rows * outDetections_.at<float>(i, 4));
                int right = static_cast<int>(frame.cols * outDetections_.at<float>(i, 5));
                int bottom = static_cast<int>(frame.rows * outDetections_.at<float>(i, 6));
    
                left = max(0, min(left, frame.cols - 1));
                top = max(0, min(top, frame.rows - 1));
                right = max(0, min(right, frame.cols - 1));
                bottom = max(0, min(bottom, frame.rows - 1));

                auto label = (labels.size() < classId + 1) ? "" : labels[classId];

                // Extract the mask for the object
                cv::Mat objectMask(outMasks.size[2], outMasks.size[3], CV_32F, outMasks.ptr<float>(i, classId));
                
                classId += class_id_offset;
                auto target = std::make_shared<vp_objects::vp_frame_target>(left, top, right - left, bottom - top, classId, score, frame_meta->frame_index, frame_meta->channel_index, label);
                target->mask = objectMask;

                frame_meta->targets.push_back(target);
            }
        }
    }
}