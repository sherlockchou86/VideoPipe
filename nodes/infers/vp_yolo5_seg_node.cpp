
#include <cmath>
#include <algorithm>
#include <numeric> 
#include "vp_yolo5_seg_node.h"


namespace vp_nodes {
    vp_yolo5_seg_node::vp_yolo5_seg_node(std::string node_name, 
                            std::string model_path, 
                            std::string labels_path, 
                            int input_width, 
                            int input_height, 
                            int batch_size,
                            int class_id_offset,
                            float conf_threshold,
                            float iou_threshold):
                            vp_primary_infer_node(node_name, model_path, "", labels_path, input_width, input_height, batch_size, class_id_offset),
                            conf_threshold(conf_threshold),
                            iou_threshold(iou_threshold) {
        this->initialized();
    }
    
    vp_yolo5_seg_node::~vp_yolo5_seg_node() {
        deinitialized();
    }

    void vp_yolo5_seg_node::prepare(const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch, std::vector<cv::Mat>& mats_to_infer) {
        for (auto& i: frame_meta_with_batch) {
            auto letterboxed = letterbox(i->frame, cv::Size(input_width, input_height), scale_ratio, padding);
            mats_to_infer.push_back(letterboxed);
        }
    }

    void vp_yolo5_seg_node::infer(const cv::Mat& blob_to_infer, std::vector<cv::Mat>& raw_outputs) {
        // blob_to_infer is a 4D matrix
        // the first dim is number of batch, MUST be 1
        assert(blob_to_infer.dims == 4);
        assert(blob_to_infer.size[0] == 1);
        assert(!net.empty());

        net.setInput(blob_to_infer);
        net.forward(raw_outputs, net.getUnconnectedOutLayersNames());
    }

    void vp_yolo5_seg_node::preprocess(const std::vector<cv::Mat>& mats_to_infer, cv::Mat& blob_to_infer) {
        // ignore preprocess logic in base class
        cv::dnn::blobFromImages(mats_to_infer, blob_to_infer, 1.0 / 255.0, cv::Size(), cv::Scalar(), true, false);
    }

    void vp_yolo5_seg_node::postprocess(const std::vector<cv::Mat>& raw_outputs, const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) {
        auto& frame_meta = frame_meta_with_batch[0];
        auto& frame = frame_meta->frame;
        auto pred = raw_outputs[0]; // [1, N, 4+1+NUM_CLASSES+32]
        auto proto = raw_outputs[1]; // [1, 32, 96, 160]

        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        std::vector<int> class_ids;
        std::vector<cv::Mat> mask_coeffs;

        int num_boxes = pred.size[1];
        float* data = (float*)pred.data;

        // decode
        for (int i = 0; i < num_boxes; ++i) {
            float obj_conf = data[4];
            if (obj_conf < conf_threshold) {
                data += (5 + labels.size() + MASK_CHANNELS);
                continue;
            }

            int class_id = -1;
            float max_cls_conf = 0;
            for (int c = 0; c < labels.size(); ++c) {
                float cls_conf = data[5 + c];
                if (cls_conf > max_cls_conf) {
                    max_cls_conf = cls_conf;
                    class_id = c;
                }
            }

            float total_conf = obj_conf * max_cls_conf;
            if (total_conf < conf_threshold || class_id == -1) {
                data += (5 + labels.size() + MASK_CHANNELS);
                continue;
            }

            float cx = data[0];
            float cy = data[1];
            float w  = data[2];
            float h  = data[3];

            float x1 = cx - w * 0.5f;
            float y1 = cy - h * 0.5f;
            float x2 = cx + w * 0.5f;
            float y2 = cy + h * 0.5f;

            if (x2 <= x1 || y2 <= y1 || w <= 0 || h <= 0) {
                data += (5 + labels.size() + MASK_CHANNELS);
                continue;
            }

            float x_scale = scale_ratio.at<float>(0);
            float y_scale = scale_ratio.at<float>(1);
            int orig_x1 = cvRound((x1 - padding.x) * x_scale);
            int orig_y1 = cvRound((y1 - padding.y) * y_scale);
            int orig_x2 = cvRound((x2 - padding.x) * x_scale);
            int orig_y2 = cvRound((y2 - padding.y) * y_scale);

            orig_x1 = max(0, orig_x1);
            orig_y1 = max(0, orig_y1);
            orig_x2 = min(frame.cols, orig_x2);
            orig_y2 = min(frame.rows, orig_y2);

            if (orig_x2 <= orig_x1 || orig_y2 <= orig_y1) {
                data += (5 + labels.size() + MASK_CHANNELS);
                continue;
            }

            cv::Rect box(orig_x1, orig_y1, orig_x2 - orig_x1, orig_y2 - orig_y1);
            boxes.push_back(box);
            confidences.push_back(total_conf);
            class_ids.push_back(class_id);

            cv::Mat coeff(1, MASK_CHANNELS, CV_32F);
            for (int m = 0; m < MASK_CHANNELS; ++m) {
                coeff.at<float>(m) = data[5 + labels.size() + m];
            }
            mask_coeffs.push_back(coeff);

            data += (5 + labels.size() + MASK_CHANNELS);
        }

        std::vector<int> final_indices;
        std::unordered_map<int, std::vector<int>> class_boxes;
        for (int i = 0; i < boxes.size(); ++i) {
            class_boxes[class_ids[i]].push_back(i);
        }

        // NMS
        for (auto& kv : class_boxes) {
            const std::vector<int>& indices_in_class = kv.second;
            std::vector<cv::Rect> boxes_cls;
            std::vector<float> scores_cls;
            for (int idx : indices_in_class) {
                boxes_cls.push_back(boxes[idx]);
                scores_cls.push_back(confidences[idx]);
            }

            std::vector<int> nms_indices = nms(boxes_cls, scores_cls, iou_threshold);
            for (int local_idx : nms_indices) {
                final_indices.push_back(indices_in_class[local_idx]);
            }
        }

        // create target push back to frame meta
        for (int idx : final_indices) {
            cv::Rect box = boxes[idx];
            float conf = confidences[idx];
            int cls = class_ids[idx];

            auto label = (labels.size() < cls + 1) ? "" : labels[cls];            
            cls += class_id_offset;
            auto target = std::make_shared<vp_objects::vp_frame_target>(box.x, box.y, box.width, box.height, cls, conf, frame_meta->frame_index, frame_meta->channel_index, label);

            // try to extract mask
            auto local_mask = process_mask(proto, mask_coeffs[idx], box,
                                          cv::Size(input_width, input_height),
                                          padding, scale_ratio.at<float>(0), scale_ratio.at<float>(1));
            target->mask = local_mask;
            
            frame_meta->targets.push_back(target);
        }

        std::cout << "Detected: " << final_indices.size() << " instances" << std::endl;
    }

    cv::Mat vp_yolo5_seg_node::letterbox(const cv::Mat& src, cv::Size target_size, cv::Mat& scale_ratio, cv::Point& padding) {
        float scale = min(static_cast<float>(target_size.height) / src.rows,
                        static_cast<float>(target_size.width) / src.cols);
        int new_width = cvRound(src.cols * scale);
        int new_height = cvRound(src.rows * scale);

        cv::Mat resized;
        resize(src, resized, cv::Size(new_width, new_height), 0, 0, cv::INTER_LINEAR);

        cv::Mat letterboxed(target_size, CV_8UC3, cv::Scalar(114, 114, 114));
        int top = (target_size.height - new_height) / 2;
        int left = (target_size.width - new_width) / 2;
        cv::Rect roi(left, top, new_width, new_height);
        resized.copyTo(letterboxed(roi));

        scale_ratio = cv::Mat(1, 2, CV_32F);
        scale_ratio.at<float>(0) = 1.0f / scale;  // x_scale
        scale_ratio.at<float>(1) = 1.0f / scale;  // y_scale

        padding = cv::Point(left, top);
        return letterboxed;
    }

    // Sigmoid
    cv::Mat vp_yolo5_seg_node::sigmoid(const cv::Mat& x) {
        cv::Mat result;
        cv::exp(-x, result);
        result = 1.0f / (1.0f + result);
        return result;
    }

    cv::Mat vp_yolo5_seg_node::process_mask(const cv::Mat& proto,           // [1, 32, H/4, W/4]
                    const cv::Mat& coeffs,          // [1, 32]
                    const cv::Rect& bbox,           
                    const cv::Size& input_size,     // e.g., 640x384
                    const cv::Point& padding,
                    const float x_scale,
                    const float y_scale) {

        // 1. 获取 proto 尺寸 [1,32,96,160] → h=96, w=160
        int h_proto = proto.size[2];
        int w_proto = proto.size[3];

        // 2. 重塑为 [32, h, w]
        cv::Mat proto_3d = proto.reshape(1, {MASK_CHANNELS, h_proto, w_proto});

        // 3. 展平为 [32, h*w]
        cv::Mat proto_flat = proto_3d.reshape(1, MASK_CHANNELS);

        // 4. 矩阵乘: [1,32] × [32, h*w] → [1, h*w]
        cv::Mat mask_raw = coeffs * proto_flat;

        // 5. 重塑为 [h, w]
        cv::Mat mask_2d = mask_raw.reshape(0, h_proto);

        // 6. Sigmoid
        cv::Mat mask_sigmoid = sigmoid(mask_2d);

        // 7. 原图 bbox → 输入图坐标（含 padding）
        float x1_in = bbox.x / x_scale + padding.x;
        float y1_in = bbox.y / y_scale + padding.y;
        float x2_in = (bbox.x + bbox.width) / x_scale + padding.x;
        float y2_in = (bbox.y + bbox.height) / y_scale + padding.y;

        // 8. 输入图 → proto 坐标（/4）
        float x1_p = x1_in / 4.0f;
        float y1_p = y1_in / 4.0f;
        float x2_p = x2_in / 4.0f;
        float y2_p = y2_in / 4.0f;

        int px1 = max(0, (int)floor(x1_p));
        int py1 = max(0, (int)floor(y1_p));
        int px2 = min(w_proto, (int)ceil(x2_p));
        int py2 = min(h_proto, (int)ceil(y2_p));

        if (px2 <= px1 || py2 <= py1) {
            return cv::Mat::zeros(bbox.size(), CV_8UC1);
        }

        // 9. 裁剪
        cv::Mat cropped = mask_sigmoid(cv::Rect(px1, py1, px2 - px1, py2 - py1)).clone();

        // 10. 上采样到 bbox 尺寸
        cv::Mat resized;
        resize(cropped, resized, bbox.size(), 0, 0, cv::INTER_LINEAR);

        // 11. 二值化
        cv::Mat mask_bin;
        threshold(resized, mask_bin, 0.5, 255, cv::THRESH_BINARY);
        mask_bin.convertTo(mask_bin, CV_8UC1);

        return mask_bin;
    }

    // NMS
    std::vector<int> vp_yolo5_seg_node::nms(const std::vector<cv::Rect>& boxes, const std::vector<float>& scores, float iou_threshold) {
        std::vector<int> indices;
        std::vector<float> areas(boxes.size());
        for (size_t i = 0; i < boxes.size(); ++i) {
            areas[i] = boxes[i].area();
        }

        std::vector<int> order(scores.size());
        iota(order.begin(), order.end(), 0);
        sort(order.begin(), order.end(), [&](int a, int b) {
            return scores[a] > scores[b];
        });

        std::vector<bool> keep(scores.size(), true);
        for (int i = 0; i < order.size(); ++i) {
            int idx = order[i];
            if (!keep[idx]) continue;
            for (int j = i + 1; j < order.size(); ++j) {
                int idx2 = order[j];
                if (!keep[idx2]) continue;
                cv::Rect intersect = boxes[idx] & boxes[idx2];
                float iou = intersect.area() / (areas[idx] + areas[idx2] - intersect.area() + 1e-6f);
                if (iou > iou_threshold) {
                    keep[idx2] = false;
                }
            }
        }

        for (int i = 0; i < keep.size(); ++i) {
            if (keep[i]) indices.push_back(i);
        }
        return indices;
    }
}