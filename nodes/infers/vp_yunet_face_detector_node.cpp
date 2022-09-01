
#include "vp_yunet_face_detector_node.h"


namespace vp_nodes {
        
    vp_yunet_face_detector_node::vp_yunet_face_detector_node(std::string node_name, 
                                                            std::string model_path, 
                                                            float score_threshold, 
                                                            float nms_threshold, 
                                                            int top_k):
                                                            vp_primary_infer_node(node_name, model_path),
                                                            scoreThreshold(score_threshold),
                                                            nmsThreshold(nms_threshold),
                                                            topK(top_k) {
        this->initialized();
    }
    
    vp_yunet_face_detector_node::~vp_yunet_face_detector_node() {

    }
    
    void vp_yunet_face_detector_node::postprocess(const std::vector<cv::Mat>& raw_outputs, const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) {
        using namespace cv;
        // 3 heads of output
        assert(raw_outputs.size() == 3);
        assert(frame_meta_with_batch.size() == 1);
        auto& frame_meta = frame_meta_with_batch[0];

        // Extract from output_blobs
        Mat loc = raw_outputs[0];
        Mat conf = raw_outputs[1];
        Mat iou = raw_outputs[2];

        // we need generate priors if input size changed or priors is not initialized
        if (loc.rows != priors.size()) {
            inputW = frame_meta->frame.cols;
            inputH = frame_meta->frame.rows;
            generatePriors();
        }
        
        assert(loc.rows == priors.size());
        assert(loc.rows == conf.rows);
        assert(loc.rows == iou.rows);

        // Decode from deltas and priors
        const std::vector<float> variance = {0.1f, 0.2f};
        float* loc_v = (float*)(loc.data);
        float* conf_v = (float*)(conf.data);
        float* iou_v = (float*)(iou.data);
        Mat faces;
        // (tl_x, tl_y, w, h, re_x, re_y, le_x, le_y, nt_x, nt_y, rcm_x, rcm_y, lcm_x, lcm_y, score)
        // 'tl': top left point of the bounding box
        // 're': right eye, 'le': left eye
        // 'nt':  nose tip
        // 'rcm': right corner of mouth, 'lcm': left corner of mouth
        Mat face(1, 15, CV_32FC1);
        for (size_t i = 0; i < priors.size(); ++i) {
            // Get score
            float clsScore = conf_v[i*2+1];
            float iouScore = iou_v[i];
            // Clamp
            if (iouScore < 0.f) {
                iouScore = 0.f;
            }
            else if (iouScore > 1.f) {
                iouScore = 1.f;
            }
            float score = std::sqrt(clsScore * iouScore);
            face.at<float>(0, 14) = score;

            // Get bounding box
            float cx = (priors[i].x + loc_v[i*14+0] * variance[0] * priors[i].width)  * inputW;
            float cy = (priors[i].y + loc_v[i*14+1] * variance[0] * priors[i].height) * inputH;
            float w  = priors[i].width  * exp(loc_v[i*14+2] * variance[0]) * inputW;
            float h  = priors[i].height * exp(loc_v[i*14+3] * variance[1]) * inputH;
            float x1 = cx - w / 2;
            float y1 = cy - h / 2;
            face.at<float>(0, 0) = x1;
            face.at<float>(0, 1) = y1;
            face.at<float>(0, 2) = w;
            face.at<float>(0, 3) = h;

            // Get landmarks
            face.at<float>(0, 4) = (priors[i].x + loc_v[i*14+ 4] * variance[0] * priors[i].width)  * inputW;  // right eye, x
            face.at<float>(0, 5) = (priors[i].y + loc_v[i*14+ 5] * variance[0] * priors[i].height) * inputH;  // right eye, y
            face.at<float>(0, 6) = (priors[i].x + loc_v[i*14+ 6] * variance[0] * priors[i].width)  * inputW;  // left eye, x
            face.at<float>(0, 7) = (priors[i].y + loc_v[i*14+ 7] * variance[0] * priors[i].height) * inputH;  // left eye, y
            face.at<float>(0, 8) = (priors[i].x + loc_v[i*14+ 8] * variance[0] * priors[i].width)  * inputW;  // nose tip, x
            face.at<float>(0, 9) = (priors[i].y + loc_v[i*14+ 9] * variance[0] * priors[i].height) * inputH;  // nose tip, y
            face.at<float>(0, 10) = (priors[i].x + loc_v[i*14+10] * variance[0] * priors[i].width)  * inputW; // right corner of mouth, x
            face.at<float>(0, 11) = (priors[i].y + loc_v[i*14+11] * variance[0] * priors[i].height) * inputH; // right corner of mouth, y
            face.at<float>(0, 12) = (priors[i].x + loc_v[i*14+12] * variance[0] * priors[i].width)  * inputW; // left corner of mouth, x
            face.at<float>(0, 13) = (priors[i].y + loc_v[i*14+13] * variance[0] * priors[i].height) * inputH; // left corner of mouth, y

            faces.push_back(face);
        }

        if (faces.rows > 1)
        {
            // Retrieve boxes and scores
            std::vector<Rect2i> faceBoxes;
            std::vector<float> faceScores;
            for (int rIdx = 0; rIdx < faces.rows; rIdx++)
            {
                faceBoxes.push_back(Rect2i(int(faces.at<float>(rIdx, 0)),
                                           int(faces.at<float>(rIdx, 1)),
                                           int(faces.at<float>(rIdx, 2)),
                                           int(faces.at<float>(rIdx, 3))));
                faceScores.push_back(faces.at<float>(rIdx, 14));
            }

            std::vector<int> keepIdx;
            dnn::NMSBoxes(faceBoxes, faceScores, scoreThreshold, nmsThreshold, keepIdx, 1.f, topK);

            // Get NMS results
            Mat nms_faces;
            for (int idx: keepIdx)
            {
                nms_faces.push_back(faces.row(idx));
            }

            // insert face target back to frame meta
            for (int i = 0; i < nms_faces.rows; i++) {
                auto x = int(nms_faces.at<float>(i, 0));
                auto y = int(nms_faces.at<float>(i, 1));
                auto w = int(nms_faces.at<float>(i, 2));
                auto h = int(nms_faces.at<float>(i, 3));
                
                // check value range
                x = std::max(x, 0);
                y = std::max(y, 0);
                w = std::min(w, frame_meta->frame.cols - x);
                h = std::min(h, frame_meta->frame.rows - y);

                auto kp1 = std::pair<int, int>(int(nms_faces.at<float>(0, 4)), int(nms_faces.at<float>(0, 5)));
                auto kp2 = std::pair<int, int>(int(nms_faces.at<float>(0, 6)), int(nms_faces.at<float>(0, 7)));
                auto kp3 = std::pair<int, int>(int(nms_faces.at<float>(0, 8)), int(nms_faces.at<float>(0, 9)));
                auto kp4 = std::pair<int, int>(int(nms_faces.at<float>(0, 10)), int(nms_faces.at<float>(0, 11)));
                auto kp5 = std::pair<int, int>(int(nms_faces.at<float>(0, 12)), int(nms_faces.at<float>(0, 13)));
                auto score = nms_faces.at<float>(0, 14);

                auto face_target = std::make_shared<vp_objects::vp_frame_face_target>(x, y, w, h, score, std::vector<std::pair<int, int>>{kp1, kp2, kp3, kp4, kp5});

                frame_meta->face_targets.push_back(face_target);
            }
            
        }
    }

    // refer to vp_infer_node::preprocess
    void vp_yunet_face_detector_node::preprocess(const std::vector<cv::Mat>& mats_to_infer, cv::Mat& blob_to_infer) {
        cv::dnn::blobFromImages(mats_to_infer, blob_to_infer);
    }
    
    // refer to vp_infer_node::infer
    void vp_yunet_face_detector_node::infer(const cv::Mat& blob_to_infer, std::vector<cv::Mat>& raw_outputs) {
        // blob_to_infer is a 4D matrix
        // the first dim is number of batch, MUST be 1
        assert(blob_to_infer.dims == 4);
        assert(blob_to_infer.size[0] == 1);
        assert(!net.empty());

        net.setInput(blob_to_infer);
        net.forward(raw_outputs, out_names);
    }

    void vp_yunet_face_detector_node::generatePriors() {
        using namespace cv;
        // Calculate shapes of different scales according to the shape of input image
        Size feature_map_2nd = {
            int(int((inputW+1)/2)/2), int(int((inputH+1)/2)/2)
        };
        Size feature_map_3rd = {
            int(feature_map_2nd.width/2), int(feature_map_2nd.height/2)
        };
        Size feature_map_4th = {
            int(feature_map_3rd.width/2), int(feature_map_3rd.height/2)
        };
        Size feature_map_5th = {
            int(feature_map_4th.width/2), int(feature_map_4th.height/2)
        };
        Size feature_map_6th = {
            int(feature_map_5th.width/2), int(feature_map_5th.height/2)
        };

        std::vector<Size> feature_map_sizes;
        feature_map_sizes.push_back(feature_map_3rd);
        feature_map_sizes.push_back(feature_map_4th);
        feature_map_sizes.push_back(feature_map_5th);
        feature_map_sizes.push_back(feature_map_6th);

        // Fixed params for generating priors
        const std::vector<std::vector<float>> min_sizes = {
            {10.0f,  16.0f,  24.0f},
            {32.0f,  48.0f},
            {64.0f,  96.0f},
            {128.0f, 192.0f, 256.0f}
        };
        CV_Assert(min_sizes.size() == feature_map_sizes.size()); // just to keep vectors in sync
        const std::vector<int> steps = { 8, 16, 32, 64 };

        // Generate priors
        priors.clear();
        for (size_t i = 0; i < feature_map_sizes.size(); ++i)
        {
            Size feature_map_size = feature_map_sizes[i];
            std::vector<float> min_size = min_sizes[i];

            for (int _h = 0; _h < feature_map_size.height; ++_h)
            {
                for (int _w = 0; _w < feature_map_size.width; ++_w)
                {
                    for (size_t j = 0; j < min_size.size(); ++j)
                    {
                        float s_kx = min_size[j] / inputW;
                        float s_ky = min_size[j] / inputH;

                        float cx = (_w + 0.5f) * steps[i] / inputW;
                        float cy = (_h + 0.5f) * steps[i] / inputH;

                        Rect2f prior = { cx, cy, s_kx, s_ky };
                        priors.push_back(prior);
                    }
                }
            }
        }
    }
}