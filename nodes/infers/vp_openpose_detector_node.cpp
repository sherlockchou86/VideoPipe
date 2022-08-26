#include <set>

#include "vp_openpose_detector_node.h"

namespace vp_nodes {
        
    vp_openpose_detector_node::vp_openpose_detector_node(std::string node_name, 
                                                        std::string model_path, 
                                                        std::string model_config_path, 
                                                        std::string labels_path, 
                                                        int input_width, 
                                                        int input_height, 
                                                        int batch_size,
                                                        int class_id_offset,
                                                        float score_threshold,
                                                        vp_objects::vp_pose_type type,
                                                        float scale,
                                                        cv::Scalar mean,
                                                        cv::Scalar std,
                                                        bool swap_rb):
                                                        vp_primary_infer_node(node_name, 
                                                                            model_path, 
                                                                            model_config_path, 
                                                                            labels_path, 
                                                                            input_width, 
                                                                            input_height, 
                                                                            batch_size, 
                                                                            class_id_offset, 
                                                                            scale, mean, 
                                                                            std, 
                                                                            swap_rb),
                                                        score_threshold(score_threshold), type(type) {
        this->initialized();
    }
    
    vp_openpose_detector_node::~vp_openpose_detector_node() {

    }
    
    void vp_openpose_detector_node::postprocess(const std::vector<cv::Mat>& raw_outputs, const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) {
        // make sure heads of output are not zero
        assert(raw_outputs.size() > 0);
        // as it just one head of output
        auto& netOutputBlobs = raw_outputs[0];
        // batch, number of heatmaps, h, w
        assert(netOutputBlobs.dims == 4);

        auto start1 = std::chrono::system_clock::now();

        int netOutputBlob_dims[3] = {netOutputBlobs.size[1], netOutputBlobs.size[2], netOutputBlobs.size[3]};
        // scan batch
        for (int b = 0; b < netOutputBlobs.size[0]; b++) {
            auto& frame_meta = frame_meta_with_batch[b];
            //auto h_scale = frame_meta->frame.rows * 1.0 / netOutputBlobs.size[2];
            //auto w_scale = frame_meta->frame.cols * 1.0 / netOutputBlobs.size[3];

            cv::Mat netOutputBlob = cv::Mat(3, netOutputBlob_dims, CV_32F, const_cast<uchar*>(netOutputBlobs.ptr(b)));
            
            int keyPointId = 0;
            std::vector<std::vector<KeyPoint>> detectedKeypoints;
            std::vector<KeyPoint> keyPointsList;

            auto start2 = std::chrono::system_clock::now();
            std::vector<cv::Mat> netOutputParts;
            splitNetOutputBlobToParts(netOutputBlob, cv::Size(frame_meta->frame.cols, frame_meta->frame.rows), netOutputParts);
            std::cout << "split cost-----" << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now()-start2).count() << "ms" << std::endl;
            

            auto start3 = std::chrono::system_clock::now();
            for(int i = 0; i < points_map.at(type); ++i) {
                std::vector<KeyPoint> keyPoints;
                getKeyPoints(netOutputParts[i], score_threshold, keyPoints);

                for(int i = 0; i< keyPoints.size(); ++i, ++keyPointId) {
                    keyPoints[i].id = keyPointId;
                }

                detectedKeypoints.push_back(keyPoints);
                keyPointsList.insert(keyPointsList.end(), keyPoints.begin(), keyPoints.end());
            }

            std::vector<std::vector<ValidPair>> validPairs;
            std::set<int> invalidPairs;
            getValidPairs(netOutputParts, detectedKeypoints, validPairs, invalidPairs);

            std::vector<std::vector<int>> personwiseKeypoints;
            getPersonwiseKeypoints(validPairs, invalidPairs, personwiseKeypoints);

            // insert pose targets back into frame meta
            for(int n = 0; n < personwiseKeypoints.size(); ++n) {
                std::vector<vp_objects::vp_pose_keypoint> kps;
                for (int i = 0; i < personwiseKeypoints[n].size(); i++) {
                    auto index = personwiseKeypoints[n][i];
                    if (index != -1) {                
                        auto& p = keyPointsList[index];
                        kps.push_back(vp_objects::vp_pose_keypoint {i, p.point.x, p.point.y, p.probability});
                    }
                    else {
                        // point not detected
                        kps.push_back(vp_objects::vp_pose_keypoint {i, -1, -1, 0});
                    }
                }
                
                auto pose_target = std::make_shared<vp_objects::vp_frame_pose_target>(type, kps);
                frame_meta->pose_targets.push_back(pose_target);
            }
            std::cout << "parse cost-----" << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now()-start3).count() << "ms" << std::endl;
        }

        std::cout << "postprocess cost-----" << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now()-start1).count() << "ms" << std::endl;
    }
}