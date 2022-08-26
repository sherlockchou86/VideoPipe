
#pragma once

#include <set>
#include <map>
#include <cmath>
#include <opencv2/imgproc.hpp>

#include "../vp_primary_infer_node.h"
#include "../../objects/vp_frame_pose_target.h"


namespace vp_nodes {
    // body keypoints detector using openpose
    // https://github.com/CMU-Perceptual-Computing-Lab/openpose
    class vp_openpose_detector_node: public vp_primary_infer_node
    {
    private:
        float score_threshold;
        // pose type (model type)
        vp_objects::vp_pose_type type;

        // map indexs for PAFs
        const std::map<vp_objects::vp_pose_type, std::vector<std::pair<int,int>>> mapIdxes_map = {
            {vp_objects::vp_pose_type::body_25, {{0,1}, {14,15}, {22,23}, {16,17}, {18,19}, {24,25}, {26,27}, {6,7}, {2,3}, {4,5}, {8,9}, {10,11}, {12,13}, {30,31}, {32,33}, {36,37}, {34,35}, 
                                                {38,39}, {40,41} ,{42,43}, {44,45}, {46,47}, {48,49}, {50,51}}},
            {vp_objects::vp_pose_type::coco, {{12,13}, {20,21}, {14,15}, {16,17}, {22,23}, {24,25}, {0,1}, 
                                            {2,3}, {4,5}, {6,7}, {8,9}, {10,11}, {28,29}, {30,31}, {34,35}, {32,33}, {36,37}, {18,19}, {26,27}}},
            {vp_objects::vp_pose_type::mpi_15, {{0,1}, {2,3}, {4,5}, {6,7}, {8,9}, {10,11}, {12,13}, {14,15}, {16,17}, {18,19}, {20,21}, {22,23}, {24,25}, {26,27}}},
            {vp_objects::vp_pose_type::hand, std::vector<std::pair<int,int>>()},
            {vp_objects::vp_pose_type::face, std::vector<std::pair<int,int>>()}
        };

        // pose pairs for PAFs
        const std::map<vp_objects::vp_pose_type, std::vector<std::pair<int,int>>> posePairs_map = {
            {vp_objects::vp_pose_type::body_25, {{1,8}, {1,2}, {1,5}, {2,3}, {3,4}, {5,6}, {6,7}, {8,9}, 
                                                {9,10}, {10,11}, {8,12}, {12,13}, {13,14}, {1,0}, {0,15}, {15,17}, {0,16}, {16,18}, 
                                                {14,19}, {19,20}, {14,21}, {11,22}, {22,23}, {11,24}}},
            {vp_objects::vp_pose_type::coco, {{1,2}, {1,5}, {2,3}, {3,4}, {5,6}, {6,7},
                                            {1,8}, {8,9}, {9,10}, {1,11}, {11,12}, {12,13},
                                            {1,0}, {0,14}, {14,16}, {0,15}, {15,17}, {2,16},
                                            {5,17}}},
            {vp_objects::vp_pose_type::mpi_15, {{0,1}, {1,2}, {2,3}, {3,4}, {1,5}, {5,6}, {6,7}, {1,14}, {14,8}, {8,9}, {9,10}, {14,11}, {11,12}, {12,13}, {0, 2}, {0, 5}}},
            {vp_objects::vp_pose_type::hand, std::vector<std::pair<int,int>>()},
            {vp_objects::vp_pose_type::face, std::vector<std::pair<int,int>>()}
        };

        // points count for each type of pose model
        const std::map<vp_objects::vp_pose_type, int> points_map = {
            {vp_objects::vp_pose_type::body_25, 25},
            {vp_objects::vp_pose_type::coco, 18},
            {vp_objects::vp_pose_type::mpi_15, 15},
            {vp_objects::vp_pose_type::hand, 0},
            {vp_objects::vp_pose_type::face, 0}
        };

        // for postprocess purpose
        struct ValidPair{
            ValidPair(int aId,int bId,float score) {
                this->aId = aId;
                this->bId = bId;
                this->score = score;
            }

            int aId;
            int bId;
            float score;
        };
        struct KeyPoint{
            KeyPoint(cv::Point point, float probability) {
                this->id = -1;
                this->point = point;
                this->probability = probability;
            }

            int id;
            cv::Point point;
            float probability;
        };
        // for postprocess purpose end

        void getKeyPoints(cv::Mat& probMap, double threshold, std::vector<KeyPoint>& keyPoints) {
            cv::Mat smoothProbMap;
            cv::GaussianBlur(probMap, smoothProbMap, cv::Size( 3, 3 ), 0, 0);

            cv::Mat maskedProbMap;
            cv::threshold(smoothProbMap, maskedProbMap, threshold, 255, cv::THRESH_BINARY);

            maskedProbMap.convertTo(maskedProbMap, CV_8U, 1);

            std::vector<std::vector<cv::Point> > contours;
            cv::findContours(maskedProbMap, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

            for(int i = 0; i < contours.size();++i) {
                cv::Mat blobMask = cv::Mat::zeros(smoothProbMap.rows, smoothProbMap.cols, smoothProbMap.type());
                cv::fillConvexPoly(blobMask, contours[i], cv::Scalar(1));

                double maxVal;
                cv::Point maxLoc;

                cv::minMaxLoc(smoothProbMap.mul(blobMask), 0, &maxVal, 0, &maxLoc);
                keyPoints.push_back(KeyPoint(maxLoc, probMap.at<float>(maxLoc.y, maxLoc.x)));
            }
        }

        void splitNetOutputBlobToParts(cv::Mat& netOutputBlob, const cv::Size& targetSize, std::vector<cv::Mat>& netOutputParts) {
            int nParts = netOutputBlob.size[0];
            int h = netOutputBlob.size[1];
            int w = netOutputBlob.size[2];

            for(int i = 0; i< nParts; ++i) {
                cv::Mat part(h, w, CV_32F, netOutputBlob.ptr(i));

                cv::Mat resizedPart;

                cv::resize(part, resizedPart, targetSize);

                netOutputParts.push_back(resizedPart);
            }
        }

        void populateInterpPoints(const cv::Point& a,const cv::Point& b,int numPoints,std::vector<cv::Point>& interpCoords){
            float xStep = ((float)(b.x - a.x)) / (float)(numPoints - 1);
            float yStep = ((float)(b.y - a.y)) / (float)(numPoints - 1);
            interpCoords.push_back(a);

            for(int i = 1; i< numPoints - 1; ++i) {
                interpCoords.push_back(cv::Point(a.x + xStep * i, a.y + yStep * i));
            }
            interpCoords.push_back(b);
        }

        void getValidPairs(const std::vector<cv::Mat>& netOutputParts,
                        const std::vector<std::vector<KeyPoint>>& detectedKeypoints,
                        std::vector<std::vector<ValidPair>>& validPairs,
                        std::set<int>& invalidPairs) {

            int nInterpSamples = 10;
            float confTh = 0.7;

            for(int k = 0; k < mapIdxes_map.at(type).size(); ++k) {
                //A->B constitute a limb
                cv::Mat pafA = netOutputParts[mapIdxes_map.at(type)[k].first + points_map.at(type) + 1];
                cv::Mat pafB = netOutputParts[mapIdxes_map.at(type)[k].second + points_map.at(type) + 1];

                //Find the keypoints for the first and second limb
                const std::vector<KeyPoint>& candA = detectedKeypoints[posePairs_map.at(type)[k].first];
                const std::vector<KeyPoint>& candB = detectedKeypoints[posePairs_map.at(type)[k].second];

                int nA = candA.size();
                int nB = candB.size();

                /*
                # If keypoints for the joint-pair is detected
                # check every joint in candA with every joint in candB
                # Calculate the distance vector between the two joints
                # Find the PAF values at a set of interpolated points between the joints
                # Use the above formula to compute a score to mark the connection valid
                */

                if(nA != 0 && nB != 0){
                    std::vector<ValidPair> localValidPairs;

                    for(int i = 0; i< nA; ++i) {
                        int maxJ = -1;
                        float maxScore = -1;
                        bool found = false;

                        for(int j = 0; j < nB; ++j) {
                            std::pair<float, float> distance(candB[j].point.x - candA[i].point.x, candB[j].point.y - candA[i].point.y);

                            float norm = std::sqrt(distance.first * distance.first + distance.second * distance.second);

                            if(!norm) {
                                continue;
                            }

                            distance.first /= norm;
                            distance.second /= norm;

                            //Find p(u)
                            std::vector<cv::Point> interpCoords;
                            populateInterpPoints(candA[i].point, candB[j].point, nInterpSamples, interpCoords);
                            //Find L(p(u))
                            std::vector<std::pair<float, float>> pafInterp;
                            for(int l = 0; l < interpCoords.size();++l) {
                                pafInterp.push_back(
                                    std::pair<float,float>(
                                        pafA.at<float>(interpCoords[l].y, interpCoords[l].x),
                                        pafB.at<float>(interpCoords[l].y, interpCoords[l].x)
                                    ));
                            }

                            std::vector<float> pafScores;
                            float sumOfPafScores = 0;
                            int numOverTh = 0;
                            for(int l = 0; l< pafInterp.size(); ++l){
                                float score = pafInterp[l].first * distance.first + pafInterp[l].second * distance.second;
                                sumOfPafScores += score;
                                if(score > score_threshold){
                                    ++numOverTh;
                                }

                                pafScores.push_back(score);
                            }

                            float avgPafScore = sumOfPafScores / ((float)pafInterp.size());

                            if(((float)numOverTh)/((float)nInterpSamples) > confTh){
                                if(avgPafScore > maxScore) {
                                    maxJ = j;
                                    maxScore = avgPafScore;
                                    found = true;
                                }
                            }
                        }/* j */

                        if(found) {
                            localValidPairs.push_back(ValidPair(candA[i].id, candB[maxJ].id, maxScore));
                        }

                    }/* i */

                    validPairs.push_back(localValidPairs);

                } else {
                    invalidPairs.insert(k);
                    validPairs.push_back(std::vector<ValidPair>());
                }
            }/* k */
        }

        void getPersonwiseKeypoints(const std::vector<std::vector<ValidPair>>& validPairs,
                                    const std::set<int>& invalidPairs,
                                    std::vector<std::vector<int>>& personwiseKeypoints) {
            for(int k = 0; k < mapIdxes_map.at(type).size(); ++k) {
                if(invalidPairs.find(k) != invalidPairs.end()) {
                    continue;
                }

                const std::vector<ValidPair>& localValidPairs(validPairs[k]);

                int indexA(posePairs_map.at(type)[k].first);
                int indexB(posePairs_map.at(type)[k].second);

                for(int i = 0; i< localValidPairs.size(); ++i) {
                    bool found = false;
                    int personIdx = -1;

                    for(int j = 0; !found && j < personwiseKeypoints.size();++j) {
                        if(indexA < personwiseKeypoints[j].size() &&
                            personwiseKeypoints[j][indexA] == localValidPairs[i].aId) {
                            personIdx = j;
                            found = true;
                        }
                    }/* j */

                    if(found) {
                        personwiseKeypoints[personIdx].at(indexB) = localValidPairs[i].bId;
                    } 
                    else if(k < points_map.at(type) - 1) {
                        std::vector<int> lpkp(std::vector<int>(points_map.at(type), -1));

                        lpkp.at(indexA) = localValidPairs[i].aId;
                        lpkp.at(indexB) = localValidPairs[i].bId;

                        personwiseKeypoints.push_back(lpkp);
                    }

                }/* i */
            }/* k */
        }

    protected:
        virtual void postprocess(const std::vector<cv::Mat>& raw_outputs, const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) override;
    public:
        vp_openpose_detector_node(std::string node_name, 
                            std::string model_path, 
                            std::string model_config_path = "", 
                            std::string labels_path = "", 
                            int input_width = 368, 
                            int input_height = 368, 
                            int batch_size = 1,
                            int class_id_offset = 0,
                            float score_threshold = 0.1,
                            vp_objects::vp_pose_type type = vp_objects::vp_pose_type::coco,
                            float scale = 1 / 255.0,
                            cv::Scalar mean = cv::Scalar(0),
                            cv::Scalar std = cv::Scalar(1),
                            bool swap_rb = false);
        ~vp_openpose_detector_node();
    };
}