#pragma once

#include <random>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "../vp_node.h"

namespace vp_nodes {
    // on screen display(short as osd) node.
    // mainly used to display vp_frame_pose_target on frame.
    class vp_pose_osd_node: public vp_node
    {
    private:
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

        void populateColorPalette(std::vector<cv::Scalar>& colors, int nColors) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis1(64, 200);
            std::uniform_int_distribution<> dis2(100, 255);
            std::uniform_int_distribution<> dis3(100, 255);

            for(int i = 0; i < nColors; ++i){
                colors.push_back(cv::Scalar(dis1(gen),dis2(gen),dis3(gen)));
            }
        }

    protected:
        virtual std::shared_ptr<vp_objects::vp_meta> handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) override;
        virtual std::shared_ptr<vp_objects::vp_meta> handle_control_meta(std::shared_ptr<vp_objects::vp_control_meta> meta) override;
    public:
        vp_pose_osd_node(std::string node_name);
        ~vp_pose_osd_node();
    };

}