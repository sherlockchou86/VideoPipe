#include <fstream>
#include "vp_lane_osd_node.h"
#include "../../utils/vp_utils.h"

namespace vp_nodes {
        
    vp_lane_osd_node::vp_lane_osd_node(std::string node_name): vp_node(node_name) {
        this->initialized();
    }
    
    vp_lane_osd_node::~vp_lane_osd_node() {
        deinitialized();
    }

    std::shared_ptr<vp_objects::vp_meta> vp_lane_osd_node::handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) {
        // operations on osd_frame
        if (meta->osd_frame.empty()) {
            meta->osd_frame = meta->frame.clone();
        }
        auto& canvas = meta->osd_frame;

        if (meta->mask.empty()) {
            return meta;
        }
        
        // resize mask to the same size of canvas
        cv::Mat mask(meta->mask.size[2], meta->mask.size[3], CV_32FC1, meta->mask.data);
        cv::Mat mask_big;
        cv::resize(mask, mask_big, canvas.size());
        cv::threshold(mask_big, mask_big, 0.5, 1, cv::THRESH_BINARY);

        auto kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(11, 11));
        cv::erode(mask_big, mask_big, kernel);
        mask_big.convertTo(mask_big, CV_8U, 255);

        // merge mask and canvas
        for (int y = 0; y < canvas.rows; ++y) {
            for (int x = 0; x < canvas.cols; ++x) {
                canvas.at<cv::Vec3b>(y, x)[2] = mask_big.at<uchar>(y, x);

                canvas.at<cv::Vec3b>(y, x)[1] = cv::saturate_cast<uchar>(
                    canvas.at<cv::Vec3b>(y, x)[1] * 0.8 + mask_big.at<uchar>(y, x) * 0.2);

                canvas.at<cv::Vec3b>(y, x)[0] = cv::saturate_cast<uchar>(
                    canvas.at<cv::Vec3b>(y, x)[0] * 0.8 + mask_big.at<uchar>(y, x) * 0.2);
            }
        }

        return meta;
    }
}