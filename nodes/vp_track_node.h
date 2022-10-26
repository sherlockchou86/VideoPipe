
#pragma once

#include "vp_node.h"
#include "./sort/Hungarian.h"
#include "./sort/KalmanTracker.h"
#include <vector>
#include <set>
namespace vp_nodes {
    class vp_track_node: public vp_node
    {
    private:
        /* data */
        typedef struct TrackingBox
        {
            int frame;
            int id;
            Rect_<float> box;
        }TrackingBox;

        int max_age = 1;
        int min_hits = 3;
        double iouThreshold = 0.3;
        vector<KalmanTracker> trackers;
        std::vector<cv::Rect_<float>> predictedBoxes;
        std::vector<vector<double>> iouMatrix;
        std::vector<int> assignment;
        std::set<int> unmatchedDetections;
        std::set<int> unmatchedTrajectories;
        std::set<int> allItems;
        std::set<int> matchedItems;
        std::vector<cv::Point> matchedPairs;
        std::vector<TrackingBox> frameTrackingResult;

    protected:
        virtual std::shared_ptr<vp_objects::vp_meta> handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) override;
        virtual std::shared_ptr<vp_objects::vp_meta> handle_control_meta(std::shared_ptr<vp_objects::vp_control_meta> meta) override;
    public:
        vp_track_node(std::string node_name);
        ~vp_track_node();

    private:
        double GetIOU(cv::Rect_<float> bb_test, cv::Rect_<float> bb_gt);
    };
}