#pragma once

#include <vector>
#include <set>
#include <map>
#include "vp_track_node.h"
#include "sort/Hungarian.h"
#include "sort/KalmanTracker.h"

namespace vp_nodes {
    // track node using sort
    class vp_sort_track_node: public vp_track_node
    {
    private:
        /* config data for sort algo */
        /* data */
        typedef struct TrackingBox
        {
            //int frame;
            int id;
            Rect_<float> box;
        }TrackingBox;

        int max_age = 1;
        int min_hits = 3;
        double iouThreshold = 0.5;
        // vector<KalmanTracker> trackers;
        std::map<int, std::vector<KalmanTracker>> all_trackers;
        std::vector<cv::Rect_<float>> predictedBoxes;
        std::vector<vector<double>> iouMatrix;
        std::vector<int> assignment;
        std::set<int> unmatchedDetections;
        std::set<int> unmatchedTrajectories;
        std::set<int> allItems;
        std::set<int> matchedItems;
        std::vector<cv::Point> matchedPairs;
        std::vector<TrackingBox> frameTrackingResult;
    private:
        double GetIOU(cv::Rect_<float> bb_test, cv::Rect_<float> bb_gt);
    protected:
        // fill track_ids using sort algo
        virtual void track(int channel_index, const std::vector<vp_objects::vp_rect>& target_rects, 
                        const std::vector<std::vector<float>>& target_embeddings, 
                        std::vector<int>& track_ids) override;
    public:
        vp_sort_track_node(std::string node_name, vp_track_for track_for = vp_track_for::NORMAL);
        virtual ~vp_sort_track_node();
    };

}

