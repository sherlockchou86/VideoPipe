#include "vp_sort_track_node.h"

namespace vp_nodes {
        
    vp_sort_track_node::vp_sort_track_node(std::string node_name, 
                                            vp_track_for track_for):
                                            vp_track_node(node_name, track_for) {
        this->initialized();
        KalmanTracker::kf_count = 0;
    }
    
    vp_sort_track_node::~vp_sort_track_node() {
        deinitialized();
    }

    void vp_sort_track_node::track(int channel_index, const std::vector<vp_objects::vp_rect>& target_rects, 
                    const std::vector<std::vector<float>>& target_embeddings, 
                    std::vector<int>& track_ids) {
        // fill track_ids according to target_rects (target_embeddings ignored)
		track_ids.resize(target_rects.size());
		for (auto&  item : track_ids) {
			item = -1;
		}

		// check if trackers are initialized or not for specific channel
		if (all_trackers.count(channel_index) == 0) {
			all_trackers[channel_index] = std::vector<KalmanTracker>();
			VP_INFO(vp_utils::string_format("[%s] initialize kalmantracker the first time for channel %d", node_name.c_str(), channel_index));
		}
		// track on specific channel
		auto& trackers = all_trackers[channel_index];

        if (trackers.empty()) {
            /* first frame*/
            for (unsigned int i = 0; i < target_rects.size(); i++) {
				KalmanTracker trk = KalmanTracker(cv::Rect_<float>(target_rects[i].x, target_rects[i].y, target_rects[i].width, target_rects[i].height));
				trackers.push_back(trk);
			}
            return;
        }
        //3.1. get predicted locations from existing trackers.
        predictedBoxes.clear();
		for (auto it = trackers.begin(); it != trackers.end();) {
			Rect_<float> pBox = (*it).predict();
			if (pBox.x >= 0 && pBox.y >= 0) {
				predictedBoxes.push_back(pBox);
				it++;
			}
			else {
				it = trackers.erase(it);
			}
		}
        
        // 3.2. associate detections to tracked object (both represented as bounding boxes)
		// dets : detFrameData[fi]
		auto trkNum = predictedBoxes.size();
		auto detNum = target_rects.size();

		iouMatrix.clear();
		iouMatrix.resize(trkNum, vector<double>(detNum, 0));
        
		// compute iou matrix as a distance matrix
        for (unsigned int i = 0; i < trkNum; i++)  {
			for (unsigned int j = 0; j < detNum; j++) {
				// use 1-iou because the hungarian algorithm computes a minimum-cost assignment.
				iouMatrix[i][j] = 1 - GetIOU(predictedBoxes[i], cv::Rect_<float>(target_rects[j].x, target_rects[j].y, target_rects[j].width, target_rects[j].height));
			}
		}

        // solve the assignment problem using hungarian algorithm.
		// the resulting assignment is [track(prediction) : detection], with len=preNum
		HungarianAlgorithm HungAlgo;
		assignment.clear();
		HungAlgo.Solve(iouMatrix, assignment);

		// find matches, unmatched_detections and unmatched_predictions
		unmatchedTrajectories.clear();
		unmatchedDetections.clear();
		allItems.clear();
		matchedItems.clear();

		// there are unmatched detections
        if (detNum > trkNum) {
			for (unsigned int n = 0; n < detNum; n++)
				allItems.insert(n);

			for (unsigned int i = 0; i < trkNum; ++i)
				matchedItems.insert(assignment[i]);

			set_difference(allItems.begin(), allItems.end(),
				matchedItems.begin(), matchedItems.end(),
				insert_iterator<set<int>>(unmatchedDetections, unmatchedDetections.begin()));
		}
		// there are unmatched trajectory/predictions
		else if (detNum < trkNum) {
			for (unsigned int i = 0; i < trkNum; ++i)
				if (assignment[i] == -1) // unassigned label will be set as -1 in the assignment algorithm
					unmatchedTrajectories.insert(i);
		}
		else {

		}
        
        // filter out matched with low IOU
		matchedPairs.clear();
		for (unsigned int i = 0; i < trkNum; ++i) {
			if (assignment[i] == -1) // pass over invalid values
				continue;
			if (1 - iouMatrix[i][assignment[i]] < iouThreshold) {
				unmatchedTrajectories.insert(i);
				unmatchedDetections.insert(assignment[i]);
			}
			else {
				matchedPairs.push_back(cv::Point(i, assignment[i]));
			}
		}


        // 3.3. updating trackers
		// update matched trackers with assigned detections.
		// each prediction is corresponding to a tracker
		int detIdx, trkIdx;
		for (unsigned int i = 0; i < matchedPairs.size(); i++) {
			trkIdx = matchedPairs[i].x;
			detIdx = matchedPairs[i].y;
			trackers[trkIdx].update(cv::Rect_<float>(target_rects[detIdx].x, 
                                                    target_rects[detIdx].y, 
                                                    target_rects[detIdx].width, 
                                                    target_rects[detIdx].height));
		}

		// create and initialise new trackers for unmatched detections
		for (auto& umd : unmatchedDetections) {
			KalmanTracker tracker = KalmanTracker(cv::Rect_<float>(target_rects[umd].x, 
                                                                   target_rects[umd].y,
                                                                   target_rects[umd].width, 
                                                                   target_rects[umd].height));
			trackers.push_back(tracker);
		}

        // get trackers' output
		frameTrackingResult.clear();
		for (auto it = trackers.begin(); it != trackers.end();) {
			if (((*it).m_time_since_update < 1) &&
				((*it).m_hit_streak >= min_hits)) {
				TrackingBox res;
				res.box = (*it).get_state();
				res.id = (*it).m_id + 1;
				//res.frame = meta->frame_index;
				frameTrackingResult.push_back(res);
				it++;
			}
			else
				it++;

			// remove dead tracklet
			if (it != trackers.end() && (*it).m_time_since_update > max_age)
				it = trackers.erase(it);
		}

        for (const auto& tb : frameTrackingResult) {
			// id and box need to correspond
			for (int i = 0; i < target_rects.size(); ++i) {
				/* code */
				if(GetIOU(cv::Rect_<float>(target_rects[i].x, 
											target_rects[i].y, 
											target_rects[i].width, 
											target_rects[i].height),
						   cv::Rect_<float>(tb.box.x, 
											tb.box.y, 
											tb.box.width, 
											tb.box.height)) > 0.8) {
				track_ids[i] = tb.id;
				}
			}
        }
        return;
    }

    double vp_sort_track_node::GetIOU(cv::Rect_<float> bb_test, cv::Rect_<float> bb_gt){
         float in = (bb_test & bb_gt).area();
        float un = bb_test.area() + bb_gt.area() - in;

        if (un < DBL_EPSILON)
            return 0;

        return (double)(in / un);
    }
}