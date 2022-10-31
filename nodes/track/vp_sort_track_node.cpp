#include "vp_sort_track_node.h"

namespace vp_nodes {
        
    vp_sort_track_node::vp_sort_track_node(std::string node_name, 
                                            vp_track_for track_for):
                                            vp_track_node(node_name, track_for) {
        this->initialized();
        KalmanTracker::kf_count = 0;
    }
    
    vp_sort_track_node::~vp_sort_track_node()
    {
    }

    void vp_sort_track_node::track(std::vector<vp_objects::vp_rect>& target_rects, 
                    const std::vector<std::vector<float>>& target_embeddings, 
                    std::vector<int>& track_ids) {
        // fill track_ids according to target_rects (target_embeddings ignored)
        auto target_rect_dects = target_rects;
        target_rects.clear();
        if (trackers.empty())
        {
            /* first frame*/
            for (unsigned int i = 0; i < target_rect_dects.size(); i++)
			{
				KalmanTracker trk = KalmanTracker(cv::Rect_<float>(target_rect_dects[i].x, target_rect_dects[i].y, target_rect_dects[i].width, target_rect_dects[i].height));
				trackers.push_back(trk);
			}
            return;
        }   
        //3.1. get predicted locations from existing trackers.
        predictedBoxes.clear();
		for (auto it = trackers.begin(); it != trackers.end();)
		{
			Rect_<float> pBox = (*it).predict();
			if (pBox.x >= 0 && pBox.y >= 0)
			{
				predictedBoxes.push_back(pBox);
				it++;
			}
			else
			{
				it = trackers.erase(it);
				//cerr << "Box invalid at frame: " << frame_count << endl;
			}
		}
        
        // 3.2. associate detections to tracked object (both represented as bounding boxes)
		// dets : detFrameData[fi]
		auto trkNum = predictedBoxes.size();
		auto detNum = target_rect_dects.size();

		iouMatrix.clear();
		iouMatrix.resize(trkNum, vector<double>(detNum, 0));
        
        for (unsigned int i = 0; i < trkNum; i++) // compute iou matrix as a distance matrix
		{
			for (unsigned int j = 0; j < detNum; j++)
			{
				// use 1-iou because the hungarian algorithm computes a minimum-cost assignment.
				iouMatrix[i][j] = 1 - GetIOU(predictedBoxes[i], cv::Rect_<float>(target_rect_dects[j].x, target_rect_dects[j].y, target_rect_dects[j].width, target_rect_dects[j].height));
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


        if (detNum > trkNum) //	there are unmatched detections
		{
			for (unsigned int n = 0; n < detNum; n++)
				allItems.insert(n);

			for (unsigned int i = 0; i < trkNum; ++i)
				matchedItems.insert(assignment[i]);

			set_difference(allItems.begin(), allItems.end(),
				matchedItems.begin(), matchedItems.end(),
				insert_iterator<set<int>>(unmatchedDetections, unmatchedDetections.begin()));
		}
		else if (detNum < trkNum) // there are unmatched trajectory/predictions
		{
			for (unsigned int i = 0; i < trkNum; ++i)
				if (assignment[i] == -1) // unassigned label will be set as -1 in the assignment algorithm
					unmatchedTrajectories.insert(i);
		}
		else
		{}
        
        // filter out matched with low IOU
		matchedPairs.clear();
		for (unsigned int i = 0; i < trkNum; ++i)
		{
			if (assignment[i] == -1) // pass over invalid values
				continue;
			if (1 - iouMatrix[i][assignment[i]] < iouThreshold)
			{
				unmatchedTrajectories.insert(i);
				unmatchedDetections.insert(assignment[i]);
			}
			else
				matchedPairs.push_back(cv::Point(i, assignment[i]));
		}


        // 3.3. updating trackers
		// update matched trackers with assigned detections.
		// each prediction is corresponding to a tracker
		int detIdx, trkIdx;
		for (unsigned int i = 0; i < matchedPairs.size(); i++)
		{
			trkIdx = matchedPairs[i].x;
			detIdx = matchedPairs[i].y;
			trackers[trkIdx].update(cv::Rect_<float>(target_rect_dects[detIdx].x, 
                                                    target_rect_dects[detIdx].y, 
                                                    target_rect_dects[detIdx].width, 
                                                    target_rect_dects[detIdx].height));
		}

		// create and initialise new trackers for unmatched detections
		for (auto& umd : unmatchedDetections)
		{
			KalmanTracker tracker = KalmanTracker(cv::Rect_<float>(target_rect_dects[umd].x, 
                                                                   target_rect_dects[umd].y,
                                                                   target_rect_dects[umd].width, 
                                                                   target_rect_dects[umd].height));
			trackers.push_back(tracker);
		}

        // get trackers' output
		frameTrackingResult.clear();
		for (auto it = trackers.begin(); it != trackers.end();)
		{
			if (((*it).m_time_since_update < 1) &&
				((*it).m_hit_streak >= min_hits))
			{
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

        for (const auto& tb : frameTrackingResult)
        {
			// id and box need to correspond
            track_ids.push_back(tb.id);
            target_rects.push_back(vp_objects::vp_rect(tb.box.x, tb.box.y, tb.box.width, tb.box.height));
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