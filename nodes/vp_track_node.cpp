
#include "vp_track_node.h"
#include "../objects/vp_frame_target.h"
#include "../objects/elements/vp_frame_element.h"

namespace vp_nodes {
        
    vp_track_node::vp_track_node(std::string node_name): vp_node(node_name)
    {
        this->initialized();
        KalmanTracker::kf_count = 0;
    }
    
    vp_track_node::~vp_track_node()
    {
    }

    std::shared_ptr<vp_objects::vp_meta> vp_track_node::handle_control_meta(std::shared_ptr<vp_objects::vp_control_meta> meta) {
        return meta;
    }

    std::shared_ptr<vp_objects::vp_meta> vp_track_node::handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) {
        /*
        if (meta->frame_index % 10 == 0) {          
            std::this_thread::sleep_for(std::chrono::milliseconds(29));
        }
        if (meta->frame_index % 100 == 0) {          
            std::this_thread::sleep_for(std::chrono::milliseconds(84));
        }
        if (meta->frame_index % 130 == 0) {          
            std::this_thread::sleep_for(std::chrono::milliseconds(60));
        }*/

        if (trackers.empty() || meta->face_targets.empty())
        {
            /* first frame*/
            for (unsigned int i = 0; i < meta->face_targets.size(); i++)
			{
				KalmanTracker trk = KalmanTracker(cv::Rect_<float>(meta->face_targets[i]->x, meta->face_targets[i]->y, meta->face_targets[i]->width, meta->face_targets[i]->height));
				trackers.push_back(trk);

			}
            return meta;
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
		auto detNum = meta->face_targets.size();

		iouMatrix.clear();
		iouMatrix.resize(trkNum, vector<double>(detNum, 0));

		for (unsigned int i = 0; i < trkNum; i++) // compute iou matrix as a distance matrix
		{
			for (unsigned int j = 0; j < detNum; j++)
			{
				// use 1-iou because the hungarian algorithm computes a minimum-cost assignment.
				iouMatrix[i][j] = 1 - GetIOU(predictedBoxes[i], cv::Rect_<float>(meta->face_targets[j]->x, meta->face_targets[j]->y, meta->face_targets[j]->width, meta->face_targets[j]->height));
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
			trackers[trkIdx].update(cv::Rect_<float>(meta->face_targets[detIdx]->x, meta->face_targets[detIdx]->y, meta->face_targets[detIdx]->width, meta->face_targets[detIdx]->height));
		}

		// create and initialise new trackers for unmatched detections
		for (auto umd : unmatchedDetections)
		{
			KalmanTracker tracker = KalmanTracker(cv::Rect_<float>(meta->face_targets[umd]->x, meta->face_targets[umd]->y, meta->face_targets[umd]->width, meta->face_targets[umd]->height));
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
				res.frame = meta->frame_index;
				frameTrackingResult.push_back(res);
				it++;
			}
			else
				it++;

			// remove dead tracklet
			if (it != trackers.end() && (*it).m_time_since_update > max_age)
				it = trackers.erase(it);
		}


        // ½«tracker ÈûÈë vp_frame_targets
        for (const auto tb : frameTrackingResult)
        {
            auto vp_element = std::make_shared<vp_objects::vp_frame_element>(tb.id);
            auto vp_target = std::make_shared<vp_objects::vp_frame_target>(tb.box.x, tb.box.y, tb.box.width, tb.box.height, 0, 1, tb.frame, meta->channel_index);
            //std::vector<std::tuple<std::shared_ptr<vp_frame_element>, std::shared_ptr<vp_frame_target>, vp_ba::vp_ba_flag>> ba_flags_map;
            vp_target->track_id = tb.id;
            vp_target->tracks.push_back(vp_objects::vp_rect(tb.box.x, tb.box.y, tb.box.width, tb.box.height));
			//printf("------------------------------------------id = %d\n", tb.id);
			meta->ba_flags_map.push_back(std::make_tuple(vp_element, vp_target, vp_ba::vp_ba_flag::NONE));
        }
        return meta;
    }

    double vp_track_node::GetIOU(cv::Rect_<float> bb_test, cv::Rect_<float> bb_gt)
    {
        float in = (bb_test & bb_gt).area();
        float un = bb_test.area() + bb_gt.area() - in;

        if (un < DBL_EPSILON)
            return 0;

        return (double)(in / un);
    }
}