
#include "vp_face_osd_node_v2.h"

namespace vp_nodes {
        
    vp_face_osd_node_v2::vp_face_osd_node_v2(std::string node_name): vp_node(node_name) {
        this->initialized();
    }
    
    vp_face_osd_node_v2::~vp_face_osd_node_v2() {

    }

    std::shared_ptr<vp_objects::vp_meta> vp_face_osd_node_v2::handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) {
        // operations on osd_frame
        if (meta->osd_frame.empty()) {
            // add a gap at the bottom of osd frame
            meta->osd_frame = cv::Mat(meta->frame.rows + gap_height + padding * 2, meta->frame.cols, meta->frame.type(), cv::Scalar(128, 128, 128));
            
            // initialize by copying frame to osd frame
            auto roi = meta->osd_frame(cv::Rect(0, 0, meta->frame.cols, meta->frame.rows));
            meta->frame.copyTo(roi);
        }
        auto& canvas = meta->osd_frame;

        // scan face targets in current frame
        for(auto& i : meta->face_targets) {
            // the first one
            if (the_baseline_face.empty()) {
                auto face = meta->frame(cv::Rect(i->x, i->y, i->width, i->height));
                cv::resize(face, the_baseline_face, cv::Size(gap_height, gap_height));   
                the_baseline_face_feature = i->embeddings;
            }
            else {
                // check if the face has existed in list by calculating distance between 2 faces
                bool exist = false;
                for(auto& f : face_features) {
                    if (match(i->embeddings, f, 0) >= cosine_similar_thresh ||
                        match(i->embeddings, f, 1) <= l2norm_similar_thresh) {
                        exist = true;
                        break;
                    }
                }

                if (!exist) {
                    auto cosine_dis = match(i->embeddings, the_baseline_face_feature, 0);
                    auto l2_dis = match(i->embeddings, the_baseline_face_feature, 1);
                    if (cosine_dis >= cosine_similar_thresh || l2_dis <= l2norm_similar_thresh) {
                        // as same as the_baseline_face
                    }
                    else {    
                        // new face, add it to list for dispaly 
                        auto face = meta->frame(cv::Rect(i->x, i->y, i->width, i->height));
                        cv::Mat resized_face;
                        cv::resize(face, resized_face, cv::Size(gap_height, gap_height));         
                        
                        faces_list.push_back(resized_face);
                        face_features.push_back(i->embeddings);
                        cosine_distances.push_back(cosine_dis);
                        l2_distances.push_back(l2_dis);
                    }
                }
                else {
                    // has existed in list
                }
            }
        }

        // too many faces, delete the first one in list
        auto width_need = faces_list.size() * (gap_height + padding) + padding;
        if (width_need > canvas.cols) {
            faces_list.erase(faces_list.begin());
            face_features.erase(face_features.begin());
            cosine_distances.erase(cosine_distances.begin());
            l2_distances.erase(l2_distances.begin());
        }
        
        // make sure the size for each vector
        assert(faces_list.size() == face_features.size());
        assert(faces_list.size() == cosine_distances.size());
        assert(faces_list.size() == l2_distances.size());

        assert(canvas.rows > (gap_height + padding) * 2);

        // display faces in list
        for (int i = 0; i < faces_list.size(); i++) {
            auto roi = canvas(cv::Rect((padding + gap_height) * i + padding, canvas.rows - gap_height - padding, gap_height, gap_height));
            faces_list[i].copyTo(roi);
        }
        
        // display the baseline face
        if (!the_baseline_face.empty()) {  
            auto roi = canvas(cv::Rect(padding, canvas.rows - gap_height * 2 - padding * 2, gap_height, gap_height));
            the_baseline_face.copyTo(roi);
        }

        return meta;
    }

    double vp_face_osd_node_v2::match(std::vector<float>& feature1, std::vector<float>& feature2, int dis_type) {
        auto _face_feature1 = cv::Mat(1, feature1.size(), CV_32F, feature1.data());
        auto _face_feature2 = cv::Mat(1, feature2.size(), CV_32F, feature2.data());
        cv::normalize(_face_feature1, _face_feature1);
        cv::normalize(_face_feature2, _face_feature2);

        if(dis_type == 0) {
            return cv::sum(_face_feature1.mul(_face_feature2))[0];
        }
        else if(dis_type == 1) {
            return cv::norm(_face_feature1, _face_feature2);
        }
        else {
            throw std::invalid_argument("invalid parameter " + std::to_string(dis_type));
        }

    }
}