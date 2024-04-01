
#include "vp_face_swap_node.h"

namespace vp_nodes {
    
    vp_face_swap_node::vp_face_swap_node(std::string node_name, 
                                        std::string yunet_face_detect_model,
                                        std::string buffalo_l_face_encoding_model,
                                        std::string emap_file_for_embeddings,
                                        std::string insightface_swap_model,
                                        std::string swap_source_image,
                                        int swap_source_face_index,
                                        bool swap_on_osd,
                                        bool act_as_primary_detector):
                                        vp_primary_infer_node(node_name, ""),
                                        act_as_primary_detector(act_as_primary_detector),
                                        swap_on_osd(swap_on_osd) {
        /* init net*/
        face_extract_net = cv::dnn::readNetFromONNX(yunet_face_detect_model);
        face_encoding_net = cv::dnn::readNetFromONNX(buffalo_l_face_encoding_model);
        face_swap_net = cv::dnn::readNetFromONNX(insightface_swap_model);
        #ifdef VP_WITH_CUDA
        face_extract_net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        face_extract_net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        face_encoding_net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        face_encoding_net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        face_swap_net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        face_swap_net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        #endif
        init_source_face_embeddings(swap_source_image, swap_source_face_index, emap_file_for_embeddings);
        this->initialized();
    }
    
    vp_face_swap_node::~vp_face_swap_node() {
        deinitialized();
    }

    // please refer to vp_infer_node::run_infer_combinations
    void vp_face_swap_node::run_infer_combinations(const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) {
        assert(frame_meta_with_batch.size() == 1);
        auto& frame_meta = frame_meta_with_batch[0];

        if (frame_meta->face_targets.size() == 0) {
            if (!act_as_primary_detector) {
                return;
            }
            // extract faces and update back to frame meta
            // to-do
        }
        
        auto start_time = std::chrono::system_clock::now();
        // iterate each face target
        for (int i = 0; i < frame_meta->face_targets.size(); i++) {
            cv::Mat aligned_face, swapped_face;

            //auto t = std::chrono::system_clock::now();
            // align and crop first
            auto warp_mat = align_crop(frame_meta->frame, frame_meta->face_targets[i]->key_points, aligned_face);
            //auto T = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - t).count();

            //std::cout << "1st T:" << T << std::endl;
            //t = std::chrono::system_clock::now();
            // swap
            swap(aligned_face, swapped_face);
            //T = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - t).count();
            //std::cout << "2nd T:" << T << std::endl;

            //t = std::chrono::system_clock::now();
            // past back to frame or osd frame
            if (swap_on_osd) {
                if (frame_meta->osd_frame.empty()) {
                    frame_meta->osd_frame = frame_meta->frame.clone();
                }
            }
            auto& bg = swap_on_osd ? frame_meta->osd_frame : frame_meta->frame;
            paste_back(bg, swapped_face, warp_mat);
            //T = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - t).count();
            //std::cout << "3rd T:" << T << std::endl;
        }

        //auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time).count();
        //std::cout << "total T:" << total_time << std::endl;
    }

    cv::Mat vp_face_swap_node::read_emap_mat_from_txt(const std::string& emap_file_for_embeddings, int rows, int cols) {
        std::ifstream file(emap_file_for_embeddings);

        cv::Mat matrix(rows, cols, CV_32F);
        std::string line;
        for (int i = 0; i < rows; ++i) {
            std::getline(file, line);
            std::istringstream iss(line);
            for (int j = 0; j < cols; ++j) {
                float value;
                assert(iss >> value);
                matrix.at<float>(i, j) = value;
            }
        }
        file.close();
        return matrix;
    }

    cv::Mat vp_face_swap_node::process_embeddings_using_emap(const cv::Mat& source_face_normed_embedding, const cv::Mat& emap) {
        cv::Mat latent = source_face_normed_embedding.clone().reshape(1, 1);
        cv::Mat result = latent * emap;
        cv::normalize(result, result);
        return result;
    }

    cv::Mat vp_face_swap_node::get_similarity_transform_matrix(float src[5][2]) {
        using namespace cv;
        //float dst[5][2] = { {38.2946f, 51.6963f}, {73.5318f, 51.5014f}, {56.0252f, 71.7366f}, {41.5493f, 92.3655f}, {70.7299f, 92.2041f} };  // for 112*112
        float dst[5][2] = { {43.0f, 58.0f}, {85.0f, 58.0f}, {64.0f, 81.0f}, {47.0f, 105.0f}, {80.0f, 105.0f} };  // for 128*128
        //float dst[5][2] = { {38.0f, 54.0f}, {90.0f, 54.0f}, {64.0f, 85.0f}, {47.0f, 109.0f}, {80.0f, 109.0f} };  // for 128*128, zoom out
        float avg0 = (src[0][0] + src[1][0] + src[2][0] + src[3][0] + src[4][0]) / 5;
        float avg1 = (src[0][1] + src[1][1] + src[2][1] + src[3][1] + src[4][1]) / 5;
        //Compute mean of src and dst.
        float src_mean[2] = { avg0, avg1 };
        float dst_mean[2] = { 56.0262f, 71.9008f };
        //Subtract mean from src and dst.
        float src_demean[5][2];
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                src_demean[j][i] = src[j][i] - src_mean[i];
            }
        }
        float dst_demean[5][2];
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                dst_demean[j][i] = dst[j][i] - dst_mean[i];
            }
        }
        double A00 = 0.0, A01 = 0.0, A10 = 0.0, A11 = 0.0;
        for (int i = 0; i < 5; i++)
            A00 += dst_demean[i][0] * src_demean[i][0];
        A00 = A00 / 5;
        for (int i = 0; i < 5; i++)
            A01 += dst_demean[i][0] * src_demean[i][1];
        A01 = A01 / 5;
        for (int i = 0; i < 5; i++)
            A10 += dst_demean[i][1] * src_demean[i][0];
        A10 = A10 / 5;
        for (int i = 0; i < 5; i++)
            A11 += dst_demean[i][1] * src_demean[i][1];
        A11 = A11 / 5;
        Mat A = (Mat_<double>(2, 2) << A00, A01, A10, A11);
        double d[2] = { 1.0, 1.0 };
        double detA = A00 * A11 - A01 * A10;
        if (detA < 0)
            d[1] = -1;
        double T[3][3] = { {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0} };
        Mat s, u, vt, v;
        SVD::compute(A, s, u, vt);
        double smax = s.ptr<double>(0)[0]>s.ptr<double>(1)[0] ? s.ptr<double>(0)[0] : s.ptr<double>(1)[0];
        double tol = smax * 2 * FLT_MIN;
        int rank = 0;
        if (s.ptr<double>(0)[0]>tol)
            rank += 1;
        if (s.ptr<double>(1)[0]>tol)
            rank += 1;
        double arr_u[2][2] = { {u.ptr<double>(0)[0], u.ptr<double>(0)[1]}, {u.ptr<double>(1)[0], u.ptr<double>(1)[1]} };
        double arr_vt[2][2] = { {vt.ptr<double>(0)[0], vt.ptr<double>(0)[1]}, {vt.ptr<double>(1)[0], vt.ptr<double>(1)[1]} };
        double det_u = arr_u[0][0] * arr_u[1][1] - arr_u[0][1] * arr_u[1][0];
        double det_vt = arr_vt[0][0] * arr_vt[1][1] - arr_vt[0][1] * arr_vt[1][0];
        if (rank == 1)
        {
            if ((det_u*det_vt) > 0)
            {
                Mat uvt = u*vt;
                T[0][0] = uvt.ptr<double>(0)[0];
                T[0][1] = uvt.ptr<double>(0)[1];
                T[1][0] = uvt.ptr<double>(1)[0];
                T[1][1] = uvt.ptr<double>(1)[1];
            }
            else
            {
                double temp = d[1];
                d[1] = -1;
                Mat D = (Mat_<double>(2, 2) << d[0], 0.0, 0.0, d[1]);
                Mat Dvt = D*vt;
                Mat uDvt = u*Dvt;
                T[0][0] = uDvt.ptr<double>(0)[0];
                T[0][1] = uDvt.ptr<double>(0)[1];
                T[1][0] = uDvt.ptr<double>(1)[0];
                T[1][1] = uDvt.ptr<double>(1)[1];
                d[1] = temp;
            }
        }
        else
        {
            Mat D = (Mat_<double>(2, 2) << d[0], 0.0, 0.0, d[1]);
            Mat Dvt = D*vt;
            Mat uDvt = u*Dvt;
            T[0][0] = uDvt.ptr<double>(0)[0];
            T[0][1] = uDvt.ptr<double>(0)[1];
            T[1][0] = uDvt.ptr<double>(1)[0];
            T[1][1] = uDvt.ptr<double>(1)[1];
        }
        double var1 = 0.0;
        for (int i = 0; i < 5; i++)
            var1 += src_demean[i][0] * src_demean[i][0];
        var1 = var1 / 5;
        double var2 = 0.0;
        for (int i = 0; i < 5; i++)
            var2 += src_demean[i][1] * src_demean[i][1];
        var2 = var2 / 5;
        double scale = 1.0 / (var1 + var2)* (s.ptr<double>(0)[0] * d[0] + s.ptr<double>(1)[0] * d[1]);
        double TS[2];
        TS[0] = T[0][0] * src_mean[0] + T[0][1] * src_mean[1];
        TS[1] = T[1][0] * src_mean[0] + T[1][1] * src_mean[1];
        T[0][2] = dst_mean[0] - scale*TS[0];
        T[1][2] = dst_mean[1] - scale*TS[1];
        T[0][0] *= scale;
        T[0][1] *= scale;
        T[1][0] *= scale;
        T[1][1] *= scale;
        Mat transform_mat = (Mat_<double>(2, 3) << T[0][0], T[0][1], T[0][2], T[1][0], T[1][1], T[1][2]);
        return transform_mat;
    }

    cv::Mat vp_face_swap_node::align_crop(cv::Mat& src_img, std::vector<std::pair<int, int>>& kps, cv::Mat& aligned_image) {
        float face_keypoints[5][2] = 
            {{kps[0].first, kps[0].second}, 
            {kps[1].first, kps[1].second}, 
            {kps[2].first, kps[2].second},  
            {kps[3].first, kps[3].second}, 
            {kps[4].first, kps[4].second}};

        cv::Mat warp_mat = get_similarity_transform_matrix(face_keypoints);
        cv::warpAffine(src_img, aligned_image, warp_mat, cv::Size(128, 128), cv::INTER_LINEAR);
        return warp_mat;
    }
    
    void vp_face_swap_node::extract_faces(const cv::Mat& frame, std::vector<face_box>& face_boxes) {
        auto blob_to_infer = cv::dnn::blobFromImage(frame);
        face_extract_net.setInput(blob_to_infer);
        const std::vector<std::string> out_names = {"loc", "conf", "iou"};
        std::vector<cv::Mat> raw_outputs;
        face_extract_net.forward(raw_outputs, out_names);

        using namespace cv;
        float scoreThreshold = 0.7;
        float nmsThreshold = 0.5;
        int topK = 50;
        int inputW = frame.cols;
        int inputH = frame.rows;

        // 3 heads of output
        assert(raw_outputs.size() == 3);

        // Extract from output_blobs
        Mat loc = raw_outputs[0];
        Mat conf = raw_outputs[1];
        Mat iou = raw_outputs[2];

        // we need generate priors if input size changed or priors is not initialized
        if (loc.rows != priors.size()) {
            generatePriors(inputW, inputH);
        }
        
        assert(loc.rows == priors.size());
        assert(loc.rows == conf.rows);
        assert(loc.rows == iou.rows);

        // Decode from deltas and priors
        const std::vector<float> variance = {0.1f, 0.2f};
        float* loc_v = (float*)(loc.data);
        float* conf_v = (float*)(conf.data);
        float* iou_v = (float*)(iou.data);
        Mat faces;
        // (tl_x, tl_y, w, h, re_x, re_y, le_x, le_y, nt_x, nt_y, rcm_x, rcm_y, lcm_x, lcm_y, score)
        // 'tl': top left point of the bounding box
        // 're': right eye, 'le': left eye
        // 'nt':  nose tip
        // 'rcm': right corner of mouth, 'lcm': left corner of mouth
        Mat face(1, 15, CV_32FC1);
        for (size_t i = 0; i < priors.size(); ++i) {
            // Get score
            float clsScore = conf_v[i*2+1];
            float iouScore = iou_v[i];
            // Clamp
            if (iouScore < 0.f) {
                iouScore = 0.f;
            }
            else if (iouScore > 1.f) {
                iouScore = 1.f;
            }
            float score = std::sqrt(clsScore * iouScore);
            face.at<float>(0, 14) = score;

            // Get bounding box
            float cx = (priors[i].x + loc_v[i*14+0] * variance[0] * priors[i].width)  * inputW;
            float cy = (priors[i].y + loc_v[i*14+1] * variance[0] * priors[i].height) * inputH;
            float w  = priors[i].width  * exp(loc_v[i*14+2] * variance[0]) * inputW;
            float h  = priors[i].height * exp(loc_v[i*14+3] * variance[1]) * inputH;
            float x1 = cx - w / 2;
            float y1 = cy - h / 2;
            face.at<float>(0, 0) = x1;
            face.at<float>(0, 1) = y1;
            face.at<float>(0, 2) = w;
            face.at<float>(0, 3) = h;

            // Get landmarks
            face.at<float>(0, 4) = (priors[i].x + loc_v[i*14+ 4] * variance[0] * priors[i].width)  * inputW;  // right eye, x
            face.at<float>(0, 5) = (priors[i].y + loc_v[i*14+ 5] * variance[0] * priors[i].height) * inputH;  // right eye, y
            face.at<float>(0, 6) = (priors[i].x + loc_v[i*14+ 6] * variance[0] * priors[i].width)  * inputW;  // left eye, x
            face.at<float>(0, 7) = (priors[i].y + loc_v[i*14+ 7] * variance[0] * priors[i].height) * inputH;  // left eye, y
            face.at<float>(0, 8) = (priors[i].x + loc_v[i*14+ 8] * variance[0] * priors[i].width)  * inputW;  // nose tip, x
            face.at<float>(0, 9) = (priors[i].y + loc_v[i*14+ 9] * variance[0] * priors[i].height) * inputH;  // nose tip, y
            face.at<float>(0, 10) = (priors[i].x + loc_v[i*14+10] * variance[0] * priors[i].width)  * inputW; // right corner of mouth, x
            face.at<float>(0, 11) = (priors[i].y + loc_v[i*14+11] * variance[0] * priors[i].height) * inputH; // right corner of mouth, y
            face.at<float>(0, 12) = (priors[i].x + loc_v[i*14+12] * variance[0] * priors[i].width)  * inputW; // left corner of mouth, x
            face.at<float>(0, 13) = (priors[i].y + loc_v[i*14+13] * variance[0] * priors[i].height) * inputH; // left corner of mouth, y

            faces.push_back(face);
        }

        if (faces.rows > 1) {
            // Retrieve boxes and scores
            std::vector<Rect2i> faceBoxes;
            std::vector<float> faceScores;
            for (int rIdx = 0; rIdx < faces.rows; rIdx++)
            {
                faceBoxes.push_back(Rect2i(int(faces.at<float>(rIdx, 0)),
                                           int(faces.at<float>(rIdx, 1)),
                                           int(faces.at<float>(rIdx, 2)),
                                           int(faces.at<float>(rIdx, 3))));
                faceScores.push_back(faces.at<float>(rIdx, 14));
            }

            std::vector<int> keepIdx;
            dnn::NMSBoxes(faceBoxes, faceScores, scoreThreshold, nmsThreshold, keepIdx, 1.f, topK);

            // Get NMS results
            Mat nms_faces;
            for (int idx: keepIdx)
            {
                nms_faces.push_back(faces.row(idx));
            }

            // insert face target back to frame meta
            for (int i = 0; i < nms_faces.rows; i++) {
                auto x = int(nms_faces.at<float>(i, 0));
                auto y = int(nms_faces.at<float>(i, 1));
                auto w = int(nms_faces.at<float>(i, 2));
                auto h = int(nms_faces.at<float>(i, 3));
                
                // check value range
                x = std::max(x, 0);
                y = std::max(y, 0);
                w = std::min(w, frame.cols - x);
                h = std::min(h, frame.rows - y);

                auto kp1 = std::pair<int, int>(int(nms_faces.at<float>(i, 4)), int(nms_faces.at<float>(i, 5)));
                auto kp2 = std::pair<int, int>(int(nms_faces.at<float>(i, 6)), int(nms_faces.at<float>(i, 7)));
                auto kp3 = std::pair<int, int>(int(nms_faces.at<float>(i, 8)), int(nms_faces.at<float>(i, 9)));
                auto kp4 = std::pair<int, int>(int(nms_faces.at<float>(i, 10)), int(nms_faces.at<float>(i, 11)));
                auto kp5 = std::pair<int, int>(int(nms_faces.at<float>(i, 12)), int(nms_faces.at<float>(i, 13)));
                auto score = nms_faces.at<float>(i, 14);

                face_box face;
                face.x = x;
                face.y = y;
                face.width = w;
                face.height = h;
                face.score = score;
                face.kps = std::vector<std::pair<int, int>>{kp1, kp2, kp3, kp4, kp5};
                face_boxes.push_back(face);
            }
        }
    }

    void vp_face_swap_node::init_source_face_embeddings(std::string& swap_source_image, int swap_source_face_index, std::string& emap_file_for_embeddings) {
        std::vector<face_box> source_faces;
        auto source_mat = cv::imread(swap_source_image);

        // extract faces
        extract_faces(source_mat, source_faces);

        assert(source_faces.size() > 0);
        // sort from left to right
        std::sort(source_faces.begin(), source_faces.end(), [](face_box a, face_box b){ return a.x < b.x;});

        auto the_selected_face = (swap_source_face_index < 0 || swap_source_face_index >= source_faces.size()) ? source_faces[0] : source_faces[swap_source_face_index];
        cv::Mat aligned_face;
        auto warp_mat = align_crop(source_mat, the_selected_face.kps, aligned_face);
        // for debug
        cv::imwrite("selected_source_face.jpg", aligned_face);

        // read emap from file
        auto emap = read_emap_mat_from_txt(emap_file_for_embeddings);

        // encoding for the selected face (infer for only 1 time)
        cv::Mat source_blob = cv::dnn::blobFromImage(aligned_face, 1 / 127.5, cv::Size(112, 112), (127.5, 127.5, 127.5), true);
        std::vector<cv::Mat> source_outputs;
        face_encoding_net.setInput(source_blob);
        face_encoding_net.forward(source_outputs, face_encoding_net.getUnconnectedOutLayersNames());

        auto& source_output = source_outputs[0];
        // process using emap
        source_face_embeddings = process_embeddings_using_emap(source_output, emap);
    }

    void vp_face_swap_node::swap(cv::Mat& aligned_face, cv::Mat& swapped_face) {
        cv::Mat target_blob = cv::dnn::blobFromImage(aligned_face, 1 / 255.0, cv::Size(128, 128), (0, 0, 0), true);
        std::vector<cv::Mat> target_outputs;
        face_swap_net.setInput(target_blob, "target");
        face_swap_net.setInput(source_face_embeddings, "source");
        face_swap_net.forward(target_outputs, face_swap_net.getUnconnectedOutLayersNames());

        auto& output = target_outputs[0];
        cv::Mat output_channel_last;
        cv::transposeND(output, {0,2,3,1}, output_channel_last);

        cv::Mat img_fake(output_channel_last.size[1], output_channel_last.size[2], CV_32FC3, output_channel_last.data);
        cv::cvtColor(img_fake, swapped_face, cv::COLOR_RGB2BGR);
    }

    void vp_face_swap_node::paste_back(cv::Mat& bg, cv::Mat& swapped_face, const cv::Mat& transform_matrix) {
        bg.convertTo(bg, CV_32FC3, 1.0 / 255);

        cv::Mat IM;
        cv::invertAffineTransform(transform_matrix, IM);
        cv::Mat img_mask(swapped_face.rows, swapped_face.cols, CV_32FC1, 1);
        cv::warpAffine(swapped_face, swapped_face, IM, bg.size());
        cv::warpAffine(img_mask, img_mask, IM, bg.size());

        // create mask
        cv::threshold(img_mask, img_mask, 0, 1, cv::THRESH_BINARY);
        cv::Point min_loc, max_loc;
        double min_val, max_val;
        cv::minMaxLoc(img_mask, &min_val, &max_val, &min_loc, &max_loc);

        int mask_h = max_loc.y - min_loc.y;
        int mask_w = max_loc.x - min_loc.x;

        int mask_size = std::sqrt(mask_h * mask_w);
        int k = std::max(mask_size / 10, 10);

        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(k, k));
        cv::erode(img_mask, img_mask, kernel);

        k = std::max(mask_size / 20, 5);
        cv::GaussianBlur(img_mask, img_mask, cv::Size(2 * k + 1, 2 * k + 1), 0);
        cv::cvtColor(img_mask, img_mask, cv::COLOR_GRAY2BGR);

        // merge swapped face and original background
        cv::Mat ones(bg.rows, bg.cols, CV_32FC3, cv::Scalar(1, 1, 1));
        bg = img_mask.mul(swapped_face) + (ones - img_mask).mul(bg);
        bg.convertTo(bg, CV_8U, 255);
    }

    void vp_face_swap_node::generatePriors(int inputW, int inputH) {
         using namespace cv;
        // Calculate shapes of different scales according to the shape of input image
        Size feature_map_2nd = {
            int(int((inputW+1)/2)/2), int(int((inputH+1)/2)/2)
        };
        Size feature_map_3rd = {
            int(feature_map_2nd.width/2), int(feature_map_2nd.height/2)
        };
        Size feature_map_4th = {
            int(feature_map_3rd.width/2), int(feature_map_3rd.height/2)
        };
        Size feature_map_5th = {
            int(feature_map_4th.width/2), int(feature_map_4th.height/2)
        };
        Size feature_map_6th = {
            int(feature_map_5th.width/2), int(feature_map_5th.height/2)
        };

        std::vector<Size> feature_map_sizes;
        feature_map_sizes.push_back(feature_map_3rd);
        feature_map_sizes.push_back(feature_map_4th);
        feature_map_sizes.push_back(feature_map_5th);
        feature_map_sizes.push_back(feature_map_6th);

        // Fixed params for generating priors
        const std::vector<std::vector<float>> min_sizes = {
            {10.0f,  16.0f,  24.0f},
            {32.0f,  48.0f},
            {64.0f,  96.0f},
            {128.0f, 192.0f, 256.0f}
        };
        CV_Assert(min_sizes.size() == feature_map_sizes.size()); // just to keep vectors in sync
        const std::vector<int> steps = { 8, 16, 32, 64 };

        // Generate priors
        priors.clear();
        for (size_t i = 0; i < feature_map_sizes.size(); ++i)
        {
            Size feature_map_size = feature_map_sizes[i];
            std::vector<float> min_size = min_sizes[i];

            for (int _h = 0; _h < feature_map_size.height; ++_h)
            {
                for (int _w = 0; _w < feature_map_size.width; ++_w)
                {
                    for (size_t j = 0; j < min_size.size(); ++j)
                    {
                        float s_kx = min_size[j] / inputW;
                        float s_ky = min_size[j] / inputH;

                        float cx = (_w + 0.5f) * steps[i] / inputW;
                        float cy = (_h + 0.5f) * steps[i] / inputH;

                        Rect2f prior = { cx, cy, s_kx, s_ky };
                        priors.push_back(prior);
                    }
                }
            }
        }
    }

    void vp_face_swap_node::postprocess(const std::vector<cv::Mat>& raw_outputs, const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) {

    }
}