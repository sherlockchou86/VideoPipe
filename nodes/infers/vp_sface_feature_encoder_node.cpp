
#include <algorithm>
#include <opencv2/imgproc.hpp>
#include "vp_sface_feature_encoder_node.h"

namespace vp_nodes {
        
    vp_sface_feature_encoder_node::vp_sface_feature_encoder_node(std::string node_name, std::string model_path):
                                                                vp_secondary_infer_node(node_name, model_path, 
                                                                "", "", 
                                                                112, 112, 
                                                                1, std::vector<int>(), 0, 0, 
                                                                0, 1, cv::Scalar()) {
        this->initialized();
    }
    
    vp_sface_feature_encoder_node::~vp_sface_feature_encoder_node()
    {
    }

    void vp_sface_feature_encoder_node::postprocess(const std::vector<cv::Mat>& raw_outputs, const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) {
        // make sure heads of output are not zero
        assert(raw_outputs.size() > 0);
        assert(frame_meta_with_batch.size() == 1);

        // just one head of output
        auto& output = raw_outputs[0];
        assert(output.dims == 2);

        auto count = output.rows;
        auto& frame_meta = frame_meta_with_batch[0];

        // update feature data back into frame meta
        for (int i = 0; i < count; i++) {
            cv::Mat feature = output.row(i);

            for (int j = 0; j < feature.cols; j++) {
                frame_meta->face_targets[i]->embeddings.push_back(feature.at<float>(0, j));
            }
        }  
    }

    // refer to vp_secondary_infer_node::prepare
    void vp_sface_feature_encoder_node::prepare(const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch, std::vector<cv::Mat>& mats_to_infer) {
        // only one by one for secondary infer node
        assert(frame_meta_with_batch.size() == 1);

        // align face and crop 
        auto& frame_meta = frame_meta_with_batch[0];

        // batch by batch inside single frame
        for (auto& i : frame_meta->face_targets) {
            // align and crop
            float face_keypoints[5][2] = 
                {{i->key_points[0].first, i->key_points[0].second}, 
                 {i->key_points[1].first, i->key_points[1].second}, 
                 {i->key_points[2].first, i->key_points[2].second}, 
                 {i->key_points[3].first, i->key_points[3].second}, 
                 {i->key_points[4].first, i->key_points[4].second}};
            cv::Mat aligned_face;
            alignCrop(frame_meta->frame, face_keypoints, aligned_face);
            mats_to_infer.push_back(aligned_face); 
        }
    }

    cv::Mat vp_sface_feature_encoder_node::getSimilarityTransformMatrix(float src[5][2]) {
        using namespace cv;
        float dst[5][2] = { {38.2946f, 51.6963f}, {73.5318f, 51.5014f}, {56.0252f, 71.7366f}, {41.5493f, 92.3655f}, {70.7299f, 92.2041f} };
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

    void vp_sface_feature_encoder_node::alignCrop(cv::Mat& _src_img, float _src_point[5][2], cv::Mat& _aligned_img) {
        cv::Mat warp_mat = getSimilarityTransformMatrix(_src_point);
        cv::warpAffine(_src_img, _aligned_img, warp_mat, cv::Size(input_width, input_height), cv::INTER_LINEAR);
    }
}