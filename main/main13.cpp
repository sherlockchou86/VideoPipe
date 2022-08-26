

#include <opencv2/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>

#include <iostream>

#include "VP.h"

#if MAIN13

using namespace cv;

Mat getSimilarityTransformMatrix(float src[5][2]) {
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

void alignCrop(Mat& _src_img, Mat& _face_mat, Mat& _aligned_img)
{
    float src_point[5][2];
    for (int row = 0; row < 5; ++row)
    {
        for(int col = 0; col < 2; ++col)
        {
            src_point[row][col] = _face_mat.at<float>(0, row*2+col+4);
        }
    }
    Mat warp_mat = getSimilarityTransformMatrix(src_point);
    warpAffine(_src_img, _aligned_img, warp_mat, Size(112, 112), INTER_LINEAR);
}

void feature(Mat& _aligned_img, Mat& _face_feature, cv::dnn::Net& net)
{
    Mat inputBolb = dnn::blobFromImage(_aligned_img, 1, Size(112, 112), Scalar(0, 0, 0), true, false);
    net.setInput(inputBolb);
    net.forward(_face_feature);
}

double match(Mat& _face_feature1, Mat& _face_feature2, int dis_type)
{
    normalize(_face_feature1, _face_feature1);
    normalize(_face_feature2, _face_feature2);

    if(dis_type == 0){
        return sum(_face_feature1.mul(_face_feature2))[0];
    }else if(dis_type == 1){
        return norm(_face_feature1, _face_feature2);
    }else{
        throw std::invalid_argument("invalid parameter " + std::to_string(dis_type));
    }

}


using namespace std;

float scoreThreshold = 0.7;
float nmsThreshold = 0.5;
int topK = 50;
int inputW;
int inputH;
std::vector<Rect2f> priors;

void generatePriors()
{
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

std::vector<Mat> postProcess(const std::vector<Mat>& output_blobs)
{
    // Extract from output_blobs
    Mat loc = output_blobs[0];
    Mat conf = output_blobs[1];
    Mat iou = output_blobs[2];

    // Decode from deltas and priors
    const std::vector<float> variance = {0.1f, 0.2f};
    float* loc_v = (float*)(loc.data);
    float* conf_v = (float*)(conf.data);
    float* iou_v = (float*)(iou.data);
    std::vector<Mat> b_faces;

    // (tl_x, tl_y, w, h, re_x, re_y, le_x, le_y, nt_x, nt_y, rcm_x, rcm_y, lcm_x, lcm_y, score)
    // 'tl': top left point of the bounding box
    // 're': right eye, 'le': left eye
    // 'nt':  nose tip
    // 'rcm': right corner of mouth, 'lcm': left corner of mouth

    assert(loc.rows % priors.size() == 0);
    assert(loc.rows == conf.rows);
    assert(loc.rows == iou.rows);

    auto batch = loc.rows / priors.size();

    for (int b = 0; b < batch; b++)
    {
        Mat faces;
        Mat face(1, 15, CV_32FC1);
        for (size_t i = 0; i < priors.size(); ++i) {
            // Get score
            float clsScore = conf_v[b*loc.rows + i*2+1];
            float iouScore = iou_v[b*loc.rows + i];
            // Clamp
            if (iouScore < 0.f) {
                iouScore = 0.f;
            }
            else if (iouScore > 1.f) {
                iouScore = 1.f;
            }
            float score = std::sqrt(clsScore * iouScore);
            if (score < scoreThreshold)
            {
                continue;
            }
            
            face.at<float>(0, 14) = score;

            // Get bounding box
            float cx = (priors[i].x + loc_v[b*loc.rows + i*14+0] * variance[0] * priors[i].width)  * inputW;
            float cy = (priors[i].y + loc_v[b*loc.rows + i*14+1] * variance[0] * priors[i].height) * inputH;
            float w  = priors[i].width  * exp(loc_v[b*loc.rows + i*14+2] * variance[0]) * inputW;
            float h  = priors[i].height * exp(loc_v[b*loc.rows + i*14+3] * variance[1]) * inputH;
            float x1 = cx - w / 2;
            float y1 = cy - h / 2;
            face.at<float>(0, 0) = x1;
            face.at<float>(0, 1) = y1;
            face.at<float>(0, 2) = w;
            face.at<float>(0, 3) = h;

            // Get landmarks
            face.at<float>(0, 4) = (priors[i].x + loc_v[b*loc.rows + i*14+ 4] * variance[0] * priors[i].width)  * inputW;  // right eye, x
            face.at<float>(0, 5) = (priors[i].y + loc_v[b*loc.rows + i*14+ 5] * variance[0] * priors[i].height) * inputH;  // right eye, y
            face.at<float>(0, 6) = (priors[i].x + loc_v[b*loc.rows + i*14+ 6] * variance[0] * priors[i].width)  * inputW;  // left eye, x
            face.at<float>(0, 7) = (priors[i].y + loc_v[b*loc.rows + i*14+ 7] * variance[0] * priors[i].height) * inputH;  // left eye, y
            face.at<float>(0, 8) = (priors[i].x + loc_v[b*loc.rows + i*14+ 8] * variance[0] * priors[i].width)  * inputW;  // nose tip, x
            face.at<float>(0, 9) = (priors[i].y + loc_v[b*loc.rows + i*14+ 9] * variance[0] * priors[i].height) * inputH;  // nose tip, y
            face.at<float>(0, 10) = (priors[i].x + loc_v[b*loc.rows + i*14+10] * variance[0] * priors[i].width)  * inputW; // right corner of mouth, x
            face.at<float>(0, 11) = (priors[i].y + loc_v[b*loc.rows + i*14+11] * variance[0] * priors[i].height) * inputH; // right corner of mouth, y
            face.at<float>(0, 12) = (priors[i].x + loc_v[b*loc.rows + i*14+12] * variance[0] * priors[i].width)  * inputW; // left corner of mouth, x
            face.at<float>(0, 13) = (priors[i].y + loc_v[b*loc.rows + i*14+13] * variance[0] * priors[i].height) * inputH; // left corner of mouth, y

            faces.push_back(face);
        }

        if (faces.rows > 1)
        {
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
            b_faces.push_back(nms_faces);
        }
        else
        {
            b_faces.push_back(faces);
        }
    }

    return b_faces;
}

void visualize(const std::vector<Mat>& inputs, int frame, const std::vector<Mat>& faces,  double fps, int thickness = 2)
{
    std::string fpsString = cv::format("FPS : %.2f", (float)fps);
    if (frame >= 0)
        cout << "Frame " << frame << ", ";

    cout << "FPS: " << fpsString << endl;

    assert(inputs.size() == faces.size());
    for (size_t j = 0; j < inputs.size(); j++)
    {
        for (int i = 0; i < faces[j].rows; i++)
        {
            // Print results
            cout << "Face " << i
                << ", top-left coordinates: (" << faces[j].at<float>(i, 0) << ", " << faces[j].at<float>(i, 1) << "), "
                << "box width: " << faces[j].at<float>(i, 2)  << ", box height: " << faces[j].at<float>(i, 3) << ", "
                << "score: " << cv::format("%.2f", faces[j].at<float>(i, 14))
                << endl;

            // Draw bounding box
            rectangle(inputs[j], Rect2i(int(faces[j].at<float>(i, 0)), int(faces[j].at<float>(i, 1)), int(faces[j].at<float>(i, 2)), int(faces[j].at<float>(i, 3))), Scalar(0, 255, 0), thickness);
            // Draw landmarks
            circle(inputs[j], Point2i(int(faces[j].at<float>(i, 4)), int(faces[j].at<float>(i, 5))), 2, Scalar(255, 0, 0), thickness);
            circle(inputs[j], Point2i(int(faces[j].at<float>(i, 6)), int(faces[j].at<float>(i, 7))), 2, Scalar(0, 0, 255), thickness);
            circle(inputs[j], Point2i(int(faces[j].at<float>(i, 8)), int(faces[j].at<float>(i, 9))), 2, Scalar(0, 255, 0), thickness);
            circle(inputs[j], Point2i(int(faces[j].at<float>(i, 10)), int(faces[j].at<float>(i, 11))), 2, Scalar(255, 0, 255), thickness);
            circle(inputs[j], Point2i(int(faces[j].at<float>(i, 12)), int(faces[j].at<float>(i, 13))), 2, Scalar(0, 255, 255), thickness);
        }
        putText(inputs[j], fpsString, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
    }
}

double cosine_similar_thresh = 0.363;
double l2norm_similar_thresh = 1.128;

int main() {
    auto frame = cv::imread("./15.jpg");
    auto frame2 = cv::imread("./17.png");

    auto net_d  = cv::dnn::readNet("./models/face/face_detection_yunet_2022mar.onnx");
    auto net_r = cv::dnn::readNet("./models/face/face_recognition_sface_2021dec.onnx");

    net_d.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net_d.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    net_r.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net_r.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    inputW = frame.cols;
    inputH = frame.rows;

    // resize(image, image, cv::Size(inputW, inputH));
    generatePriors();

    auto blob = cv::dnn::blobFromImages(std::vector<cv::Mat>{frame});
    // auto blob = cv::dnn::blobFromImage(image);
    net_d.setInput(blob);
    std::vector<cv::Mat> outputs;
    std::vector<std::string> out_names {"loc", "conf", "iou"};

    net_d.forward(outputs, out_names);
    auto faces = postProcess(outputs);

    // --------------

    inputW = frame2.cols;
    inputH = frame2.rows;
    // resize(image, image, cv::Size(inputW, inputH));
    generatePriors();
    
    auto blob2 = cv::dnn::blobFromImages(std::vector<cv::Mat>{frame2});
    // auto blob = cv::dnn::blobFromImage(image);
    net_d.setInput(blob2);
    std::vector<cv::Mat> outputs2;
    std::vector<std::string> out_names2 {"loc", "conf", "iou"};

    net_d.forward(outputs2, out_names2);
    auto faces2 = postProcess(outputs2);

    // --------------

    assert(faces.size() == 1);
    assert(faces2.size() == 1);

    auto face1_mat = faces[0].row(0);
    auto face2_mat = faces2[0].row(0);
    Mat face1 = frame(cv::Rect(int(face1_mat.at<float>(0, 0)), int(face1_mat.at<float>(0, 1)), int(face1_mat.at<float>(0, 2)), int(face1_mat.at<float>(0, 3))));
    Mat face2 = frame2(cv::Rect(int(face2_mat.at<float>(0, 0)), int(face2_mat.at<float>(0, 1)), int(face2_mat.at<float>(0, 2)), int(face2_mat.at<float>(0, 3))));

    cv::imshow("face_1", face1);
    cv::imshow("face_2", face2);

    Mat aligned_face_1, aligned_face_2;
    alignCrop(frame, face1_mat, aligned_face_1);
    alignCrop(frame2, face2_mat, aligned_face_2);

    cv::imshow("aligned_face_1", aligned_face_1);
    cv::imshow("aligned_face_2", aligned_face_2);

    Mat face_feature_1, face_feature_2;
    feature(aligned_face_1, face_feature_1, net_r);
    face_feature_1 = face_feature_1.clone();

    feature(aligned_face_2, face_feature_2, net_r);
    face_feature_2 = face_feature_2.clone();

    double cos_score = match(face_feature_1, face_feature_2, 0);
    double L2_score = match(face_feature_1, face_feature_2, 1);

    if (cos_score >= cosine_similar_thresh)
    {
        std::cout << "They have the same identity;";
    }
    else
    {
        std::cout << "They have different identities;";
    }
    std::cout << " Cosine Similarity: " << cos_score << ", threshold: " << cosine_similar_thresh << ". (higher value means higher similarity, max 1.0)\n";

    if (L2_score <= l2norm_similar_thresh)
    {
        std::cout << "They have the same identity;";
    }
    else
    {
        std::cout << "They have different identities.";
    }
    std::cout << " NormL2 Distance: " << L2_score << ", threshold: " << l2norm_similar_thresh << ". (lower value means higher similarity, min 0.0)\n";

    cv::waitKey(0);
    getchar();
}

#endif