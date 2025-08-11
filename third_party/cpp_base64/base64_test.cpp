#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include "base64.h"

/*
 * convert base64 string to cv::Mat
*/
cv::Mat base64ToMat(const std::string& base64_data) {
    std::string decoded_data = base64_decode(base64_data);
    std::vector<uchar> data(decoded_data.begin(), decoded_data.end());
    cv::Mat img = cv::imdecode(data, cv::IMREAD_UNCHANGED);
    return img;
}

/*
 * convert cv::Mat to base64 string
*/
std::string matToBase64(const cv::Mat& img, const std::string& ext = ".jpg") {
    std::vector<uchar> buf;
    cv::imencode(ext, img, buf);
    std::string encoded = base64_encode(buf.data(), buf.size());
    return encoded;
}

/*
 * read image into mat, convert it to base64 string and convert back to mat
 */
int main() {
    auto ori_mat = cv::imread("/windows2/zhzhi/github/vp_data/test_images/vehicle/0.jpg");
    auto base64_str = matToBase64(ori_mat);
    auto decoded_mat = base64ToMat(base64_str);

    std::cout << "based64_str: " << base64_str << std::endl;
    std::cout << "based64_str's length: " << base64_str.size() << std::endl;
    std::cout << "ori mat's byte size: " << ori_mat.cols * ori_mat.rows * ori_mat.elemSize() << std::endl;

    cv::imshow("ori_mat", ori_mat);
    cv::imshow("decoded_mat", decoded_mat);

    cv::waitKey(0);
}