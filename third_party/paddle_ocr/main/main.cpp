/*
* test code for paddle_ocr
*/
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/freetype.hpp>
#include <iostream>
#include <vector>

#include "../include/paddleocr.h"
#include "../include/paddlestructure.h"

using namespace PaddleOCR;

int main(int argc, char **argv) {

  auto det_model_dir = "/windows2/zhzhi/video_pipe_c/main/models/text/ppocr/ch_PP-OCRv3_det_infer";
  auto rec_model_dir = "/windows2/zhzhi/video_pipe_c/main/models/text/ppocr/ch_PP-OCRv3_rec_infer";
  auto cls_model_dir = "/windows2/zhzhi/video_pipe_c/main/models/text/ppocr/ch_ppocr_mobile_v2.0_cls_infer";
  auto rec_char_dict_path = "/windows2/zhzhi/video_pipe_c/third_party/paddle_ocr/ppocr_keys_v1.txt";
  auto ppocr = PPOCR(det_model_dir, cls_model_dir, rec_model_dir, rec_char_dict_path);

  auto video_path = "/windows2/zhzhi/video_pipe_c/main/test_video/4.mp4";
  auto capture = cv::VideoCapture(video_path);
  auto ft2 = cv::freetype::createFreeType2();
  ft2->loadFontData("../font/NotoSansCJKsc-Medium.otf", 0);

  while (true) {
    cv::Mat frame;
    if(!capture.read(frame)) {
      break;
    }
    cv::Mat frame1;
    frame.copyTo(frame1);

    cv::Mat frame2(frame.rows, frame.cols, frame.type(), cv::Scalar(255, 255, 255));
    cv::Mat img_vis(frame.rows * 2, frame.cols, frame.type(), cv::Scalar(255, 255, 255));


    std::vector<cv::Mat> frames;
    //cv::resize(frame, frame, cv::Size(), 0.5, 0.5);
    frames.push_back(frame);

    auto start = std::chrono::system_clock::now();
    auto ocr_results = ppocr.ocr(frames);
    auto delta = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start);
    std::cout << "time cost:===" << delta.count() << std::endl;

    assert(ocr_results.size() == 1);

    auto& ocr_result = ocr_results[0];


    for (int n = 0; n < ocr_result.size(); n++) {
      cv::Point rook_points[4];
      for (int m = 0; m < ocr_result[n].box.size(); m++) {
        rook_points[m] =
            cv::Point(int(ocr_result[n].box[m][0]), int(ocr_result[n].box[m][1]));
      }

      const cv::Point *ppt[1] = {rook_points};
      int npt[] = {4};
      cv::polylines(frame1, ppt, npt, 1, 1, CV_RGB(0, 255, 0), 2, cv::LINE_AA, 0);
      cv::polylines(frame2, ppt, npt, 1, 1, CV_RGB(0, 255, 0), 1, cv::LINE_AA, 0);
      //cv::putText(img_vis, ocr_result[n].text, rook_points[0], 1, 0.8, CV_RGB(0, 255, 0));
      ft2->putText(frame2, ocr_result[n].text, rook_points[3], 20, cv::Scalar(255, 0, 0), cv::FILLED, cv::LINE_AA, true);
    }

    auto roi1 = img_vis(cv::Rect(0, 0, frame.cols, frame.rows));
    auto roi2 = img_vis(cv::Rect(0, frame.rows, frame.cols, frame.rows));

    frame1.copyTo(roi1);
    frame2.copyTo(roi2);

    //cv::imshow("paddle ocr 2", frame2);
    //cv::imshow("paddle ocr 1", frame1);
    cv::resize(img_vis, img_vis, cv::Size(), 0.5, 0.5);
    cv::imshow("paddle ocr 3", img_vis);
    cv::waitKey(40);
  }
  
  return 1;
}
