

#include "../trt_yolov8_pose_detector.h"

int main() {
    trt_yolov8::trt_yolov8_pose_detector detector("./vp_data/models/trt/others/yolov8s-pose_v8.5.engine");

    cv::VideoCapture cap("./vp_data/test_video/face2.mp4");
    cv::Mat frame;
    while (true) {
        if (!cap.read(frame)) {
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            continue;
        }
        
        std::vector<std::vector<Detection>> detections;
        std::vector<cv::Mat> frames = {frame};
        detector.detect(frames, detections);

        draw_bbox_keypoints_line(frames, detections);
        cv::imshow("pose", frame);
        cv::waitKey(40);
    }
    return 0;
}