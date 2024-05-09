

#include "../trt_yolov8_seg_detector.h"

int main() {
    trt_yolov8::trt_yolov8_seg_detector detector("./vp_data/models/trt/others/yolov8s-seg_v8.5.engine");
    
    cv::VideoCapture cap("./vp_data/test_video/face2.mp4");
    std::unordered_map<int, std::string> labels_map;
    read_labels("./vp_data/models/coco_80classes.txt", labels_map);

    cv::Mat frame;
    while (true) {
        if (!cap.read(frame)) {
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            continue;
        }
        
        std::vector<std::vector<Detection>> detections;
        std::vector<std::vector<cv::Mat>> masks;
        std::vector<cv::Mat> frames = {frame};
        detector.detect(frames, detections, masks);

        draw_mask_bbox(frame, detections[0], masks[0], labels_map);
        cv::imshow("seg", frame);
        cv::waitKey(40);
    }
    return 0;
}