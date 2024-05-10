

#include "../trt_yolov8_detector.h"
#include "../trt_yolov8_pose_detector.h"
#include "../trt_yolov8_seg_detector.h"
#include "../trt_yolov8_classifier.h"

int main(int argc, char** argv) {
    /* run command:
     * ./trt_yolov8_wts_2_engine [-det/-seg/-pose/-cls] [.wts] [.engine] [n/s/m/l/x/n2/s2/m2/l2/x2/n6/s6/m6/l6/x6] 
    */
    
    if (argc != 5) {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./trt_yolov8_wts_2_engine [-det/-seg/-pose/-cls] [.wts] [.engine] [n/s/m/l/x/n2/s2/m2/l2/x2/n6/s6/m6/l6/x6]" << std::endl;
    }
    
    std::string task_type = std::string(argv[1]);
    std::string wts_name = std::string(argv[2]);
    std::string engine_name = std::string(argv[3]);
    std::string sub_type = std::string(argv[4]);

    if (task_type == "-det") {
        trt_yolov8::trt_yolov8_detector detector;
        detector.wts_2_engine(wts_name, engine_name, sub_type);
    }
    else if (task_type == "-seg") {
        trt_yolov8::trt_yolov8_seg_detector detector;
        detector.wts_2_engine(wts_name, engine_name, sub_type);
    }
    else if (task_type == "-pose") {
        trt_yolov8::trt_yolov8_pose_detector detector;
        detector.wts_2_engine(wts_name, engine_name, sub_type);
    }
    else if (task_type == "-cls") {
        trt_yolov8::trt_yolov8_classifier classifier;
        classifier.wts_2_engine(wts_name, engine_name, sub_type);
    }
    else {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./trt_yolov8_wts_2_engine [-det/-seg/-pose/-cls] [.wts] [.engine] [n/s/m/l/x/n2/s2/m2/l2/x2/n6/s6/m6/l6/x6]" << std::endl;
    }

    return 0;
}