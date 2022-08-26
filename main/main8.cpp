#include <vector>
#include <iostream>
#include <memory>

#include "../nodes/vp_file_src_node.h"
#include "../nodes/vp_screen_des_node.h"
#include "../nodes/vp_split_node.h"
#include "../nodes/vp_rtmp_des_node.h"
#include "../nodes/osd/vp_osd_node.h"
#include "../nodes/vp_track_node.h"
#include "../nodes/vp_rtsp_src_node.h"
#include "../nodes/vp_udp_src_node.h"
#include "../nodes/vp_message_broker_node.h"
#include "../utils/analysis_board/vp_analysis_board.h"
#include "../utils/statistics_board/vp_statistics_board.h"

#include "../nodes/infers/vp_yolo_detector_node.h"
#include "VP.h"

#if MAIN8

#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


// connection table, in the format [model_id][pair_id][from/to]
// please look at the nice explanation at the bottom of:
// https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md
//
const int POSE_PAIRS[4][24][2] = {
{   // BODY_25 body, 25 parts, 24 pairs
    {1,2}, {1,5}, {2,3},
    {3,4}, {5,6}, {6,7},
    {1,8}, {8,9}, {9,10}, {10,11}, {11,22}, {22,23}, {11,24},
    {8,12}, {12,13}, {13,14}, {14,19}, {19,20}, {14,21},
    {0,15}, {0,16}, {15,17}, {16,18},
    {0,1}
},
{   // COCO body, 18 parts, 17 pairs
    {1,2}, {1,5}, {2,3},
    {3,4}, {5,6}, {6,7},
    {1,8}, {8,9}, {9,10},
    {1,11}, {11,12}, {12,13},
    {1,0}, {0,14},
    {14,16}, {0,15}, {15,17}
},
{   // MPI body, 16 parts, 14 pairs
    {0,1}, {1,2}, {2,3},
    {3,4}, {1,5}, {5,6},
    {6,7}, {1,14}, {14,8}, {8,9},
    {9,10}, {14,11}, {11,12}, {12,13}
},
{   // hand, 21 parts, 20 pairs
    {0,1}, {1,2}, {2,3}, {3,4},         // thumb
    {0,5}, {5,6}, {6,7}, {7,8},         // pinkie
    {0,9}, {9,10}, {10,11}, {11,12},    // middle
    {0,13}, {13,14}, {14,15}, {15,16},  // ring
    {0,17}, {17,18}, {18,19}, {19,20}   // small
}};


int main() {
    auto net = cv::dnn::readNet("./models/openpose/pose/coco_pose_iter_440000.caffemodel", "./models/openpose/pose/coco_pose_deploy.prototxt");
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    auto capture = cv::VideoCapture("./1.mp4");

    int midx = 1, npairs = 17, nparts = 18;
    float threshold = 0.1;

    while (1) {
        cv::Mat frame;
        
        if (!capture.read(frame)) {
            capture.set(cv::CAP_PROP_POS_FRAMES, 0);
            continue;
        }

        //frame = cv::imread("./12.png");
        auto start = std::chrono::system_clock::now();
        auto blob = cv::dnn::blobFromImage(frame, 1 / 255.0, cv::Size(368, 368), cv::Scalar(0), false, false);

        net.setInput(blob);
        std::vector<cv::Mat> results;
        cv::Mat result = net.forward();
        // the result is an array of "heatmaps", the probability of a body part being in location x,y

        auto delta = std::chrono::system_clock::now() - start;

        std::cout << "time cost: " << std::chrono::duration_cast<std::chrono::milliseconds>(delta).count() << std::endl;
        // the first output, only one output
        // n,nparts,h,w
        //auto result = results[0];

        int H = result.size[2];
        int W = result.size[3];

        std::cout << H << "---" << W << std::endl;

        // find the position of the body parts
        vector<cv::Point> points(nparts);
        for (int n = 0; n < nparts; n++) {
            // Slice heatmap of corresponding body's part.
            cv::Mat heatMap(H, W, CV_32F, result.ptr(0, n));
            // 1 maximum per heatmap
            cv::Point p(-1,-1), pm;
            double conf;
            cv::minMaxLoc(heatMap, 0, &conf, 0, &pm);
            if (conf >= threshold)
                p = pm;
            points[n] = p;

        }

        // connect body parts and draw it !
        float SX = float(frame.cols) / W;
        float SY = float(frame.rows) / H;
        for (int n = 0; n < npairs; n++) {
            // lookup 2 connected body/hand parts
            cv::Point2f a = points[POSE_PAIRS[midx][n][0]];
            cv::Point2f b = points[POSE_PAIRS[midx][n][1]];

            // we did not find enough confidence before
            if (a.x<=0 || a.y<=0 || b.x<=0 || b.y<=0)
                continue;

            // scale to image size
            a.x*=SX; a.y*=SY;
            b.x*=SX; b.y*=SY;

            cv::line(frame, a, b, cv::Scalar(0,200,0), 2);
            cv::circle(frame, a, 3, cv::Scalar(0,0,200), -1);
            cv::circle(frame, b, 3, cv::Scalar(0,0,200), -1);

            std::cout << a.x << "-" << b.y << std::endl;
        }

        cv::imshow("OpenPose", frame);
        cv::waitKey(1);
    }
    
}

#endif