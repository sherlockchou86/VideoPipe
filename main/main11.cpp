

#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "VP.h"


#if MAIN11

int main() {
    cv::VideoCapture capture("./6.mp4");
    cv::VideoWriter writer;
    cv::Mat frame;

    while(1) {
        if (!capture.read(frame)) {
            continue;
        }

        if (!writer.isOpened()) {
            auto video_width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
            auto video_height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
            auto fps = capture.get(cv::CAP_PROP_FPS);
            writer.open("appsrc ! videoconvert ! ximagesink", fps, 25, cv::Size(video_width, video_height));
        }
        
        writer.write(frame);

        cv::waitKey(20);
    }
}

#endif