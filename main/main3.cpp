#include <vector>
#include <iostream>
#include <memory>


#include "../nodes/vp_file_src_node.h"
#include "../nodes/vp_primary_infer_node.h"
#include "../nodes/vp_screen_des_node.h"
#include "../nodes/vp_split_node.h"
#include "VP.h"


#if MAIN3

typedef void a_type(int, int);

std::queue<cv::Mat> q;

void func() {
    cv::VideoCapture cap("./1.mp4");
    
    cv::Mat frame;
    while (true)
    {
        /* code */

        cap >> frame;

        if (frame.empty())
        {
            break;
        }

        q.push(frame);
        std::this_thread::sleep_for(std::chrono::milliseconds(40));
    }
}

void func2() {
    cv::VideoWriter writer("appsrc ! videoconvert ! video/x-raw,framerate=25/1 ! textoverlay text=111 halignment=left valignment=top font-desc='Sans,16' shaded-background=true ! timeoverlay halignment=right valignment=top font-desc='Sans,16' shaded-background=true ! queue ! ximagesink sync=false", cv::CAP_GSTREAMER, 0, 25, cv::Size(1920, 1080));
    while (true)
    {
        if (q.size() > 0)
        {
            auto frame = q.front();
            q.pop();
            writer.write(frame);
        }
        else
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
}


int main() {
    std::thread th = std::thread(func);
    std::thread th2 = std::thread(func2);

    th.join();
    th2.join();
    return 0;
}

#endif