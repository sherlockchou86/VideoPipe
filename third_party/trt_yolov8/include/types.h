#pragma once
#include <string>
#include "config.h"

struct alignas(float) Detection {
    //center_x center_y w h
    float bbox[4];
    float conf;  // bbox_conf * cls_conf
    int class_id;
    float mask[32];
    float keypoints[51];  // 17*3 keypoints
};

struct Classification {
    int class_id;
    float conf;
};

struct AffineMatrix {
    float value[6];
};

const int bbox_element =
        sizeof(AffineMatrix) / sizeof(float) + 1;  // left, top, right, bottom, confidence, class, keepflag
