#pragma once

#include "detect_model.h"
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>


namespace trt_vehicle{

    struct Plate {
        int x;
        int y;
        int width;
        int height;
        std::string color;
        std::string text;
    };

    class VehiclePlateDetector{
    public:
        VehiclePlateDetector(const std::string& plateModelPath,
                        const std::string& charModelPath,
                        float plateScoreThres = 0.3,
                        float plateIouThres = 0.1,
                        float charScoreThres = 0.5,
                        float charIouThres = 0.25);
        ~VehiclePlateDetector();
        void detect(std::vector<cv::Mat>& img_datas, std::vector<std::vector<Plate>>&, bool only_one = false);

    private:
        DetectModel* plateModel = nullptr;
        DetectModel* charModel = nullptr;
        std::vector<std::string> chars = {"0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","WJ","J",\
            "K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","云","粤","鄂","豫","渝","新","湘","皖","苏","陕",\
            "琼","青","宁","闽","蒙","鲁","辽","京","晋","津","冀","吉","沪","黑","桂","贵","赣","甘","浙","川","藏","挂","学","警","防"};
        std::vector<std::string> colors = {"blue", "yellow", "green", "white"};
    };
}
