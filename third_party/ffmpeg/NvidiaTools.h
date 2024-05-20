//
// Created by lijin on 2023/7/28.
//

#ifndef VIDEO_FUSION_2_0_NVIDIATOOLS_H
#define VIDEO_FUSION_2_0_NVIDIATOOLS_H


#include <dlfcn.h>
#include <dlfcn.h>
#include "FileOperate.h"

namespace vp {

    static bool string_start_with(const std::string &str, const std::string &substr) {
        return str.find(substr) == 0;
    }

    static bool checkIfSupportedNvidia_l() {
        auto so = dlopen("libnvcuvid.so.1", RTLD_LAZY);
        if (!so) {
            std::cout << "libnvcuvid.so.1加载失败:" << std::endl;
            return false;
        }
        dlclose(so);
        bool find_driver = false;
        utils::FileOperate::scanDir("/dev", [&](const std::string &path, bool is_dir) {
            if (!is_dir && string_start_with(path, "/dev/nvidia")) {
                //找到nvidia的驱动
                find_driver = true;
                return false;
            }
            return true;
        }, false);

        if (!find_driver) {
            std::cout << "英伟达硬件编解码器驱动文件 /dev/nvidia* 不存在";
        }
        return find_driver;
    }

    static bool checkIfSupportedNvidia() {
        static auto ret = checkIfSupportedNvidia_l();
        return ret;
    }


}


#endif //VIDEO_FUSION_2_0_NVIDIATOOLS_H
