#include "../nodes/vp_file_src_node.h"
#include "../nodes/infers/vp_yunet_face_detector_node.h"
#include "../nodes/infers/vp_sface_feature_encoder_node.h"
#include "../nodes/osd/vp_face_osd_node_v2.h"
#include "../nodes/vp_screen_des_node.h"
#include "../nodes/vp_rtmp_des_node.h"

#include "../utils/analysis_board/vp_analysis_board.h"
#include "gst/gst.h"
#include "../nodes/vp_node.h"

/*
* ## 1-1-1 sample ##
* 1 video input, 1 infer task, and 1 output.
*/

int main(int argc ,char * argv[]) {

    std::cout << "当前目录: " << std::filesystem::current_path() << std::endl;
    VP_LOGGER_INIT();

    _putenv_s("GST_PLUGIN_PATH","D:\\Works\\github\\vcpkg\\installed\\x64-windows\\debug\\plugins\\gstreamer");

    // 初始化 GStreamer
    gst_init(&argc, &argv);

    // 获取插件注册表
    GstRegistry* registry = gst_registry_get();
    GList* plugins = gst_registry_get_plugin_list(registry);

    std::cout << "已注册的 GStreamer 插件列表：" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    for (GList* l = plugins; l != nullptr; l = l->next) {
        GstPlugin* plugin = GST_PLUGIN(l->data);
        const gchar* name = gst_plugin_get_name(plugin);
        const gchar* description = gst_plugin_get_description(plugin);
        std::cout << "- " << name << ": " << (description ? description : "无描述") << std::endl;
    }

    // 释放资源
    gst_plugin_list_free(plugins);

    VP_SET_LOG_INCLUDE_CODE_LOCATION(false);
    VP_SET_LOG_INCLUDE_THREAD_ID(false);
    
    // create nodes
    auto file_src_0 = std::make_shared<vp_nodes::vp_file_src_node>("file_src_0", 0, "./vp_data/test_video/face.mp4", 0.6);
    auto yunet_face_detector_0 = std::make_shared<vp_nodes::vp_yunet_face_detector_node>("yunet_face_detector_0", "./vp_data/models/face/face_detection_yunet_2022mar.onnx");
    auto sface_face_encoder_0 = std::make_shared<vp_nodes::vp_sface_feature_encoder_node>("sface_face_encoder_0", "./vp_data/models/face/face_recognition_sface_2021dec.onnx");
    auto osd_0 = std::make_shared<vp_nodes::vp_face_osd_node_v2>("osd_0");
    auto screen_des_0 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_0", 0);

    // construct pipeline
    yunet_face_detector_0->attach_to({file_src_0});
    sface_face_encoder_0->attach_to({yunet_face_detector_0});
    osd_0->attach_to({sface_face_encoder_0});
    screen_des_0->attach_to({osd_0});

    file_src_0->start();

    // for debug purpose
    vp_utils::vp_analysis_board board({file_src_0});
    board.display(1, false);

    std::string wait;
    std::getline(std::cin, wait);
    file_src_0->detach_recursively();
}