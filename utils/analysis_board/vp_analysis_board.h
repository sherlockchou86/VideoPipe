#pragma once

#include <vector>
#include <map>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include "../vp_utils.h"
#include "../../nodes/vp_node.h"
#include "../../nodes/vp_src_node.h"
#include "../../objects/vp_meta.h"
#include "../../objects/shapes/vp_rect.h"

#include "vp_node_on_screen.h"

namespace vp_utils {
    class vp_analysis_board final
    {
    private:
        // configure for render
        const int node_width = 140;
        const int node_height = 140;
        const int canvas_gap_horizontal = 120;
        const int canvas_gap_vertical = 60;
        const int node_gap_horizontal = 40;
        const int node_gap_vertical = 10;

        int canvas_width = 0;
        int canvas_height = 0;

        std::string gst_template = "appsrc ! videoconvert ! x264enc bitrate=%d ! h264parse ! flvmux ! rtmpsink location=%s";
        cv::VideoWriter rtmp_writer;

        // 
        bool displaying = false;
        
        // width of pipe
        int pipe_width;
        // height of pipe
        int pipe_height;

        // start points of pipe
        std::vector<std::shared_ptr<vp_nodes::vp_node>> src_nodes_in_pipe;

        // cache for easy access purpose
        std::vector<std::shared_ptr<vp_node_on_screen>> src_nodes_on_screen;
        std::vector<std::shared_ptr<vp_node_on_screen>> des_nodes_on_screen;

        // canvas to draw
        cv::Mat bg_canvas;
        
        // display thread(on screen)
        std::thread display_th;

        // display thread(via rtmp)
        std::thread rtmp_th;

        // render nodes in a layer
        void render_layer(std::vector<std::shared_ptr<vp_node_on_screen>> nodes_in_layer, cv::Mat& canvas, bool static_parts = true);

        // map nodes in memory to screen, one layer by layer.
        void map_nodes(std::vector<std::shared_ptr<vp_node_on_screen>> nodes_on_screen, int layer);
        
        // tool methods
        std::function<int(int)> layer_base_left_cal = [=](int layer_index) {return canvas_gap_horizontal + layer_index * ( node_width + node_gap_horizontal);};
        std::function<int(int)> layer_base_top_cal = [=](int num_nodes_in_layer) {return (canvas_height - (num_nodes_in_layer * node_height + (num_nodes_in_layer - 1) * node_gap_vertical)) / 2; };
    public:
        vp_analysis_board(std::vector<std::shared_ptr<vp_nodes::vp_node>> src_nodes_in_pipe);
        ~vp_analysis_board();

        // save pipe structure to png
        void save(std::string path);

        // display pipe on screen and refresh it automatically
        void display(int interval = 1, bool block = true);

        // display pipe by rtmp and refresh it automatically
        void push_rtmp(std::string rtmp, int bitrate = 1024);
    };
}