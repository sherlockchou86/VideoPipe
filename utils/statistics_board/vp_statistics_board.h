
#pragma once

#include <vector>
#include <map>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "../../nodes/vp_node.h"
#include "../../nodes/vp_src_node.h"
#include "../../objects/vp_meta.h"
#include "../../objects/shapes/vp_rect.h"

namespace vp_utils {

    class vp_statistics_board final
    {    
    private:
        // mainly used to store data from meta_hookers' callback
        struct vp_hooker_storage {
            vp_objects::vp_rect node_rect;       // node location on canvas
            int queue_size;                      // size of in/out queue of node
            int called_count_since_epoch_start;  // used for fps
            std::chrono::system_clock::time_point time_epoch_start;  // used for fps
            std::shared_ptr<vp_objects::vp_meta> meta;               // the latest meta (ptr) pushed to queue / poped from queue inside node 
        };

        int font_face = cv::FONT_HERSHEY_SIMPLEX;

        std::thread display_th;
        // epoch to calculate fps, milliseconds
        int fps_epoch = 2000;

        // start points of pipe
        std::vector<std::shared_ptr<vp_nodes::vp_node>> pipe_src_nodes;
        // canvas to draw
        cv::Mat bg_canvas;

        // width of pipe
        int pipe_width;
        // height of pipe
        int pipe_height;

        // meta hookers
        vp_nodes::vp_meta_hooker arriving_hooker;
        vp_nodes::vp_meta_hooker handling_hooker;
        vp_nodes::vp_meta_hooker handled_hooker;
        vp_nodes::vp_meta_hooker leaving_hooker;
        vp_nodes::vp_stream_info_hooker stream_info_hooker;

        // store data from meta_hookers' callbacks, will be updated frequently.
        // node_name -> vp_hooker_storage
        std::map<std::string, vp_hooker_storage> arriving_hooker_storages;
        std::map<std::string, vp_hooker_storage> handling_hooker_storages;
        std::map<std::string, vp_hooker_storage> handled_hooker_storages;
        std::map<std::string, vp_hooker_storage> leaving_hooker_storages;

        // store data from stream_info_hooker's callback
        std::map<std::string, vp_nodes::vp_stream_info> stream_info_hooker_storages;

        // render static parts on canvas as background.
        void render_static_parts();
        // render a layer for pipe with recursion
        void render_layer(std::vector<std::shared_ptr<vp_nodes::vp_node>> nodes_in_layer, int layer);
        // render dynamic parts based on static parts, for example, refresh UI every interval seconds.
        void render_dynamic_parts(cv::Mat& ouput);

        // configure for render
        const int node_width = 120;
        const int node_height = 140;
        const int node_gap_horizontal = 40;
        const int node_gap_vertical = 10;
        const int canvas_gap_horizontal = 120;
        const int canvas_gap_vertical = 60;
        const int node_title_h = 24;
        const int node_queue_width = 30;

        const int node_queue_port_w_h = 6;
        const int node_queue_port_padding = 8;

        const int node_handle_logic_radius = 10;

        int canvas_width = 0;
        int canvas_height = 0;

        

    public:
        vp_statistics_board(std::vector<std::shared_ptr<vp_nodes::vp_node>> pipe_src_nodes);
        ~vp_statistics_board();

        // save pipeline to disk as png
        void save_graph(std::string path);

        // display pipe on screen, refresh every 1 second by default.
        void display(int interval = 1, bool block = true);
    };
}