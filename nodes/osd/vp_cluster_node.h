#pragma once

#include "../vp_node.h"

namespace vp_nodes {
    // cluster node for vp_frame_targets which has ability to display targets on screen according to its embeddings contained in vp_frame_target::embbedings variable or labels contained in vp_frame_target::secondary_labels vector。
    // note!!!
    // it is not an osd node which would operates on vp_frame_meta::osd_frame data member.
    // it is not a DES node either which can be the last node in pipeline.
    // it is just a normal MID node.
    class vp_cluster_node: public vp_node
    {
    private:
        // call tSNE algorithm to reduce high dims of feature and display target on 2D screen
        bool use_tSNE;
        // display target based on categories, empty means for all categories. if you want to disable it, just let vector including just a wrong category name take '123abcd' for example.
        std::vector<std::string> s_labels_to_display;
        // how often to sampling (miliseconds). since targets differences in adjacent frames are small, we need not sampling continuously.
        int sampling_frequency = 1000;
        std::chrono::system_clock::time_point last_sampling_time = NOW;

        // filter for small targets
        int min_sampling_width = 40;
        int min_sampling_height = 40;

        /* draw parameters for tsne */
        int tsne_canvas_w_h = 600;
        int tsne_thumbnail_w_h = 60;
        int max_sample_num_for_tsne = 100;
        std::vector<std::pair<cv::Mat, std::vector<float>>> cache_high_features;  // mat -> feature

        /* draw parameters for category */
        int category_num_per_row = 10;
        int category_thumbnail_w_h = 60;
        int category_gap = 10;
        int max_sample_num_per_category = 10;
        std::map<std::string, std::vector<cv::Mat>> cache_categories;  // label -> mat list
        
        // reduce dims of features (to xy) so as to display it on 2D screen 
        void reduce_dims_using_tsne(std::vector<std::vector<float>>& low_features,
                /* default parameters for t-SNE algorithm */
                int no_dims = 2, int max_iter = 500, double perplexity = 2, double theta = 0.5, int rand_seed = -1, bool skip_random_init = false, int stop_lying_iter = 250, int mom_switch_iter = 250);
    protected:
        virtual std::shared_ptr<vp_objects::vp_meta> handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) override;
        virtual std::shared_ptr<vp_objects::vp_meta> handle_control_meta(std::shared_ptr<vp_objects::vp_control_meta> meta) override;
    public:
        vp_cluster_node(std::string node_name, 
                        bool use_tSNE = true, 
                        std::vector<std::string> s_labels_to_display = std::vector<std::string>{},
                        int sampling_frequency = 1000,
                        int min_sampling_width = 40,
                        int min_sampling_height = 40,
                        int max_sample_num_for_tsne = 100,
                        int max_sample_num_per_category = 10);
        ~vp_cluster_node();
    };
}