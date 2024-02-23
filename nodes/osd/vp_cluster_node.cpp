#include "vp_cluster_node.h"
#include "../../third_party/bhtsne/tsne.h"

namespace vp_nodes {
    vp_cluster_node::vp_cluster_node(std::string node_name, 
                        bool use_tSNE, 
                        std::vector<std::string> s_labels_to_display,
                        int sampling_frequency,
                        int min_sampling_width,
                        int min_sampling_height,
                        int max_sample_num_for_tsne,
                        int max_sample_num_per_category):
                        vp_node(node_name),
                        use_tSNE(use_tSNE),
                        s_labels_to_display(s_labels_to_display),
                        sampling_frequency(sampling_frequency),
                        min_sampling_width(min_sampling_width),
                        min_sampling_height(min_sampling_height),
                        max_sample_num_for_tsne(max_sample_num_for_tsne),
                        max_sample_num_per_category(max_sample_num_per_category) {
        this->initialized();
    }
    
    vp_cluster_node::~vp_cluster_node() {
        deinitialized();        
    }

    // please refer to ../../third_party/trt_vehicle/main/vehicle_cluster.cpp
    void vp_cluster_node::reduce_dims_using_tsne(std::vector<std::vector<float>>& low_features,
                                int no_dims, int max_iter, double perplexity, 
                                double theta, int rand_seed, bool skip_random_init, 
                                int stop_lying_iter, int mom_switch_iter) {
        assert(cache_high_features.size() != 0);
        auto N = cache_high_features.size();
        auto D = cache_high_features[0].second.size();  // all the same as the first feature's dims

        // prepare input
        double data[N * D];
        double Y[N * no_dims];
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < D; j++) {
                data[i * D + j] = cache_high_features[i].second[j];
            }
        }
        
        // call t-SNE
        TSNE::run(data, N, D, Y, no_dims, perplexity, theta, rand_seed, skip_random_init, max_iter, stop_lying_iter, mom_switch_iter);

        // prepare output
        for (int i = 0; i < N; i++) {
            std::vector<float> low_dims_feature;
            for (int j = 0; j < no_dims; j++) {
                low_dims_feature.push_back(float(Y[i * no_dims + j]));
            }
            low_features.push_back(low_dims_feature);
        }
    }


    std::shared_ptr<vp_objects::vp_meta> vp_cluster_node::handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) {
        // sampling frequency
        if (std::chrono::duration_cast<std::chrono::milliseconds>(NOW - last_sampling_time) < std::chrono::milliseconds(sampling_frequency)) {
            return meta;
        }
        last_sampling_time = NOW;
        
        /* t-SNE */
        if (use_tSNE) {
            for (int i = 0; i < meta->targets.size(); i++) {
                if (meta->targets[i]->width < min_sampling_width || meta->targets[i]->height < min_sampling_height) {
                    continue;
                }
                
                if (cache_high_features.size() >= max_sample_num_for_tsne) {
                    // too many cache, we need control number of samples
                    cache_high_features.erase(cache_high_features.begin());
                }
                // save mat->feature pair to cache
                cv::Mat m;
                cv::Mat roi(meta->frame, cv::Rect(meta->targets[i]->x, meta->targets[i]->y, meta->targets[i]->width, meta->targets[i]->height));
                roi.copyTo(m);
                std::pair<cv::Mat, std::vector<float>> p {m, meta->targets[i]->embeddings};
                cache_high_features.push_back(p);
            }

            // at least 10 samples to t-SNE
            if (cache_high_features.size() >= 10) {
                // reduce dims first
                std::vector<std::vector<float>> low_features;
                reduce_dims_using_tsne(low_features);

                // now display on screen
                // normalize low dims feature to coordinate of [0:1] and display them on 2D screen
                auto max_x = 0.0f, max_y = 0.0f, min_x = 0.0f, min_y = 0.0f;
                for(int i = 0; i < low_features.size(); i++) {
                    auto& f = low_features[i];
                    // 2 values in f
                    max_x = std::max(max_x, f[0]);
                    max_y = std::max(max_y, f[1]);
                    min_x = std::min(min_x, f[0]);
                    min_y = std::min(min_y, f[1]);
                }
                auto x_range = max_x - min_x;
                auto y_range = max_y - min_y;

                // draw on (tsne_canvas_w_h + tsne_thumbnail_w_h) * (tsne_canvas_w_h + tsne_thumbnail_w_h)
                cv::Mat canvas(tsne_canvas_w_h + tsne_thumbnail_w_h, tsne_canvas_w_h + tsne_thumbnail_w_h, CV_8UC3, cv::Scalar(127, 127, 127));
                for(int i = 0; i < low_features.size(); i++) {
                    auto& f = low_features[i];
                    // convert to [0:1]
                    f[0] = (f[0] - min_x) / x_range;
                    f[1] = (f[1] - min_y) / y_range;

                    auto& img = cache_high_features[i].first;
                    cv::Mat img_tmp;
                    cv::resize(img, img_tmp, cv::Size(tsne_thumbnail_w_h, tsne_thumbnail_w_h));
                    cv::rectangle(img_tmp, cv::Rect(0, 0, img_tmp.cols, img_tmp.rows), cv::Scalar(255, 0, 0));
                    cv::Mat roi(canvas, cv::Rect(int(f[0] * tsne_canvas_w_h), int(f[1] * tsne_canvas_w_h), tsne_thumbnail_w_h, tsne_thumbnail_w_h));
                    img_tmp.copyTo(roi);
                }

                cv::imshow("cluster using features powered by t-SNE", canvas);
            }
        }

        /* categories */
        for (int i = 0; i < meta->targets.size(); i++) {
            auto& t = meta->targets[i];
            if (t->width < min_sampling_width || t->height < min_sampling_height) {
                continue;
            }

            cv::Mat m;
            cv::Mat roi(meta->frame, cv::Rect(t->x, t->y, t->width, t->height));
            roi.copyTo(m);

            for (int j = 0; j < t->secondary_labels.size(); j++) {
                auto& category = t->secondary_labels[j];
                bool filter_pass = false;
                // all
                if (s_labels_to_display.size() == 0) {
                    cache_categories[category].push_back(m);
                    filter_pass = true;
                }
                else {
                    // has a filter
                    if (std::find(s_labels_to_display.begin(), s_labels_to_display.end(), category) != s_labels_to_display.end()) {
                        cache_categories[category].push_back(m);
                        filter_pass = true;
                    }
                }

                if (filter_pass) {                
                    // too many cache, we need control number of samples
                    if (cache_categories[category].size() >= max_sample_num_per_category) {
                        cache_categories[category].erase(cache_categories[category].begin());
                    }
                }
            }
        }
        // display on screen
        // calculate total number of rows
        auto rows_num = 0; 
        for (auto& p: cache_categories) {
            auto num = p.second.size() / category_num_per_row;
            if (p.second.size() % category_num_per_row != 0 || num == 0) {
                num++;
            }
            rows_num += num;
        }
        
        // draw on (category_canvas_w) * (category_canvas_h)
        auto category_canvas_w = (category_thumbnail_w_h + category_gap) * (category_num_per_row + 1);
        auto category_canvas_h = (category_thumbnail_w_h + category_gap) * (rows_num + 1);
        cv::Mat canvas(category_canvas_h, category_canvas_w, CV_8UC3, cv::Scalar(127, 127, 127));

        auto row_index = 0;
        for (auto& p: cache_categories) {
            auto col_index = 0;
            auto& category_items = p.second;
            auto& category_name = p.first;
            cv::putText(canvas, category_name, cv::Point(10, (category_gap + category_thumbnail_w_h) * row_index + category_gap), 1, 1, cv::Scalar(0, 0, 255));

            bool new_row = false;
            for (int i = 0; i < category_items.size(); i++) {
                cv::Mat img_tmp;
                cv::resize(category_items[i], img_tmp, cv::Size(category_thumbnail_w_h, category_thumbnail_w_h));
                cv::Mat roi(canvas, cv::Rect((category_gap + category_thumbnail_w_h) * col_index + category_gap, (category_gap + category_thumbnail_w_h) * row_index + category_gap, category_thumbnail_w_h, category_thumbnail_w_h));
                img_tmp.copyTo(roi);

                col_index++;
                if ((col_index + 1) % category_num_per_row == 0) {
                    col_index = 0;
                    row_index ++;
                    new_row = true;
                } 
                else {
                    new_row = false;
                }
            }
            if (!new_row) {
                row_index++;
            }
        }

        cv::imshow("cluster using labels", canvas);

        return meta;
    }

    std::shared_ptr<vp_objects::vp_meta> vp_cluster_node::handle_control_meta(std::shared_ptr<vp_objects::vp_control_meta> meta) {
        return meta;
    }
}