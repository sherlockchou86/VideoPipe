
#include <fstream>

#include "vp_infer_node.h"

namespace vp_nodes {
    vp_infer_node::vp_infer_node(std::string node_name, 
                            vp_infer_type infer_type, 
                            std::string model_path, 
                            std::string model_config_path, 
                            std::string labels_path, 
                            int input_width, 
                            int input_height, 
                            int batch_size,
                            float scale,
                            cv::Scalar mean,
                            cv::Scalar std,
                            bool swap_rb,
                            bool swap_chn):
                            vp_node(node_name),
                            infer_type(infer_type),
                            model_path(model_path),
                            model_config_path(model_config_path),
                            labels_path(labels_path),
                            input_width(input_width),
                            input_height(input_height),
                            batch_size(batch_size),
                            scale(scale),
                            mean(mean),
                            std(std),
                            swap_rb(swap_rb),
                            swap_chn(swap_chn) {
        // try to load network from file,
        // failing means maybe it has a custom implementation for model loading in derived class such as using other backends other than opencv::dnn.
        try {
            net = cv::dnn::readNet(model_path, model_config_path);
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        }
        catch(const std::exception& e) {
            VP_WARN(vp_utils::string_format("[%s] cv::dnn::readNet load network failed!", node_name.c_str()));
        }

        // load labels if labels_path is specified
        if (labels_path != "") {
            load_labels();
        }

        assert(batch_size > 0);
        // primary infer nodes can handle frame meta batch by batch(whole frame), 
        // others can handle multi batchs ONLY inside a single frame(small croped image).
        if (infer_type == vp_infer_type::PRIMARY && batch_size > 1) {
            frame_meta_handle_batch = batch_size;
        }
    }
    
    vp_infer_node::~vp_infer_node() {

    } 

    // handle frame meta one by one
    std::shared_ptr<vp_objects::vp_meta> vp_infer_node::handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) {
        std::vector<std::shared_ptr<vp_objects::vp_frame_meta>> frame_meta_with_batch {meta};
        run_infer_combinations(frame_meta_with_batch);    
        return meta;
    }

    // handle frame meta batch by batch
    void vp_infer_node::handle_frame_meta(const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& meta_with_batch) {
        const auto& frame_meta_with_batch = meta_with_batch;
        run_infer_combinations(frame_meta_with_batch);
        // no return
    }

    // default implementation
    // infer batch by batch
    void vp_infer_node::infer(const cv::Mat& blob_to_infer, std::vector<cv::Mat>& raw_outputs) {
        // blob_to_infer is a 4D matrix
        // the first dim is number of batch
        assert(blob_to_infer.dims == 4);
        assert(!net.empty());
        
        auto number_of_batch = blob_to_infer.size[0];
        if (number_of_batch <= batch_size) {
            // infer one time directly
            net.setInput(blob_to_infer);
            net.forward(raw_outputs/*, net.getUnconnectedOutLayersNames()*/);
        }
        else {
            // infer more times
            int b_size[] = {batch_size, blob_to_infer.size[1], blob_to_infer.size[2], blob_to_infer.size[3]};
            auto times = (number_of_batch % batch_size) == 0 ? (number_of_batch / batch_size) : (number_of_batch / batch_size + 1);
            for (int i = 0; i < times; i++) {
                // split to small piece
                int i_hwc[] = {i * batch_size, 0, 0, 0};  // 4D
                auto ptr = blob_to_infer.ptr(i_hwc);
                cv::Mat b_blob(4, b_size, CV_32F, (void*)ptr);
                std::vector<cv::Mat> b_outputs;

                net.setInput(b_blob);
                net.forward(b_outputs/*, net.getUnconnectedOutLayersNames()*/);

                // first time, initialize it
                if (raw_outputs.size() == 0) {
                    // scan multi heads of output
                    for (int j = 0; j < b_outputs.size(); j++) {
                        if (batch_size == 1) {
                            // keep dims as usual, but change size[0] == number_of_batch
                            if (b_outputs[j].dims <= 2 && b_outputs[j].rows == 1) {
                                raw_outputs.push_back(cv::Mat(2, std::vector<int>{number_of_batch, b_outputs[j].cols}.data(), CV_32F));
                            }
                            else {
                                // dims add 1, and set size[0] == number_of_batch
                                std::vector<int> t_size;
                                t_size.push_back(number_of_batch);
                                for (int s = 0; s < b_outputs[j].dims; s++) {
                                    t_size.push_back(b_outputs[j].size[s]);
                                }
                                raw_outputs.push_back(cv::Mat(b_outputs[j].dims + 1, t_size.data(), CV_32F));
                            }
                        }
                        else {
                            // kepp dims as usual, but change size[0] == number_of_batch
                            std::vector<int> t_size;
                            t_size.push_back(number_of_batch);

                            // start from 1
                            for (int s = 1; s < b_outputs[j].dims; s++) {
                                /* code */
                                t_size.push_back(b_outputs[j].size[s]);
                            }
                            raw_outputs.push_back(cv::Mat(b_outputs[j].dims, t_size.data(), CV_32F));
                        }
                    }
                }

                assert(raw_outputs.size() == b_outputs.size());
                // merge data directly
                for (int j = 0; j < b_outputs.size(); j++) {
                    auto& des = raw_outputs[j];
                    auto& src = b_outputs[j];

                    std::vector<int> t_size;
                    auto s_dims_n = src.dims <= 2 ? 2 : src.dims;
                    for (int s = 0; s < s_dims_n; s++) {
                        t_size.push_back(src.size[s]);
                    }

                    auto ptr = des.ptr(i * batch_size);
                    cv::Mat tmp(s_dims_n, t_size.data(), CV_32F, (void*)ptr);

                    src.copyTo(tmp);
                }
            }
        }
    }

    // default implementation
    // create a 4D matrix(n, c, h, w)
    void vp_infer_node::preprocess(const std::vector<cv::Mat>& mats_to_infer, cv::Mat& blob_to_infer) {
        cv::dnn::blobFromImages(mats_to_infer, blob_to_infer, scale, cv::Size(input_width, input_height), mean, swap_rb);
        if (std != cv::Scalar(1)) {
            // divide by std
        }

        // NCHW -> NHWC
        if (swap_chn) {
            cv::Mat blob_to_infer_tmp;
            cv::transposeND(blob_to_infer, {0, 2, 3, 1}, blob_to_infer_tmp);
            blob_to_infer_tmp.copyTo(blob_to_infer);
        }
    }

    void vp_infer_node::run_infer_combinations(const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) {
        /*
        * call logic by default:
        * frame_meta_with_batch -> mats_to_infer -> blob_to_infer -> raw_outputs -> frame_meta_with_batch
        */
        std::vector<cv::Mat> mats_to_infer;
        // 4D matrix
        cv::Mat blob_to_infer;
        // multi heads of output in network, raw matrix output which need to be parsed by users.
        std::vector<cv::Mat> raw_outputs;

        // start
        auto start_time = std::chrono::system_clock::now();
        // 1st step, prepare
        prepare(frame_meta_with_batch, mats_to_infer);
        auto prepare_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time);

        // nothing to infer
        if (mats_to_infer.size() == 0) {
            return;
        }

        start_time = std::chrono::system_clock::now();
        // 2nd step, preprocess
        preprocess(mats_to_infer, blob_to_infer);
        auto preprocess_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time);

        start_time = std::chrono::system_clock::now();
        // 3rd step, infer
        infer(blob_to_infer, raw_outputs);
        auto infer_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time);

        start_time = std::chrono::system_clock::now();
        // 4th step, postprocess
        postprocess(raw_outputs, frame_meta_with_batch);   
        auto postprocess_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time);

        // end
        infer_combinations_time_cost(mats_to_infer.size(), prepare_time.count(), preprocess_time.count(), infer_time.count(), postprocess_time.count());
    }

    // print all by default
    void vp_infer_node::infer_combinations_time_cost(int data_size, int prepare_time, int preprocess_time, int infer_time, int postprocess_time) {
        /*
        std::cout << "########## infer combinations summary ##########" << std::endl;
        std::cout << " node_name:" << node_name << std::endl;
        std::cout << " data_size:" << data_size << std::endl;
        std::cout << " prepare_time:" << prepare_time << "ms" << std::endl;
        std::cout << " preprocess_time:" << preprocess_time << "ms" << std::endl;
        std::cout << " infer_time:" << infer_time << "ms" << std::endl;
        std::cout << " postprocess_time:" << postprocess_time << "ms" << std::endl;
        std::cout << "########## infer combinations summary ##########" << std::endl;
        */

        std::ostringstream s_stream;
        s_stream << "\n########## infer combinations summary ##########\n";
        s_stream << " node_name:" << node_name << "\n";
        s_stream << " data_size:" << data_size << "\n";
        s_stream << " prepare_time:" << prepare_time << "ms\n";
        s_stream << " preprocess_time:" << preprocess_time << "ms\n";
        s_stream << " infer_time:" << infer_time << "ms\n";
        s_stream << " postprocess_time:" << postprocess_time << "ms\n";
        s_stream << "########## infer combinations summary ##########\n";     

        // to log
        VP_DEBUG(s_stream.str());
    }

    void vp_infer_node::load_labels() {
        try {
            std::ifstream label_stream(labels_path);
            for (std::string line; std::getline(label_stream, line); ) {
                if (!line.empty() && line[line.length() - 1] == '\r') {
                    line.erase(line.length() - 1);
                }
                labels.push_back(line);
            }
        }
        catch(const std::exception& e) {
            
        }  
    }
}