
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
                            bool swap_rb):
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
                            swap_rb(swap_rb) {
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

        net.setInput(blob_to_infer);
        
        auto number_of_batch = blob_to_infer.size[0];
        if (number_of_batch <= batch_size) {
            // infer one time directly
            net.forward(raw_outputs, net.getUnconnectedOutLayersNames());
        }
        else {
            // infer more times
            // TO-DO
            net.forward(raw_outputs, net.getUnconnectedOutLayersNames());
        }
    }

    // default implementation
    // create a 4D matrix(n, c, h, w)
    void vp_infer_node::preprocess(const std::vector<cv::Mat>& mats_to_infer, cv::Mat& blob_to_infer) {
        cv::dnn::blobFromImages(mats_to_infer, blob_to_infer, scale, cv::Size(input_width, input_height), mean, swap_rb);
        if (std != cv::Scalar(1)) {
            // divide by std
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
        std::cout << "########## infer combinations summary ##########" << std::endl;
        std::cout << " node_name:" << node_name << std::endl;
        std::cout << " data_size:" << data_size << std::endl;
        std::cout << " prepare_time:" << prepare_time << "ms" << std::endl;
        std::cout << " preprocess_time:" << preprocess_time << "ms" << std::endl;
        std::cout << " infer_time:" << infer_time << "ms" << std::endl;
        std::cout << " postprocess_time:" << postprocess_time << "ms" << std::endl;
        std::cout << "########## infer combinations summary ##########" << std::endl;
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