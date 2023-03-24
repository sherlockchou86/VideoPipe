

#pragma once
#include <sstream>
#include <opencv2/dnn.hpp>
#include "vp_node.h"

namespace vp_nodes {

    // infer type
    // infer on the whole frame or small cropped image?
    enum vp_infer_type {
        PRIMARY,      // infer on the whole frame, like detector, pose estimatation
        SECONDARY     // infer on small cropped image, like classifier, feature extractor and secondary detector which need detect on small cropped images.
    };

    // base class for infer node, can't be instanstiated directly. 
    // note: 
    // the class is based on opencv::dnn module which is the default way for all deep learning inference in code, 
    // we can implement it using other backends such as tensorrt with cuda acceleration, see vp_ppocr_text_detector_node which is based on PaddlePaddle dl framework from BaiDu corporation.
    class vp_infer_node: public vp_node {
    private:
        // load labels if need
        void load_labels();
    protected:
        vp_infer_type infer_type;
        std::string model_path;
        std::string model_config_path;
        std::string labels_path;
        int input_width;
        int input_height;
        int batch_size;
        cv::Scalar mean;
        cv::Scalar std;
        float scale;
        bool swap_rb;

        // transpose channel or not, NCHW -> NHWC
        bool swap_chn;

        // protected as it can't be instanstiated directly.
        vp_infer_node(std::string node_name, 
                    vp_infer_type infer_type, 
                    std::string model_path, 
                    std::string model_config_path = "", 
                    std::string labels_path = "", 
                    int input_width = 128, 
                    int input_height = 128, 
                    int batch_size = 1,
                    float scale = 1.0,
                    cv::Scalar mean = cv::Scalar(123.675, 116.28, 103.53),  // imagenet dataset
                    cv::Scalar std = cv::Scalar(1),
                    bool swap_rb = true,
                    bool swap_chn = false);
        
        // the 1st step, MUST implement in specific derived class.
        // prepare data for infer, fetch frames from frame meta.
        virtual void prepare(const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch, std::vector<cv::Mat>& mats_to_infer) = 0;
        
        // the 2nd step, has a default implementation.
        // preprocess data, such as normalization, mean substract.
        virtual void preprocess(const std::vector<cv::Mat>& mats_to_infer, cv::Mat& blob_to_infer);

        // the 3rd step, has a default implementation.
        // infer and retrive raw outputs.
        virtual void infer(const cv::Mat& blob_to_infer, std::vector<cv::Mat>& raw_outputs);

        // the 4th step, MUST implement in specific derived class.
        // postprocess on raw outputs and create/update something back to frame meta again.
        virtual void postprocess(const std::vector<cv::Mat>& raw_outputs, const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) = 0;

        // debug purpose(ms)
        virtual void infer_combinations_time_cost(int data_size, int prepare_time, int preprocess_time, int infer_time, int postprocess_time);

        // infer operations(call prepare/preprocess/infer/postprocess by default)
        // we can define new logic for infer operations by overriding it.
        virtual void run_infer_combinations(const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch);

        // labels as text format
        std::vector<std::string> labels;
        
        // opencv::dnn as backend 
        cv::dnn::Net net;

        // re-implementation for one by one mode, marked as 'final' as we need not override any more in specific derived classes.
        virtual std::shared_ptr<vp_objects::vp_meta> handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) override final; 
        // re-implementation for batch by batch mode, marked as 'final' as we need not override any more in specific derived classes.
        virtual void handle_frame_meta(const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& meta_with_batch) override final; 
    public:
        ~vp_infer_node();
    };
}