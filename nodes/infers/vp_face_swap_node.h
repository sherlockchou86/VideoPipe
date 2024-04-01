#pragma once

#include "../vp_primary_infer_node.h"

namespace vp_nodes {
    // face swap node
    // used to swap faces in videos/images using a specific face
    class vp_face_swap_node: public vp_primary_infer_node
    {   
    private:
        // inner temporary use
        struct face_box {
            int x;
            int y;
            int width;
            int height;
            float score;
            std::vector<std::pair<int, int>> kps;
        };

        /* onnx network using opencv::dnn as backend */
        cv::dnn::Net face_extract_net;
        cv::dnn::Net face_encoding_net;
        cv::dnn::Net face_swap_net;

        cv::Mat read_emap_mat_from_txt(const std::string& emap_file_for_embeddings, int rows = 512, int cols = 512);
        cv::Mat process_embeddings_using_emap(const cv::Mat& source_face_normed_embedding, const cv::Mat& emap);

        cv::Mat get_similarity_transform_matrix(float src[5][2]);
        cv::Mat align_crop(cv::Mat& _src_img, std::vector<std::pair<int, int>>& kps, cv::Mat& _aligned_img);
        
        void extract_faces(const cv::Mat& frame, std::vector<face_box>& faces);
        void init_source_face_embeddings(std::string& swap_source_image, int swap_source_face_index, std::string& emap_file_for_embeddings);
        void swap(cv::Mat& aligned_face, cv::Mat& swapped_face);
        void paste_back(cv::Mat& bg, cv::Mat& swapped_face, const cv::Mat& transform_matrix);

        cv::Mat source_face_embeddings;
        bool act_as_primary_detector = false;
        bool swap_on_osd = true;

        /* used for extract faces */
        std::vector<cv::Rect2f> priors;
        void generatePriors(int inputW, int inputH);
    protected:
        // we need a totally new logic for the whole infer combinations
        // no separate step pre-defined needed in base class
        virtual void run_infer_combinations(const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) override;
        // override pure virtual method, for compile pass
        virtual void postprocess(const std::vector<cv::Mat>& raw_outputs, const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) override;
    public:
        vp_face_swap_node(std::string node_name, 
                            std::string yunet_face_detect_model,
                            std::string buffalo_l_face_encoding_model,
                            std::string emap_file_for_embeddings,
                            std::string insightface_swap_model,
                            std::string swap_source_image,
                            int swap_source_face_index = 0,
                            bool swap_on_osd = true,
                            bool act_as_primary_detector = false);
        ~vp_face_swap_node();
    };
}