// SWIG interface file for VideoPipe - C# version
%module videopipe

%{
#include "nodes/vp_node.h"
#include "nodes/vp_src_node.h"
#include "nodes/vp_des_node.h"
#include "nodes/vp_infer_node.h"
#include "nodes/vp_primary_infer_node.h"
#include "nodes/vp_secondary_infer_node.h"
#include "nodes/vp_file_src_node.h"
#include "nodes/vp_screen_des_node.h"
#include "nodes/infers/vp_yunet_face_detector_node.h"
#include "nodes/infers/vp_sface_feature_encoder_node.h"
#include "nodes/osd/vp_face_osd_node_v2.h"
using namespace vp_nodes;
using namespace vp_objects;
%}

// C# specific directives
%include "std_string.i"
%include "std_vector.i"
%include "std_shared_ptr.i"

// Enable directors for virtual methods that can be overridden in C#
%feature("director") vp_nodes::vp_node;
%feature("director") vp_nodes::vp_src_node;
%feature("director") vp_nodes::vp_des_node;
%feature("director") vp_nodes::vp_infer_node;
%feature("director") vp_nodes::vp_primary_infer_node;
%feature("director") vp_nodes::vp_secondary_infer_node;

// Handle shared_ptr for common types
%shared_ptr(vp_nodes::vp_node);
%shared_ptr(vp_nodes::vp_src_node);
%shared_ptr(vp_nodes::vp_des_node);
%shared_ptr(vp_nodes::vp_infer_node);
%shared_ptr(vp_nodes::vp_primary_infer_node);
%shared_ptr(vp_nodes::vp_secondary_infer_node);
%shared_ptr(vp_nodes::vp_file_src_node);
%shared_ptr(vp_nodes::vp_screen_des_node);
%shared_ptr(vp_nodes::vp_yunet_face_detector_node);
%shared_ptr(vp_nodes::vp_sface_feature_encoder_node);
%shared_ptr(vp_nodes::vp_face_osd_node_v2);
%shared_ptr(vp_objects::vp_meta);
%shared_ptr(vp_objects::vp_frame_meta);
%shared_ptr(vp_objects::vp_control_meta);

// Handle std::vector<std::shared_ptr<T>> templates
%template(NodeVector) std::vector<std::shared_ptr<vp_nodes::vp_node> >;
%template(FrameMetaVector) std::vector<std::shared_ptr<vp_objects::vp_frame_meta> >;

// Enumerations
enum vp_node_type { SRC, DES, MID };
enum vp_infer_type { PRIMARY, SECONDARY };

// Forward declarations for dependencies
namespace vp_objects {
    class vp_meta;
    class vp_frame_meta;
    class vp_control_meta;
    class vp_frame_target;
    class vp_frame_face_target;
    struct vp_size;
}

// Node creation functions
%inline %{
    std::shared_ptr<vp_nodes::vp_file_src_node> create_file_src_node(const std::string& node_name, 
                                                           int channel_index, 
                                                           const std::string& file_path, 
                                                           float resize_ratio = 1.0, 
                                                           bool cycle = true,
                                                           const std::string& gst_decoder_name = "avdec_h264",
                                                           int skip_interval = 0) {
        return std::make_shared<vp_nodes::vp_file_src_node>(node_name, channel_index, file_path, resize_ratio, cycle, gst_decoder_name, skip_interval);
    }
    
    std::shared_ptr<vp_nodes::vp_screen_des_node> create_screen_des_node(const std::string& node_name, 
                                                               int channel_index, 
                                                               bool osd = true,
                                                               const vp_objects::vp_size& display_w_h = vp_objects::vp_size()) {
        return std::make_shared<vp_nodes::vp_screen_des_node>(node_name, channel_index, osd, display_w_h);
    }
    
    std::shared_ptr<vp_nodes::vp_yunet_face_detector_node> create_yunet_face_detector_node(const std::string& node_name, 
                                                                                const std::string& model_path, 
                                                                                float score_threshold = 0.7, 
                                                                                float nms_threshold = 0.5, 
                                                                                int top_k = 50) {
        return std::make_shared<vp_nodes::vp_yunet_face_detector_node>(node_name, model_path, score_threshold, nms_threshold, top_k);
    }
    
    std::shared_ptr<vp_nodes::vp_sface_feature_encoder_node> create_sface_feature_encoder_node(const std::string& node_name, 
                                                                                    const std::string& model_path) {
        return std::make_shared<vp_nodes::vp_sface_feature_encoder_node>(node_name, model_path);
    }
    
    std::shared_ptr<vp_nodes::vp_face_osd_node_v2> create_face_osd_node_v2(const std::string& node_name) {
        return std::make_shared<vp_nodes::vp_face_osd_node_v2>(node_name);
    }
%}

namespace vp_nodes {
    // Base classes
    // vp_node - base class for all nodes
    class vp_node {
    private:
        // Hide private members
        std::vector<std::shared_ptr<vp_node> > pre_nodes;
        std::thread handle_thread;
        std::thread dispatch_thread;
        
    protected:
        bool alive;
        int frame_meta_handle_batch;
        std::queue<std::shared_ptr<vp_objects::vp_meta> > in_queue;
        std::mutex in_queue_lock;
        std::queue<std::shared_ptr<vp_objects::vp_meta> > out_queue;
        
        // Virtual methods that can be overridden
        virtual void handle_run();
        virtual void dispatch_run();
        virtual std::shared_ptr<vp_objects::vp_meta> handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta);
        virtual std::shared_ptr<vp_objects::vp_meta> handle_control_meta(std::shared_ptr<vp_objects::vp_control_meta> meta);
        virtual void handle_frame_meta(const std::vector<std::shared_ptr<vp_objects::vp_frame_meta> >& meta_with_batch);
        virtual void deinitialized();
        void pendding_meta(std::shared_ptr<vp_objects::vp_meta> meta);
        
    public:
        // Constructor and destructor
        vp_node(std::string node_name);
        virtual ~vp_node();
        
        // Public members
        std::string node_name;
        
        // Public methods
        virtual void meta_flow(std::shared_ptr<vp_objects::vp_meta> meta);
        virtual vp_node_type node_type();
        void detach();
        void detach_from(std::vector<std::string> pre_node_names);
        void detach_recursively();
        void attach_to(std::vector<std::shared_ptr<vp_node> > pre_nodes);
        std::vector<std::shared_ptr<vp_node> > next_nodes();
        virtual std::string to_string();
        
        // Needed for enable_shared_from_this
        std::shared_ptr<vp_node> shared_from_this() {
            return shared_from_this();
        }
    };
    
    // vp_src_node - base class for source nodes
    class vp_src_node : public vp_node {
    private:
        // Hide private members
        
    protected:
        virtual void handle_run() override;
        virtual std::shared_ptr<vp_objects::vp_meta> handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) override; 
        virtual std::shared_ptr<vp_objects::vp_meta> handle_control_meta(std::shared_ptr<vp_objects::vp_control_meta> meta) override;
        
        vp_src_node(std::string node_name, int channel_index, float resize_ratio = 1.0);
        
        int original_fps;
        int original_width;
        int original_height;
        int frame_index;
        int channel_index;
        float resize_ratio;
        virtual void deinitialized() override;
        
    public:
        ~vp_src_node();
        virtual vp_node_type node_type() override;
        void start();
        void stop();
        virtual std::string to_string() override;
    };
    
    // vp_des_node - base class for destination nodes
    class vp_des_node : public vp_node {
    private:
        // Hide private members
        
    protected:
        virtual void dispatch_run() override;
        virtual std::shared_ptr<vp_objects::vp_meta> handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) override; 
        virtual std::shared_ptr<vp_objects::vp_meta> handle_control_meta(std::shared_ptr<vp_objects::vp_control_meta> meta) override;
        
        vp_des_node(std::string node_name, int channel_index);
        
    public:
        ~vp_des_node();
        virtual vp_node_type node_type() override;
        int channel_index;
    };
    
    // vp_infer_node - base class for inference nodes
    class vp_infer_node : public vp_node {
    private:
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
        bool swap_chn;
        
        vp_infer_node(std::string node_name, 
                     vp_infer_type infer_type, 
                     std::string model_path, 
                     std::string model_config_path = "", 
                     std::string labels_path = "", 
                     int input_width = 128, 
                     int input_height = 128, 
                     int batch_size = 1,
                     float scale = 1.0,
                     cv::Scalar mean = cv::Scalar(123.675, 116.28, 103.53),
                     cv::Scalar std = cv::Scalar(1),
                     bool swap_rb = true,
                     bool swap_chn = false);
        
        virtual void prepare(const std::vector<std::shared_ptr<vp_objects::vp_frame_meta> >& frame_meta_with_batch, std::vector<cv::Mat>& mats_to_infer) = 0;
        virtual void preprocess(const std::vector<cv::Mat>& mats_to_infer, cv::Mat& blob_to_infer);
        virtual void infer(const cv::Mat& blob_to_infer, std::vector<cv::Mat>& raw_outputs);
        virtual void postprocess(const std::vector<cv::Mat>& raw_outputs, const std::vector<std::shared_ptr<vp_objects::vp_frame_meta> >& frame_meta_with_batch) = 0;
        virtual void infer_combinations_time_cost(int data_size, int prepare_time, int preprocess_time, int infer_time, int postprocess_time);
        virtual void run_infer_combinations(const std::vector<std::shared_ptr<vp_objects::vp_frame_meta> >& frame_meta_with_batch);
        
        std::vector<std::string> labels;
        cv::dnn::Net net;
        
        virtual std::shared_ptr<vp_objects::vp_meta> handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) override;
        virtual void handle_frame_meta(const std::vector<std::shared_ptr<vp_objects::vp_frame_meta> >& meta_with_batch) override;
        
    public:
        ~vp_infer_node();
    };
    
    // vp_primary_infer_node - base class for primary inference nodes
    class vp_primary_infer_node : public vp_infer_node {
    protected:
        vp_primary_infer_node(std::string node_name, 
                             std::string model_path, 
                             std::string model_config_path = "", 
                             std::string labels_path = "", 
                             int input_width = 640, 
                             int input_height = 640, 
                             int batch_size = 1,
                             float scale = 1.0,
                             cv::Scalar mean = cv::Scalar(0, 0, 0),
                             cv::Scalar std = cv::Scalar(1),
                             bool swap_rb = true,
                             bool swap_chn = false);
        
        virtual void prepare(const std::vector<std::shared_ptr<vp_objects::vp_frame_meta> >& frame_meta_with_batch, std::vector<cv::Mat>& mats_to_infer) override;
        
    public:
        ~vp_primary_infer_node();
    };
    
    // vp_secondary_infer_node - base class for secondary inference nodes
    class vp_secondary_infer_node : public vp_infer_node {
    protected:
        vp_secondary_infer_node(std::string node_name, 
                               std::string model_path, 
                               std::string model_config_path = "", 
                               std::string labels_path = "", 
                               int input_width = 128, 
                               int input_height = 128, 
                               int batch_size = 1,
                               float scale = 1.0,
                               cv::Scalar mean = cv::Scalar(0, 0, 0),
                               cv::Scalar std = cv::Scalar(1),
                               bool swap_rb = true,
                               bool swap_chn = false);
                               
        virtual void prepare(const std::vector<std::shared_ptr<vp_objects::vp_frame_meta> >& frame_meta_with_batch, std::vector<cv::Mat>& mats_to_infer) override;
        
    public:
        ~vp_secondary_infer_node();
    };
    
    // Concrete node classes
    // vp_file_src_node - file source node
    class vp_file_src_node : public vp_src_node {
    private:
        std::string gst_template;
        cv::VideoCapture file_capture;
        
    protected:
        virtual void handle_run() override;
        
    public:
        vp_file_src_node(std::string node_name, 
                        int channel_index, 
                        std::string file_path, 
                        float resize_ratio = 1.0, 
                        bool cycle = true,
                        std::string gst_decoder_name = "avdec_h264",
                        int skip_interval = 0);
        ~vp_file_src_node();
        
        virtual std::string to_string() override;
        std::string file_path;
        bool cycle;
        std::string gst_decoder_name;
        int skip_interval;
    };
    
    // vp_screen_des_node - screen display node
    class vp_screen_des_node : public vp_des_node {
    private:
        std::string gst_template;
        cv::VideoWriter screen_writer;
        
    protected:
        virtual std::shared_ptr<vp_objects::vp_meta> handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) override;
        virtual std::shared_ptr<vp_objects::vp_meta> handle_control_meta(std::shared_ptr<vp_objects::vp_control_meta> meta) override;
        
    public:
        vp_screen_des_node(std::string node_name, 
                          int channel_index, 
                          bool osd = true,
                          vp_objects::vp_size display_w_h = vp_objects::vp_size());
        ~vp_screen_des_node();
        
        bool osd;
        vp_objects::vp_size display_w_h;
    };
    
    // vp_yunet_face_detector_node - face detection node
    class vp_yunet_face_detector_node : public vp_primary_infer_node {
    private:
        const std::vector<std::string> out_names;
        float scoreThreshold;
        float nmsThreshold;
        int topK;
        int inputW;
        int inputH;
        std::vector<cv::Rect2f> priors;
        void generatePriors();
        
    protected:
        virtual void infer(const cv::Mat& blob_to_infer, std::vector<cv::Mat>& raw_outputs) override;
        virtual void preprocess(const std::vector<cv::Mat>& mats_to_infer, cv::Mat& blob_to_infer) override;
        virtual void postprocess(const std::vector<cv::Mat>& raw_outputs, const std::vector<std::shared_ptr<vp_objects::vp_frame_meta> >& frame_meta_with_batch) override;
        
    public:
        vp_yunet_face_detector_node(std::string node_name, std::string model_path, float score_threshold = 0.7, float nms_threshold = 0.5, int top_k = 50);
        ~vp_yunet_face_detector_node();
    };
    
    // vp_sface_feature_encoder_node - face feature encoding node
    class vp_sface_feature_encoder_node : public vp_secondary_infer_node {
    private:
        cv::Mat getSimilarityTransformMatrix(float src[5][2]);
        void alignCrop(cv::Mat& _src_img, float _src_point[5][2], cv::Mat& _aligned_img);
        
    protected:
        virtual void prepare(const std::vector<std::shared_ptr<vp_objects::vp_frame_meta> >& frame_meta_with_batch, std::vector<cv::Mat>& mats_to_infer) override;
        virtual void postprocess(const std::vector<cv::Mat>& raw_outputs, const std::vector<std::shared_ptr<vp_objects::vp_frame_meta> >& frame_meta_with_batch) override;
        
    public:
        vp_sface_feature_encoder_node(std::string node_name, std::string model_path);
        ~vp_sface_feature_encoder_node();
    };
    
    // vp_face_osd_node_v2 - face on-screen display node
    class vp_face_osd_node_v2 : public vp_node {
    private:
        int gap_height;
        int padding;
        cv::Mat the_baseline_face;
        std::vector<float> the_baseline_face_feature;
        std::vector<cv::Mat> faces_list;
        std::vector<std::vector<float> > face_features;
        std::vector<float> cosine_distances;
        std::vector<float> l2_distances;
        double cosine_similar_thresh;
        double l2norm_similar_thresh;
        double match(std::vector<float>& feature1, std::vector<float>& feature2, int dis_type);
        
    protected:
        virtual std::shared_ptr<vp_objects::vp_meta> handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) override;
        
    public:
        vp_face_osd_node_v2(std::string node_name);
        ~vp_face_osd_node_v2();
    };
}