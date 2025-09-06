// SWIG interface file for VideoPipe - C# version
%module videopipe

%{
#include "../nodes/vp_node.h"
#include "../nodes/vp_src_node.h"
#include "../nodes/vp_des_node.h"
#include "../nodes/vp_infer_node.h"
#include "../nodes/vp_primary_infer_node.h"
#include "../nodes/vp_secondary_infer_node.h"
#include "../nodes/vp_file_src_node.h"
#include "../nodes/vp_screen_des_node.h"
#include "../nodes/infers/vp_yunet_face_detector_node.h"
#include "../nodes/infers/vp_sface_feature_encoder_node.h"
#include "../nodes/osd/vp_face_osd_node_v2.h"
#include "../nodes/vp_meta_hookable.h"
#include "../nodes/vp_meta_subscriber.h"
#include "../nodes/vp_meta_publisher.h"
#include "../nodes/vp_stream_info_hookable.h"
#include "../nodes/vp_stream_status_hookable.h"
#include "../objects/vp_meta.h"
#include "../objects/vp_frame_meta.h"
#include "../objects/vp_control_meta.h"
#include "../objects/vp_frame_target.h"
#include "../objects/vp_frame_face_target.h"
#include "../objects/shapes/vp_size.h"
#include "../objects/shapes/vp_rect.h"
#include "../objects/shapes/vp_point.h"
#include "../objects/vp_frame_pose_target.h"
#include "../objects/vp_frame_text_target.h"
#include "../objects/vp_sub_target.h"
#include "../objects/ba/vp_ba_result.h"








using namespace vp_nodes;
using namespace vp_objects;
%}

// C# specific directives
%include <std_string.i>
%include <std_vector.i>
%include <std_shared_ptr.i>
%include <std_pair.i>

%template(IntPair) std::pair<int, int>;
%template(IntPairVector) std::vector<std::pair<int, int>>;

%template(IntVector) std::vector<int>;
%template(FloatVector) std::vector<float>;
%template(StringVector) std::vector<std::string>;
%template(RectVector) std::vector<vp_objects::vp_rect>;
%template(PointVector) std::vector<vp_objects::vp_point>;


// Enable directors for virtual methods that can be overridden in C#
%feature("director") vp_nodes::vp_node;
%feature("director") vp_nodes::vp_src_node;
%feature("director") vp_nodes::vp_des_node;
%feature("director") vp_nodes::vp_infer_node;
%feature("director") vp_nodes::vp_primary_infer_node;
%feature("director") vp_nodes::vp_secondary_infer_node;
%feature("director") vp_meta_hookable;

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
%shared_ptr(vp_nodes::vp_meta_hookable);
%shared_ptr(vp_nodes::vp_meta_subscriber);
%shared_ptr(vp_nodes::vp_meta_publisher);
%shared_ptr(vp_nodes::vp_stream_info_hookable);
%shared_ptr(vp_nodes::vp_stream_status_hookable);
%shared_ptr(vp_objects::vp_ba_result);


%shared_ptr(vp_objects::vp_frame_target);
%shared_ptr(vp_objects::vp_frame_face_target);
%shared_ptr(vp_objects::vp_frame_pose_target);
%shared_ptr(vp_objects::vp_frame_text_target);
%shared_ptr(vp_objects::vp_sub_target);
%shared_ptr(vp_objects::vp_size);
%shared_ptr(vp_objects::vp_rect);
%shared_ptr(vp_objects::vp_point);





%template(vp_node_vector) std::vector<std::shared_ptr<vp_nodes::vp_node>>;
%template(vp_frame_target_vector) std::vector<std::shared_ptr<vp_objects::vp_frame_target>>;
%template(vp_frame_face_target_vector) std::vector<std::shared_ptr<vp_objects::vp_frame_face_target>>;
%template(vp_frame_pose_target_vector) std::vector<std::shared_ptr<vp_objects::vp_frame_pose_target>>;
%template(vp_frame_text_target_vector) std::vector<std::shared_ptr<vp_objects::vp_frame_text_target>>;
%template(vp_sub_target_vector) std::vector<std::shared_ptr<vp_objects::vp_sub_target>>;


// 添加回调函数的typemap
%typemap(ctype) vp_meta_hooker "void*"
%typemap(imtype) vp_meta_hooker "System.IntPtr"
%typemap(cstype) vp_meta_hooker "vp_meta_hookable.MetaHookerDelegate"

// 处理C#到C++的回调转换
%typemap(csin) vp_meta_hooker %{
    // 创建委托的回调指针
    System.Runtime.InteropServices.GCHandle.Alloc($csinput);
    vp_meta_hookable.MetaHookerDelegate.CreateDelegate($csinput).ToIntPtr()
%}

// 处理C++到C#的回调转换
%typemap(csout) vp_meta_hooker %{
    return vp_meta_hookable.MetaHookerDelegate.FromIntPtr($imcall);
%}


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

%template(vp_pose_keypoint_vector) std::vector<vp_objects::vp_pose_keypoint>;

%include "../objects/shapes/vp_point.h"
%include "../objects/shapes/vp_size.h"
%include "../objects/shapes/vp_rect.h"
%include "../objects/ba/vp_ba_result.h"

%include "../objects/vp_frame_target.h"
%include "../nodes/vp_meta_hookable.h"
%include "../nodes/vp_meta_subscriber.h"
%include "../nodes/vp_meta_publisher.h"
%include "../nodes/vp_stream_info_hookable.h"
%include "../nodes/vp_stream_status_hookable.h"
%include "../nodes/vp_node.h"
%include "../nodes/vp_src_node.h"
%include "../nodes/vp_des_node.h"
%include "../nodes/vp_infer_node.h"
%include "../nodes/vp_primary_infer_node.h"
%include "../nodes/vp_secondary_infer_node.h"
%include "../nodes/vp_file_src_node.h"
%include "../nodes/vp_screen_des_node.h"
%include "../nodes/infers/vp_yunet_face_detector_node.h"
%include "../nodes/infers/vp_sface_feature_encoder_node.h"
%include "../nodes/osd/vp_face_osd_node_v2.h"
%include "../objects/vp_meta.h"
%include "../objects/vp_frame_meta.h"
%include "../objects/vp_control_meta.h"
%include "../objects/vp_frame_face_target.h"
%include "../objects/vp_frame_pose_target.h"
%include "../objects/vp_frame_text_target.h"
%include "../objects/vp_sub_target.h"







