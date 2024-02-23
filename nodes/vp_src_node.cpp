
#include <memory>
#include "vp_src_node.h"
#include "../objects/vp_control_meta.h"
#include "../objects/vp_image_record_control_meta.h"
#include "../objects/vp_video_record_control_meta.h"

namespace vp_nodes {
    
    vp_src_node::vp_src_node(std::string node_name, 
                            int channel_index, 
                            float resize_ratio): 
                            vp_node(node_name), 
                            channel_index(channel_index), 
                            resize_ratio(resize_ratio), 
                            frame_index(-1) {
        assert(resize_ratio > 0 && resize_ratio <= 1.0f);
    }
    
    vp_src_node::~vp_src_node() {

    }
    
    void vp_src_node::deinitialized() {
        alive = false;
        gate.open();
        vp_node::deinitialized();
    }

    void vp_src_node::handle_run() {
        throw vp_excepts::vp_not_implemented_error("must have re-implementaion for 'handle_run' method in src nodes!");
    }

    std::shared_ptr<vp_objects::vp_meta> 
            vp_src_node::handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) {
        throw vp_excepts::vp_invalid_calling_error("'handle_frame_meta' method could not be called in src nodes!");
    }

    std::shared_ptr<vp_objects::vp_meta> 
            vp_src_node::handle_control_meta(std::shared_ptr<vp_objects::vp_control_meta> meta) {
        throw vp_excepts::vp_invalid_calling_error("'handle_control_meta' method could not be called in src nodes!");
    }

    void vp_src_node::start() {
        gate.open();
    }

    void vp_src_node::stop() {
        gate.close();
    }

    void vp_src_node::speak() {
        auto speak_control_meta = std::make_shared<vp_objects::vp_control_meta>(vp_objects::vp_control_type::SPEAK, this->channel_index);
        this->push_meta(speak_control_meta);
    }

     vp_node_type vp_src_node::node_type() {
         return vp_node_type::SRC;
     }

    int vp_src_node::get_original_fps() const {
        return original_fps;
    }

    int vp_src_node::get_original_width() const {
        return original_width;
    }

    int vp_src_node::get_original_height() const {
        return original_height;
    }

    void vp_src_node::record_video_manually(bool osd, int video_duration) {
        // make sure file is not too long
        assert(video_duration <= 60 && video_duration >= 5);

        // generate file_name_without_ext
        // MUST be unique
        auto file_name_without_ext = vp_utils::time_format(NOW, "manually_record_video_<year><mon><day><hour><min><sec><mili>");

        // create control meta
        auto video_record_control_meta = std::make_shared<vp_objects::vp_video_record_control_meta>(channel_index, file_name_without_ext, video_duration, osd);

        // push meta to pipe
        push_meta(video_record_control_meta);
    }

    void vp_src_node::record_image_manually(bool osd) {
        // generate file_name_without_ext
        // MUST be unique
        auto file_name_without_ext = vp_utils::time_format(NOW, "manually_record_image_<year><mon><day><hour><min><sec><mili>");

        // create control meta
        auto image_record_control_meta = std::make_shared<vp_objects::vp_image_record_control_meta>(channel_index, file_name_without_ext, osd);

        // push meta to pipe
        push_meta(image_record_control_meta);
    }
} 
