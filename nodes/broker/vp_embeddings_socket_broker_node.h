#pragma once

#include "vp_msg_broker_node.h"
#include "cereal_archive/vp_objects_cereal_archive.h"

// light weight socket support
#include "../../third_party/kissnet/kissnet.hpp"

namespace vp_nodes {
    // message broker node, broke ONLY embeddings(vp_frame_target or vp_frame_face_target) data to socket via udp.
    class vp_embeddings_socket_broker_node: public vp_msg_broker_node
    {
    private:
        // save dir for cropped images, which would be used for embeddings similiarity search later
        std::string cropped_dir = "cropped_images";
        // min width to crop (embedding will be ignored if target's width is smaller than this value)
        int min_crop_width = 50;
        // min height to crop (embedding will be ignored if target's height is smaller than this value)
        int min_crop_height = 50;
        // host the data sent to via udp
        std::string des_ip = "";
        // port the data sent to via udp
        int des_port = 0;

        // udp socket writer
        kissnet::udp_socket udp_writer;
    protected:
        // to xml
        virtual void format_msg(const std::shared_ptr<vp_objects::vp_frame_meta>& meta, std::string& msg) override;
        // to socket via udp
        virtual void broke_msg(const std::string& msg) override;
    public:
        vp_embeddings_socket_broker_node(std::string node_name, 
                                std::string des_ip = "",
                                int des_port = 0,
                                std::string cropped_dir = "cropped_images",
                                int min_crop_width = 50,
                                int min_crop_height = 50,
                                vp_broke_for broke_for = vp_broke_for::NORMAL, 
                                int broking_cache_warn_threshold = 50, 
                                int broking_cache_ignore_threshold = 200);
        ~vp_embeddings_socket_broker_node();
    };
}