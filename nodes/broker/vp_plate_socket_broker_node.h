#pragma once

#include "vp_msg_broker_node.h"
#include "cereal_archive/vp_objects_cereal_archive.h"

// light weight socket support
#include "../../third_party/kissnet/kissnet.hpp"

namespace vp_nodes {
    // message broker node, broke text & color for license plate (hold by vp_frame_target) to socket via udp.
    // which could be used for lpr camera.
    class vp_plate_socket_broker_node: public vp_msg_broker_node
    {
    private:
        // save dir for cropped images and whole images, which would be used for displaying in lpr camera
        std::string plates_dir = "plate_images";
        // min width to crop (license plate will be ignored if target's width is smaller than this value)
        int min_crop_width = 50;
        // min height to crop (license plate will be ignored if target's height is smaller than this value)
        int min_crop_height = 50;
        // only broke for tracked targets (track_id is not -1)
        bool only_for_tracked = false;
        
        // min tracked frames if only_for_tracked is true
        int min_tracked_frames = 25;

        // host the data sent to via udp
        std::string des_ip = "";
        // port the data sent to via udp
        int des_port = 0;

        // udp socket writer
        kissnet::udp_socket udp_writer;

        // support multi-channel, used for avoid duplicate data
        std::map<int, std::vector<int>> all_broked_ids;           // channel -> target ids which have been broked
        std::map<int, std::vector<std::string>> all_broked_texts;  // channel -> plate texts which have been broked (filter using similiarity comparison)
    protected:
        // to custom format
        virtual void format_msg(const std::shared_ptr<vp_objects::vp_frame_meta>& meta, std::string& msg) override;
        // to socket via udp
        virtual void broke_msg(const std::string& msg) override;
    public:
        vp_plate_socket_broker_node(std::string node_name, 
                                std::string des_ip = "",
                                int des_port = 0,
                                std::string plates_dir = "plate_images",
                                int min_crop_width = 100,
                                int min_crop_height = 0,
                                vp_broke_for broke_for = vp_broke_for::NORMAL, 
                                bool only_for_tracked = true, 
                                int broking_cache_warn_threshold = 50, 
                                int broking_cache_ignore_threshold = 200);
        ~vp_plate_socket_broker_node();
    };
}