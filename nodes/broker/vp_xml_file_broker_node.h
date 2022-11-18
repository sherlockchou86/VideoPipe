#pragma once

#include <fstream>

#include "vp_msg_broker_node.h"
#include "cereal_archive/vp_objects_cereal_archive.h"

namespace vp_nodes {
    // message broker node (for demo/debug purpose), broke xml data to file.
    class vp_xml_file_broker_node: public vp_msg_broker_node
    {
    private:
        // xml file writer
        ofstream xml_writer;
    protected:
        // to xml
        virtual void format_msg(const std::shared_ptr<vp_objects::vp_frame_meta>& meta, std::string& msg) override;
        // to file
        virtual void broke_msg(const std::string& msg) override;
    public:
        vp_xml_file_broker_node(std::string node_name, 
                                vp_broke_for broke_for = vp_broke_for::NORMAL, 
                                std::string file_path_and_name = "",
                                int broking_cache_warn_threshold = 50, 
                                int broking_cache_ignore_threshold = 200);
        ~vp_xml_file_broker_node();
    };
}