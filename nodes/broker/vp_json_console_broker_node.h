#pragma once

#include <sstream>

#include "vp_msg_broker_node.h"
#include "cereal_archive/vp_objects_cereal_archive.h"

namespace vp_nodes {
    // message broker node (for debug purpose), broke json data to console.
    class vp_json_console_broker_node: public vp_msg_broker_node
    {
    private:
        /* data */
    protected:
        // to json
        virtual void format_msg(const std::shared_ptr<vp_objects::vp_frame_meta>& meta, std::string& msg) override;
        // to console
        virtual void broke_msg(const std::string& msg) override;
    public:
        vp_json_console_broker_node(std::string node_name, 
                                    vp_broke_for broke_for = vp_broke_for::NORMAL, 
                                    int broking_cache_warn_threshold = 50, 
                                    int broking_cache_ignore_threshold = 200);
        ~vp_json_console_broker_node();
    };
}