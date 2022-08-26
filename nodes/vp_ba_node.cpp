

#include "vp_ba_node.h"

namespace vp_nodes {
        
    vp_ba_node::vp_ba_node(std::string node_name, 
                            std::vector<std::shared_ptr<vp_objects::vp_frame_element>> elements_in_scene):
                            vp_node(node_name),
                            elements_in_scene(elements_in_scene) {
        // register built-in analysers for every vp_ba_node.
        this->register_default_ba_analysers();
        // initialize default element if need. 
        this->initialize_default_elements_if_need();
        this->initialized();
    }
    
    vp_ba_node::~vp_ba_node() {

    }

    std::shared_ptr<vp_objects::vp_meta> vp_ba_node::handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) {
        return meta;
    }

    std::shared_ptr<vp_objects::vp_meta> vp_ba_node::handle_control_meta(std::shared_ptr<vp_objects::vp_control_meta> meta) {
        return meta;
    }

    void vp_ba_node::register_ba_analysers(std::vector<std::shared_ptr<vp_ba::vp_ba_analyser>> ba_analysers) {
        for(auto & i: ba_analysers) {
            this->ba_analysers.push_back(i);
        }
    }

    void vp_ba_node::register_default_ba_analysers() {
        // register built-in analysers here
        auto cl_analyser = std::make_shared<vp_ba::vp_ba_crossline_analyser>();
        auto enter_analyser = std::make_shared<vp_ba::vp_ba_enter_analyser>();
        // ...
        this->register_ba_analysers({cl_analyser, enter_analyser});
    }

    void vp_ba_node::initialize_default_elements_if_need() {
        if (this->elements_in_scene.empty())
        {
            // add a default vp_region element
            std::vector<vp_objects::vp_point> vertexs = {{}, {}, {}};
            auto region = std::make_shared<vp_objects::vp_region>(0, vertexs, "", vp_ba::vp_ba_flag::STOP);
            this->elements_in_scene.push_back(region);
        }
    }
}