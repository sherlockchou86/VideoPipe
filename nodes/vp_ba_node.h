
#pragma once

#include <string>
#include <memory>
#include <vector>
#include <assert.h>

#include "vp_node.h"
#include "../objects/elements/vp_frame_element.h"
#include "../objects/elements/vp_region.h"
#include "../ba/vp_ba_analyser.h"
#include "../ba/vp_ba_crossline_analyser.h"
#include "../ba/vp_ba_enter_analyser.h"

namespace vp_nodes {
    // behaviour analysis node, short as ba
    // 
    class vp_ba_node: public vp_node {
    private:
        // elements in video scene
        // since ba analysers work based on elements, if there are no elements in current video scene, 
        // it will add a default element(vp_region) with setting vp_region::ba_abilities_mask = vp_ba::vp_ba_flag::STOP.
        std::vector<std::shared_ptr<vp_objects::vp_frame_element>> elements_in_scene;
        
        // ba analysers
        // the later added analysers will overwrite the previous ones if they have the same ba_ability.
        std::vector<std::shared_ptr<vp_ba::vp_ba_analyser>> ba_analysers;

        // register built-in ba analysers for vp_ba_node.
        void register_default_ba_analysers();

        // add a default element if no elements given when creating the vp_ba_node instance.
        void initialize_default_elements_if_need();

    protected:
        virtual std::shared_ptr<vp_objects::vp_meta> handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) override;
        virtual std::shared_ptr<vp_objects::vp_meta> handle_control_meta(std::shared_ptr<vp_objects::vp_control_meta> meta) override;
    public:
        vp_ba_node(std::string node_name, std::vector<std::shared_ptr<vp_objects::vp_frame_element>> elements_in_scene);
        ~vp_ba_node();

        // register ba analysers for the node
        void register_ba_analysers(std::vector<std::shared_ptr<vp_ba::vp_ba_analyser>> ba_analysers);
    };
}