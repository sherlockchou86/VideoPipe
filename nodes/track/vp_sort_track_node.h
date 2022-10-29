

#pragma once
#include "vp_track_node.h"

namespace vp_nodes {
    // track node using sort
    class vp_sort_track_node: public vp_track_node
    {
    private:
        /* config data for sort algo */
    protected:
        // fill track_ids using sort algo
        virtual void track(const std::vector<vp_objects::vp_rect>& target_rects, 
                        const std::vector<std::vector<float>>& target_embeddings, 
                        std::vector<int>& track_ids) override;
    public:
        vp_sort_track_node(std::string node_name, vp_track_for track_for = vp_track_for::NORMAL);
        virtual ~vp_sort_track_node();
    };

}