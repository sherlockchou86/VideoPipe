#pragma once
#include "vp_track_node.h"

namespace vp_nodes {
    // track node using deep sort
    class vp_dsort_track_node: public vp_track_node
    {
    private:
        /* config data for deep sort algo*/
    protected:
        // fill track_ids using deep sort algo
        virtual void track(int channel_index, const std::vector<vp_objects::vp_rect>& target_rects, 
                        const std::vector<std::vector<float>>& target_embeddings, 
                        std::vector<int>& track_ids) override;
    public:
        vp_dsort_track_node(std::string node_name, vp_track_for track_for = vp_track_for::NORMAL);
        virtual ~vp_dsort_track_node();
    };
}
