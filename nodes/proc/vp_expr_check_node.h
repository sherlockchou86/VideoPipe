#pragma once

#include "../vp_node.h"

namespace vp_nodes {
    // math expression checker, give right for `1+1=2` and wrong for `sqrt(4)=4`.
    // note: this node works based on vp_frame_text_target, it will parse expression at the left of `=` and calculate it then compare with the right side of `=` .
    class vp_expr_check_node: public vp_node
    {
    private:
    protected:
        virtual std::shared_ptr<vp_objects::vp_meta> handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) override;
    public:
        vp_expr_check_node(std::string node_name);
        ~vp_expr_check_node();
    };
}