
#include "vp_expr_check_node.h"
#include "../../third_party/tinyexpr/tinyexpr.h"

namespace vp_nodes {
    
    vp_expr_check_node::vp_expr_check_node(std::string node_name):vp_node(node_name) {
        this->initialized();  
    }
    
    vp_expr_check_node::~vp_expr_check_node() {
        deinitialized();
    }

    std::shared_ptr<vp_objects::vp_meta> vp_expr_check_node::handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) {
        for (auto& i: meta->text_targets) {
            auto& text = i->text;
            auto left_right = vp_utils::string_split(text, '=');

            // only deal with equation such as `1+1=2`, excluding `1+1` or `1+1=` or `=1+1`
            /*
            * 1. yes means equation is right
            * 2. no means equation is wrong
            * 3. invalid means expression is not valid
            */
            if (left_right.size() == 2) {
                auto left = left_right[0];
                auto right = left_right[1];
                
                // replace specific symbols
                left = std::regex_replace(left, std::regex("x"), "*");
                left = std::regex_replace(left, std::regex("÷"), "/");
                //...

                if (left.empty()) {
                    i->flags = "invalid";
                    continue;
                }

                if (right.empty()) {
                    i->flags = "invalid";
                    continue;
                }

                double right_value;
                try {
                    right_value = std::stod(right);
                }
                catch(const std::exception& e) {
                    i->flags = "invalid";
                    continue;
                }

                // parse and calculate the left part
                int error;
                auto cal_value = te_interp(left.c_str(), &error);

                if (error != 0) {
                    i->flags = "invalid";
                    continue;
                }
                
                if (cal_value == right_value) {
                    i->flags = "yes_" + vp_utils::round_any(cal_value, 2);
                }
                else {
                    i->flags = "no_" + vp_utils::round_any(cal_value, 2);
                }
            }
            else {
                i->flags = "invalid";
            }
        }
        
        return meta;
    }
}