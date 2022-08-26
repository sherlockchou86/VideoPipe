
#include "vp_node.h"

namespace vp_nodes {

    // split pipeline into multi branches. although all other non-Des nodes have the ability to split pipeline, but vp_split_node has more parameters and flexible behaviour.  
    // by default, other non-Des nodes split pipeline by simple copying, which just copy pointer of meta and push to next nodes, each next node handle the same meta allocated in heap(not thread-safe).
    // in addition, other non-Des nodes push meta to next nodes without difference, each next node receive equal number of meta to previous node.  
    // 
    // in vp_split_node, we have below parameters to set:
    // split_with_channel_index (false by default): if true, push meta according to its channel index, only those next nodes having the same channel index can receive meta.
    // split_with_deep_copy (false by default)    : if true, copy meta in heap and create new pointer, then push the new pointer to next nodes, next nodes have a totally different meta with previous.
    // 
    // note: split_with_deep_copy = true will affect the performance of pipeline.
    // above paramters can be set as true at the same time.
    class vp_split_node: public vp_node
    {
    private:
        /* data */
    protected:
        // re-implement how to push meta to next nodes.
        virtual void push_meta(std::shared_ptr<vp_objects::vp_meta> meta) override;
    public:
        vp_split_node(std::string node_name, bool split_with_channel_index = false, bool split_with_deep_copy = false);
        ~vp_split_node();

        bool split_with_channel_index;
        bool split_with_deep_copy;
    };

}