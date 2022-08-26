#pragma once

#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <string>
#include <memory>
#include <chrono>

#include "../utils/vp_semaphore.h"
#include "vp_meta_publisher.h"
#include "vp_meta_hookable.h"
#include "../objects/vp_control_meta.h"
#include "../objects/vp_frame_meta.h"
#include "../excepts/vp_invalid_calling_error.h"

namespace vp_nodes {
    // node type
    enum vp_node_type {
        SRC,  // src node, must not have input branchs
        DES,  // des node, must not have output branchs
        MID   // middle node, can have input and output branchs
    };

    // base class for all nodes
    class vp_node: public vp_meta_publisher, 
                    public vp_meta_subscriber, 
                    public vp_meta_hookable, 
                    public std::enable_shared_from_this<vp_node> {
    private:
        // previous nodes
        std::vector<std::shared_ptr<vp_node>> pre_nodes;

        // handle thread
        std::thread handle_thread;
        // dispatch thread
        std::thread dispatch_thread;
    protected:
        // status of the node
        bool active = false;

        // by default we handle frame meta one by one, in some situations we need handle them batch by batch(such as vp_infer_node).
        // setting this member greater than 1 means the node will handle frame meta with batch, and vp_node::handle_frame_meta_by_batch(...) will be called other than vp_node::handle_frame_meta(...).
        // note: control meta is not allowed like above, only one by one supported.
        int frame_meta_handle_batch = 1;

        // cache input meta from previous nodes
        std::queue<std::shared_ptr<vp_objects::vp_meta>> in_queue;
        // 
        std::mutex in_queue_lock;
        // cache output meta to next nodes
        std::queue<std::shared_ptr<vp_objects::vp_meta>> out_queue;

        // synchronize for in_queue
        vp_utils::vp_semaphore in_queue_semaphore;
        // synchronize for out_queue
        vp_utils::vp_semaphore out_queue_semaphore;

        // get meta from in_queue, handle meta and put them into out_queue looply.
        // we need re-implement(define how to create meta and put it into out_queue) in src nodes since they have no previous nodes. 
        virtual void handle_run();
        // get meta from out_queue and push them to next nodes looply.
        // we need re-implement(just do nothing) in des nodes since they have no next nodes. 
        virtual void dispatch_run();

        // define how to handle frame meta [one by one], ignored in src nodes.
        // return nullptr means do not push it to next nodes, such as des nodes which have no next nodes.
        virtual std::shared_ptr<vp_objects::vp_meta> handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta);
        // define how to handle control meta, ignored in src nodes. 
        // return nullptr means do not push it to next nodes, such as des nodes which have no next nodes.
        virtual std::shared_ptr<vp_objects::vp_meta> handle_control_meta(std::shared_ptr<vp_objects::vp_control_meta> meta);

        // define how to handle frame meta [batch by batch], ignored in src nodes.
        virtual void handle_frame_meta(const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& meta_with_batch);

        // called by child class after all resources have been initialized.
        void initialized();

        // protected as it can't be instanstiated directly.
        vp_node(std::string node_name);
    public:
        ~vp_node();
        // clear meaningful string, such as 'file_src_0' stands for file source node at channel 0.
        std::string node_name;

        // receive meta from previous nodes, 
        // we can hook(such as modifying meta) in child class but do not forget calling vp_node.meta_flow(...) followly.
        virtual void meta_flow(std::shared_ptr<vp_objects::vp_meta> meta) override;

        // retrive current node type
        virtual vp_node_type node_type();
        
        // detach myself from all previous nodes
        void detach();
        // attach myself to previous nodes(can be a list)
        void attach_to(std::vector<std::shared_ptr<vp_node>> pre_nodes);

        // get next nodes
        std::vector<std::shared_ptr<vp_node>> next_nodes();

        // get description of node
        virtual std::string to_string();
    };

}