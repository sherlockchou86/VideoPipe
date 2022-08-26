#pragma once

#include <memory>
#include <vector>
#include <mutex>
#include <iostream>

#include "vp_meta_subscriber.h"

namespace vp_nodes {
    class vp_meta_publisher {
    private:
    protected:
        // push meta to next nodes
        virtual void push_meta(std::shared_ptr<vp_objects::vp_meta> meta);

        // non-copyable for all child class
        std::mutex subscribers_lock;
        // next nodes as subscribers
        std::vector<std::shared_ptr<vp_meta_subscriber>> subscribers;
    public:
        vp_meta_publisher(/* args */);
        ~vp_meta_publisher();

        // add next node
        void add_subscriber(std::shared_ptr<vp_meta_subscriber> subscriber);
        // remove next node
        void remove_subscriber(std::shared_ptr<vp_meta_subscriber> subscriber);
    };

}

