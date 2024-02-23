

#include "vp_meta_publisher.h"

namespace vp_nodes {
    vp_meta_publisher::vp_meta_publisher(/* args */) {

    }

    vp_meta_publisher::~vp_meta_publisher() {

    }

    void vp_meta_publisher::add_subscriber(std::shared_ptr<vp_meta_subscriber> subscriber) {
        std::lock_guard<std::mutex> guard(this->subscribers_lock);
        this->subscribers.push_back(subscriber);
    }

    void vp_meta_publisher::remove_subscriber(std::shared_ptr<vp_meta_subscriber> subscriber) {
        std::lock_guard<std::mutex> guard(this->subscribers_lock);
        for (auto i = this->subscribers.begin(); i != this->subscribers.end();) {
            if(*i == subscriber) {
                i = this->subscribers.erase(i);
            }
            else {
                i++;
            }
        }     
    }

    // by default, we push meta to next nodes indiscriminately, each next node has the same meta pointer.
    // in some situations, we need push meta depend on condition, refer to vp_split_node which would push meta by channel index or push a deep copy pf meta(new pointer to new memory). 
    void vp_meta_publisher::push_meta(std::shared_ptr<vp_objects::vp_meta> meta) {
        std::lock_guard<std::mutex> guard(this->subscribers_lock);
        for (auto i = this->subscribers.begin(); i != this->subscribers.end(); i++) {
            (*i)->meta_flow(meta);
        }
    }
}