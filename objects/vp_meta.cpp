
#include "vp_meta.h"
#include "../excepts/vp_invalid_argument_error.h"

namespace vp_objects {
    
    vp_meta::vp_meta(vp_meta_type meta_type, int channel_index): 
        meta_type(meta_type), 
        channel_index(channel_index) {
            create_time = std::chrono::system_clock::now();
    }

    vp_meta::~vp_meta() {

    }

    std::string vp_meta::get_traces_str() {
        return "";
    }

    std::string vp_meta::get_meta_str() {
        return "";
    }

/*
    void vp_meta::attach_trace(std::string node_name) {
        if (trace_table.count(node_name)) {
            return;
        }
        
        std::map<vp_meta_trace_field, std::any> new_trace_record {
            {vp_meta_trace_field::SEQUENCE, trace_table.size()},
            {vp_meta_trace_field::NODE_NAME, node_name},
            {vp_meta_trace_field::IN_TIME, -1},
            {vp_meta_trace_field::OUT_TIME, -1},
            {vp_meta_trace_field::TEXT_INFO, std::vector<std::string>{}}
        };

        // append to the end of table
        trace_table[node_name] = new_trace_record;
    }

    void vp_meta::update_trace(std::string node_name, vp_meta_trace_field trace_key, std::any trace_value) {
        if (trace_table.count(node_name)) {
            auto & trace_record = trace_table[node_name];
            assert(trace_record.count(trace_key));

            switch (trace_key) {
                case vp_meta_trace_field::SEQUENCE:
                case vp_meta_trace_field::NODE_NAME:
                case vp_meta_trace_field::IN_TIME:
                case vp_meta_trace_field::OUT_TIME: {
                    // replace directly
                    trace_record[trace_key] = trace_value;
                    break;
                }
                case vp_meta_trace_field::TEXT_INFO: {
                    // append to the end of vector
                    auto & trace_desc = std::any_cast<std::vector<std::string>&>(trace_record[trace_key]);
                    trace_desc.push_back(std::any_cast<std::string>(trace_value));
                    break;
                }
                default: {
                    throw vp_excepts::vp_invalid_argument_error("invalid trace_key for meta!");
                    break;
                }
            }
        }
    }*/
}