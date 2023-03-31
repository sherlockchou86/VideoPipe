#pragma once

#include <string>
#include <map>
#include <any>
#include <vector>
#include <memory>
#include <chrono>
#include <assert.h>

namespace vp_objects {

    // meta type
    enum vp_meta_type {
        FRAME,
        CONTROL
    };

    // meta trace field
    // 1. sequence   ->int       ,sequence number the meta flowing through pipeline
    // 2. node_name  ->string    ,name of current node the meta flow through
    // 3. in_time    ->long      ,time when the meta arrive current node
    // 4. out_time   ->long      ,time when the meta leave current node
    // 5. text_info  ->vector    ,text info while the meta inside node
    enum vp_meta_trace_field {
        SEQUENCE,
        NODE_NAME,
        IN_TIME,
        OUT_TIME,
        TEXT_INFO
    };

    // base class for meta
    class vp_meta {
    private:
    
    protected:
        /*
        trace table, single record describe how meta flows in each node.
        {
            "file_src_0": {
                "sequence": 0,
                "node_name": "file_src_0",
                "in_time": 1692384,
                "out_time": 1692384,
                "trace_desc": ["create frame meta at time 1692384", "...", "..."]
            },
            "primary_infer": {
                "sequence": 1,
                "node_name": "file_src_0",
                "in_time": 1692384,
                "out_time": 1692384,
                "trace_desc": ["add 6 targets in frame meta at time 1692384", "...", "..."]
            },
            "secondary_infer": {
                "sequence": 2,
                "node_name": "file_src_0",
                "in_time": 1692384,
                "out_time": 1692384,
                "trace_desc": ["updated 6 targets in target_list at 1692384", "...", "..."]
            },
            "osd": {
                "sequence": 3,
                "node_name": "file_src_0",
                "in_time": 1692384,
                "out_time": 1692384,
                "trace_desc": ["draw 6 targets in on frame at time 1692384", "...", "..."]
            },
            "file_des_0": {
                "sequence": 4,
                "node_name": "file_src_0",
                "in_time": 1692384,
                "out_time": 1692384,
                "trace_desc": ["balabala", "...", "..."]
            }
        }
        */
        //std::map<std::string, std::map<vp_meta_trace_field, std::any>> trace_table;
    public:
        vp_meta(vp_meta_type meta_type, int channel_index);
        ~vp_meta();

        // the time when meta created
        std::chrono::system_clock::time_point create_time;

        vp_meta_type meta_type;

        // channel the meta belongs to
        int channel_index;

        // write trace record or not
        bool trace_on = false;

        // format traces string
        virtual std::string get_traces_str();

        // format meta string
        virtual std::string get_meta_str();

        // virtual clone method since we do not know what specific meta we need copy in some situations, return a new pointer pointting to new memory allocation in heap.
        // note: every child class need implement its own clone() method.
        virtual std::shared_ptr<vp_meta> clone() = 0;

        // attach a new trace record for specific node (initialize key-value for current node)
        //void attach_trace(std::string node_name);

        // update trace record
        //void update_trace(std::string node_name, vp_meta_trace_field trace_key, std::any trace_value);
    };

}