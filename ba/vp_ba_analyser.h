#pragma once

#include <vector>
#include <tuple>
#include <memory>
#include <assert.h>

#include "vp_ba_flag.h"
#include "../objects/elements/vp_frame_element.h"
#include "../objects/vp_frame_target.h"
/* 
* ################################################
* how to add a new custom ba analyser?
* ################################################
* there are some built-in analysers already, but you can define yourself:
* 1. define a new flag in vp_ba_flag, such as vp_ba_flag::NEW_FLAG.
* 2. define a new analyser class derived from vp_ba_analyser.
* 3. override 'ba_ability()' method in new class, return the new flag defined in step 1.
* 4. override 'analyse(...)' method in new class, add ba logic.
* then, a new ba analyser is created. if you want to use this new analyser in your pipeline, you need:

* 1. initialize a analyser newly created above, and register it into vp_ba_node using vp_ba_node::register_ba_analysers(...).
* 2. turn on the ba ability of any frame elements in vp_ba_node by setting 'vp_frame_element::ba_abilities_mask|=vp_ba_flag::NEW_FLAG'.
* 3. then, the newly created ba analyser would work together with elements.
* note: ba analyser MUST work based on frame elements.
* #################################################
*/

namespace vp_ba {
    // base analyser class for behaviour analysis(short as ba)
    class vp_ba_analyser {
    private:
        /* data */
    public:
        vp_ba_analyser(/* args */);
        ~vp_ba_analyser();

        // non-copyable since it is unmeaningful.
        vp_ba_analyser(const vp_ba_analyser&) = delete;
        vp_ba_analyser& operator=(const vp_ba_analyser&) = delete;

        // which specific ba it can analyse, override in child class
        // each analyser can only handle one specific ba
        virtual vp_ba_flag ba_ability() = 0;

        // analyse method, override in child class
        // return a vector standing for the relationship of ba analyse results with 3 values:
        // 1. element: where
        // 2. target : who
        // 3. ba_flag: what 
        // the result is a vector since it is 1*n, multi targets to one element. 
        virtual std::vector<std::tuple<std::shared_ptr<vp_objects::vp_frame_element>, std::shared_ptr<vp_objects::vp_frame_target>, vp_ba::vp_ba_flag>> 
            analyse(std::shared_ptr<vp_objects::vp_frame_element>& element, std::vector<std::shared_ptr<vp_objects::vp_frame_target>>& targets) = 0;
    };

}