#pragma once

#include <string>
#include <vector>
#include <memory>

#include "../shapes/vp_point.h"
#include "../../ba/vp_ba_flag.h"

/*
* ###################################################
* what is frame element? see vp_frame_target also.
* ###################################################
* frame element is not target detected by deep learning models, it is usually configured manually by us for ba(short for behaviour analysis) purpose.
* for example, we can 'draw' a region in video scene and then make a rule that all vehicles are forbidden to drive in, the 'region' here is such a element.
* there are some kinds of built-in elements such as 'lane', 'no_parking_zone', the former stands for a lane on road and the latter stands for a region where parking is not allowed.
* each frame element has a data member indicating the shape of element(line/polygon/...), they are relationship of 'has-a', not is-a.
*
* all ba logic are based on frame elements, we can define new frame elements by deriving from vp_frame_element, and assign the vp_ba_flag needed to care for to vp_frame_element::ba_abilities_mask. 
* ###################################################
*/
namespace vp_objects {
    // element in video scene
    // commonly used for ba, can be a region/line... in video scene.
    class vp_frame_element
    {
    private:
        /* data */
    public:
        vp_frame_element(int element_id, std::string element_name = "", int ba_abilities_mask = 0);
        ~vp_frame_element();

        // ba flags of the element, hold by this value (created/updated by vp_ba_node).
        // for example, 0001/0010/0100/1000 stands for 4 different flags, 1110 means 3 flags are on and another one is off, using ^|& operators to update and read. 
        // if 0100 stands for 'Stop' flag of element,  'ba_flags|=0100' means set 'Stop' flag as On, '(ba_flags & 0100) == 0100' means 'Stop' flag is already On. 
        // see vp_frame_target also.
        int ba_flags;
        
        // which specific ba it need to care for, each element can care for multi ba.
        const int ba_abilities_mask;

        // unique id for element
        const int element_id;

        // name of the element, used for print or log.
        const std::string element_name;

        // check whether the element has ba ability
        bool check_ba_ability(vp_ba::vp_ba_flag flag);

        // retrive all vertexs of the element
        virtual std::vector<vp_objects::vp_point> key_points() = 0;

        // virtual clone method since we do not know what specific element we need copy in some situations, return a new pointer pointting to new memory allocation in heap.
        // note: every child class need implement its own clone() method.
        // see vp_meta also.
        virtual std::shared_ptr<vp_frame_element> clone() = 0;
    };

}