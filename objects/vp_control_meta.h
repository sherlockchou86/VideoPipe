#pragma once

#include <chrono>
#include "vp_meta.h"


namespace vp_objects {
    // type of control meta
    enum vp_control_type {
        SPEAK,
        VIDEO_RECORD,
        IMAGE_RECORD
    };

    // control meta, which contains control data.
    class vp_control_meta: public vp_meta {
    private:
        // help to generate control uid if need
        void generate_uid();
    public:
        vp_control_meta(vp_control_type control_type, int channel_index, std::string control_uid = "");
        ~vp_control_meta();

        vp_control_type control_type;
        // unique id to identify control meta (caould be generated in random)
        std::string control_uid;

        // copy myself
        virtual std::shared_ptr<vp_meta> clone() override;
    };

}