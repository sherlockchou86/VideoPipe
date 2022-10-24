#pragma once

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
        /* data */
    public:
        vp_control_meta(vp_control_type control_type, int channel_index);
        ~vp_control_meta();

        vp_control_type control_type;

        // copy myself
        virtual std::shared_ptr<vp_meta> clone() override;
    };

}