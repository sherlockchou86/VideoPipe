#pragma once

namespace vp_ba {
    
    // define ba flags
    // each flag stands for one result of ba
    enum vp_ba_flag {
        NONE = 0b0000000000000000,       // none flag
        STOP = 0b0000000000000001,       // single target stops and keep still
        ENTER = 0b0000000000000010,      // single target enters a region
        LEAVE = 0b0000000000000100,      // single target leaves a region
        CROSSLINE = 0b0000000000001000,  // single target cross a line
        JAM = 0b0000000000010000,        // many targets keep still in a region(traffic jam or people gathering)
        HOVER = 0b0000000000100000       // single target wanders in a region
    };

}