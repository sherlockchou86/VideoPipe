
#pragma once

#include <memory>
#include <vector>
#include <string>

namespace vp_objects {
    class vp_frame_text_target
    {
    private:
        /* data */
    public:
        vp_frame_text_target(std::vector<std::pair<int, int>> region_vertexes, std::string text, float score);
        ~vp_frame_text_target();

        std::vector<std::pair<int, int>> region_vertexes;
        std::string text;
        float score;

        // flags for text
        std::string flags = "";
        // clone myself
        std::shared_ptr<vp_frame_text_target> clone();
    };
}