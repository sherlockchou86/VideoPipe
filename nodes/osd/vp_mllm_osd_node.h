
#pragma once

#ifdef VP_WITH_LLM
#include <opencv2/imgproc.hpp>
#include <opencv2/freetype.hpp>

#include "../vp_node.h"

namespace vp_nodes {
    // on screen display(short as osd) node.
    // mainly used to display description(output from LLM) on frame.
    class vp_mllm_osd_node: public vp_node
    {
    private:
        // leave a gap at the bottom of osd frame
        int gap_height = 112;
        int padding = 5;

        // support chinese font
        cv::Ptr<cv::freetype::FreeType2> ft2;
        std::vector<std::string> utf8_split(const std::string& text);
        void draw_text_in_rect(cv::Mat& img,
                                const std::string& text,
                                const cv::Rect& rect,
                                int fontHeight,
                                cv::Scalar color);
    protected:
        virtual std::shared_ptr<vp_objects::vp_meta> handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) override;
    public:
        vp_mllm_osd_node(std::string node_name, std::string font);
        ~vp_mllm_osd_node();
    };
}
#endif