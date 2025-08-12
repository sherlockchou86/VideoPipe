
#include "vp_mllm_osd_node.h"

#ifdef VP_WITH_LLM
namespace vp_nodes {
        
    vp_mllm_osd_node::vp_mllm_osd_node(std::string node_name, std::string font): vp_node(node_name) {
        ft2 = cv::freetype::createFreeType2();
        ft2->loadFontData(font, 0);
        this->initialized();
    }
    
    vp_mllm_osd_node::~vp_mllm_osd_node() {
        deinitialized();
    }

    std::shared_ptr<vp_objects::vp_meta> vp_mllm_osd_node::handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) {
        if (meta->description.empty()) {
            return meta;
        }
        
        // operations on osd_frame
        if (meta->osd_frame.empty()) {
            // add a gap at the bottom of osd frame
            meta->osd_frame = cv::Mat(meta->frame.rows + gap_height + padding * 2, meta->frame.cols, meta->frame.type(), cv::Scalar(128, 128, 128));
            
            // initialize by copying frame to osd frame
            auto roi = meta->osd_frame(cv::Rect(0, 0, meta->frame.cols, meta->frame.rows));
            meta->frame.copyTo(roi);
        }

        auto& canvas = meta->osd_frame;
        //ft2->putText(canvas, meta->description, cv::Point(10, meta->frame.rows + gap_height + padding * 2), 20, cv::Scalar(255, 0, 0), cv::FILLED, cv::LINE_AA, true);
        draw_text_in_rect(canvas, meta->description, cv::Rect(5, meta->frame.rows + 5, meta->frame.cols - 10, gap_height + padding * 2 - 10), 20, cv::Scalar(255, 0, 0));

        // log using INFO level
        VP_INFO(vp_utils::string_format("[%s] [%s]", node_name.c_str(), meta->description.c_str()));
        return meta;
    }

    std::vector<std::string> vp_mllm_osd_node::utf8_split(const std::string& text) {
        std::vector<std::string> chars;
        for (size_t i = 0; i < text.size();) {
            unsigned char c = text[i];
            size_t len = 1;
            if ((c & 0x80) == 0x00) len = 1;          // ASCII
            else if ((c & 0xE0) == 0xC0) len = 2;     // 2字节
            else if ((c & 0xF0) == 0xE0) len = 3;     // 3字节
            else if ((c & 0xF8) == 0xF0) len = 4;     // 4字节
            chars.push_back(text.substr(i, len));
            i += len;
        }
        return chars;
    }

    void vp_mllm_osd_node::draw_text_in_rect(cv::Mat& img,
                        const std::string& text,
                        const cv::Rect& rect,
                        int fontHeight,
                        cv::Scalar color) {
        std::vector<std::string> chars = utf8_split(text);
        std::string currentLine;
        int baseline = 0;

        int y = rect.y; // 初始绘制高度

        for (size_t i = 0; i < chars.size(); i++) {
            std::string tempLine = currentLine + chars[i];
            cv::Size textSize = ft2->getTextSize(tempLine, fontHeight, -1, &baseline);

            if (textSize.width > rect.width && !currentLine.empty()) {
                // 绘制当前行
                int drawY = y + textSize.height;
                if (drawY > rect.y + rect.height) break; // 超出矩形区域
                ft2->putText(img, currentLine, cv::Point(rect.x, drawY),
                            fontHeight, color, -1, 8, true);
                y += textSize.height + 5; // 行间距
                currentLine.clear();
            }
            currentLine += chars[i];
        }

        // 绘制最后一行
        if (!currentLine.empty()) {
            int drawY = y + ft2->getTextSize(currentLine, fontHeight, -1, &baseline).height;
            if (drawY <= rect.y + rect.height) {
                ft2->putText(img, currentLine, cv::Point(rect.x, drawY),
                            fontHeight, color, -1, 8, true);
            }
        }
    }
}
#endif