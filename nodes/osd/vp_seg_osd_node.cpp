#include <fstream>
#include "vp_seg_osd_node.h"
#include "../../utils/vp_utils.h"

namespace vp_nodes {
        
    vp_seg_osd_node::vp_seg_osd_node(std::string node_name, std::string classes_file, std::string colors_file): vp_node(node_name) {
        // load classes names if possible
        if (!classes_file.empty()) {
            std::ifstream ifs(classes_file.c_str());
            assert(ifs.is_open());
            std::string line;
            while (std::getline(ifs, line)) {
                classes.push_back(line);
            }
            ifs.close();
        }

        // load colors if possible
        if (!colors_file.empty()) {
            std::ifstream ifs(colors_file.c_str());
            assert(ifs.is_open());
            std::string line;
            while (std::getline(ifs, line)) {
                auto color_s = vp_utils::string_split(line, ',');
                cv::Vec3b color(static_cast<uchar>(std::stoi(color_s[0])), static_cast<uchar>(std::stoi(color_s[1])), static_cast<uchar>(std::stoi(color_s[2])));
                colors.push_back(color);
            }
            ifs.close();
        }

        this->initialized();
    }
    
    vp_seg_osd_node::~vp_seg_osd_node() {

    }

    std::shared_ptr<vp_objects::vp_meta> vp_seg_osd_node::handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) {
        // operations on osd_frame
        if (meta->osd_frame.empty()) {
            // add a gap at the left of osd frame
            meta->osd_frame = cv::Mat(meta->frame.rows, meta->frame.cols + gap, meta->frame.type(), cv::Scalar(255, 255, 255));
            
            // initialize by copying frame to osd frame
            auto roi = meta->osd_frame(cv::Rect(gap, 0, meta->frame.cols, meta->frame.rows));
            meta->frame.copyTo(roi);
        }
        
        // left for display color/class pairs
        auto canvas_left = meta->osd_frame(cv::Rect(0, 0, gap, meta->osd_frame.rows));
        // right for display result
        auto canvas_right = meta->osd_frame(cv::Rect(gap, 0, meta->frame.cols, meta->osd_frame.rows)); 

        if (!meta->mask.empty()) {
            cv::Mat segm;
            colorizeSegmentation(meta->mask, segm);

            cv::resize(segm, segm, canvas_right.size(), 0, 0, cv::INTER_NEAREST);
            cv::addWeighted(canvas_right, 0, segm, 1, 0.0, canvas_right);
        }

        if (!classes.empty()) {
            showLegend(canvas_left);
        }

        return meta;
    }


    void vp_seg_osd_node::colorizeSegmentation(const cv::Mat &score, cv::Mat &segm) {
        using namespace cv;
        const int rows = score.size[2];
        const int cols = score.size[3];
        const int chns = score.size[1];

        if (colors.empty()) {
            // Generate colors.
            colors.push_back(Vec3b());
            for (int i = 1; i < chns; ++i) {
                Vec3b color;
                for (int j = 0; j < 3; ++j)
                    color[j] = (colors[i - 1][j] + rand() % 256) / 2;
                colors.push_back(color);
            }
        }

        assert(colors.size() == chns);

        Mat maxCl = Mat::zeros(rows, cols, CV_8UC1);
        Mat maxVal(rows, cols, CV_32FC1, score.data);
        for (int ch = 1; ch < chns; ch++) {
            for (int row = 0; row < rows; row++) {
                const float *ptrScore = score.ptr<float>(0, ch, row);
                uint8_t *ptrMaxCl = maxCl.ptr<uint8_t>(row);
                float *ptrMaxVal = maxVal.ptr<float>(row);
                for (int col = 0; col < cols; col++) {
                    if (ptrScore[col] > ptrMaxVal[col]) {
                        ptrMaxVal[col] = ptrScore[col];
                        ptrMaxCl[col] = (uchar)ch;
                    }
                }
            }
        }

        segm.create(rows, cols, CV_8UC3);
        for (int row = 0; row < rows; row++) {
            const uchar *ptrMaxCl = maxCl.ptr<uchar>(row);
            Vec3b *ptrSegm = segm.ptr<Vec3b>(row);
            for (int col = 0; col < cols; col++) {
                ptrSegm[col] = colors[ptrMaxCl[col]];
            }
        }
    }

    void vp_seg_osd_node::showLegend(cv::Mat& board) {
        using namespace cv;
        auto kBlockHeight = 30;
        const int numClasses = (int)classes.size();

        for (int i = 0; i < numClasses; i++) {
            Mat block = board.rowRange(i * kBlockHeight, (i + 1) * kBlockHeight);
            block.setTo(colors[i]);
            putText(block, classes[i], Point(0, kBlockHeight / 2), FONT_HERSHEY_SIMPLEX, 0.5, Vec3b(255, 255, 255));
        }
    }
}