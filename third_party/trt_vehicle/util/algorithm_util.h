#ifndef __IMAGE_ANALYSIS_ROI_OBJECT_BOX_UTIL_H__
#define __IMAGE_ANALYSIS_ROI_OBJECT_BOX_UTIL_H__

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>

using namespace std;

namespace trt_vehicle{
    
    struct GridCoord
    {
        int x;
        int y;
        int i;
    };

    struct ObjBox
    {
        int x;
        int y;
        int width;
        int height;
        float score;
        int class_;
        std::string label;
        std::vector<GridCoord> gridCoords;
    };
    
    struct ObjCls {
        int class_;
        std::string label;
        float score;
    };

    float overlap( float x1, float w1, float x2, float w2);

    //calculation intersection area
    float boxIntersection(ObjBox& a, ObjBox& b);

    //calculation union area
    float boxUnion(ObjBox& a, ObjBox& b);

    //calculate iou of tow boxes
    float rectIou(ObjBox& a, ObjBox& b);
    float rectIouRegion(ObjBox& a, ObjBox& b);
    float rectIouDivOne(ObjBox& a, ObjBox& b);

    //coords change : from xyxy to objBox
    void xyxy2objBox(int x1,int y1,int x2,int y2, ObjBox& b);

    //Sort from large to small , return indeices
    std::vector<int> getSortIndex(std::vector<float>& scores);

    //nms for bounding boxes
    int nonMaximumSuppression(vector<ObjBox>& boxes, vector<float>& scores, float overlapThreshold,vector<ObjBox>& outBoxes);

    //difference map pf two pictures
    cv::Mat diffImgs(cv::Mat bgImg,cv::Mat frontImg);

    //sort
    void sort(std::vector<float>& scores,std::vector<float>& coordXs,std::vector<float>& coordYs);
    void sort(std::vector<double>& scores);
    
    //draw boxes to img , and return the drawed img
    cv::Mat drawRectangle(cv::Mat img,std::vector<ObjBox>& boxes,cv::Scalar color);

    //Calculate the Euclidean distance between two vectors
    double calVectorDistance(std::vector<float> f1,std::vector<float> f2);

    //Batch calculation of Euclidean distance between two vectors
    std::vector<double> calVectorDistances(std::vector<std::vector<float>> fs1,std::vector<std::vector<float>> fs2);

    //Generate binary vector from distance value(score)
    void genBinaryVectorfromScore(vector<int> binaryScores,vector<double> scores,int imgW,int imgH,int downScale);

    //Generate gray image from distance value(score)
    void genImgfromScore(cv::Mat& img,vector<double> scores,int imgW,int imgH,int downScale);

    //filter rectangles through regions of interest region
    std::vector<ObjBox> selectRectByInterestMask(std::vector<ObjBox>& rects,cv::Mat interestedMask);
    
    //find adjacent grids
    int disInVector(GridCoord gridCoord,std::vector<std::vector<GridCoord>>& indicesAll);

    //generate rectangles using distance vector
    void genBoxesFromVector(std::vector<double>& dis,int gridW,int downScale,std::vector<ObjBox>* rects,std::vector<std::vector<GridCoord>>* indicesAll);

    //generate rectangles using binary
    std::vector<ObjBox> genBoxesFromBinary(cv::Mat& binary);

    //compare the current image with the background image to find different areas
    bool compareAlarmRectToBg(ObjBox rectCurr,std::vector<ObjBox>& rects,ObjBox& rect);

    //filter rectangular boxes by scale / aspect ratio / region of interest / iou , etc
    class RectSelector{
        private:
            std::vector<ObjBox> m_rects;
            int m_cursor;
            ObjBox next();
            bool hasNext();
            void remove();
            void remove_inside(ObjBox rect);
            void remove_overlop(ObjBox rect,float iouThres);
        public:
            void init(std::vector<ObjBox> rects);
            void rmInside();
            void rmOverlop(float iouThres);
            void rmLittle(int minW,int minH);
            void rmBig(int maxW,int maxH);
            void rmBigRatio(int maxW,int maxH,int imgH, float ratio);
            void rmBigWithRoadMask(cv::Mat& mask,std::vector<float>& roadNumsList,float carLeastWidth);
            void rmRatio();
            std::vector<ObjBox> getRect();
    };

}
#endif
