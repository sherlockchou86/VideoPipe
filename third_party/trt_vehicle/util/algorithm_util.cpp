#include "algorithm_util.h"

namespace trt_vehicle{

    float overlap( float x1, float w1, float x2, float w2)
    {
        float l1 = x1;
        float l2 = x2;
        float left = ( l1 > l2 ) ? l1 : l2;
        float r1 = x1 + w1;
        float r2 = x2 + w2;
        float right = ( r1 < r2 ) ? r1 : r2;
        return ( right - left );
    }

    //calculation intersection area
    float boxIntersection(ObjBox& a, ObjBox& b)
    {
        float w = overlap( a.x, a.width, b.x, b.width);
        float h = overlap( a.y, a.height, b.y, b.height);
        if( w < 0 || h < 0 )
        {
            return 0.0;
        }

        return ( w * h );
    }

    //calculation union area
    float boxUnion(ObjBox& a, ObjBox& b)
    {
        float comm_area = boxIntersection( a, b );
        return ( a.width * a.height + b.width * b.height - comm_area );
    }

    //calculate iou of tow boxes
    float rectIou(ObjBox& a, ObjBox& b)
    {
        return boxIntersection(a, b)/boxUnion(a, b);
    }

    //calculate iou of tow boxes
    float rectIouRegion(ObjBox& a, ObjBox& b)
    {
        int inner = 0;
        for(int i=0;i<a.gridCoords.size();i++){
            for(int j=0;j<b.gridCoords.size();j++){
                if(a.gridCoords[i].i == b.gridCoords[j].i){
                    inner++;
                    break;
                }
            }
        }
        return inner*1.0/(a.gridCoords.size()+b.gridCoords.size()-inner);
    }

    //calculate iou of tow boxes
    float rectIouDivOne(ObjBox& a, ObjBox& b)
    {
        float inner = boxIntersection(a, b)*1.0;
        return max(inner/(a.width * a.height),inner/(b.width * b.height));
    }

    //coords change : from xyxy to objBox
    void xyxy2objBox(int x1,int y1,int x2,int y2, ObjBox& b )
    {
        b.x = x1;
        b.width = x2-x1;
        b.y = y1;
        b.height = y2-y1;
    }

    //Sort from large to small , return indeices
    std::vector<int> getSortIndex(std::vector<float>& scores)
    {
        std::vector<int> indices;
        for (int i = 0; i < scores.size(); i++)
            indices.push_back(i);
        for (int i = 0; i < scores.size(); i++)
            for (int j = i + 1; j < scores.size(); j++)
            {
                if(scores[j] > scores[i])
                {
                    int tmp = indices[i];
                    indices[i] = indices[j];
                    indices[j] = tmp;
                    float tmpS = scores[i];
                    scores[i] = scores[j];
                    scores[j] = tmpS;
                }
            }
       return indices;
    }

    //nms for bounding boxes
    int nonMaximumSuppression(vector<ObjBox>& boxes, vector<float>& scores, float overlapThreshold,vector<ObjBox>& outBoxes)
    {
        outBoxes.clear();
        std::vector<int> indices = getSortIndex(scores);
        int numBoxes = scores.size();
        vector<float> box_area(numBoxes);
        vector<bool> is_suppressed(numBoxes);
        for (int i = 0; i < numBoxes; i++)
        {
            is_suppressed[i] = false;
            box_area[i] = (float)(boxes[i].width*boxes[i].height);
        }
        for (int i = 0; i < numBoxes; i++)
        {
            if (!is_suppressed[indices[i]])
            {
                outBoxes.push_back(boxes[indices[i]]);
                for (int j = i + 1; j < numBoxes; j++)
                {
                    if (!is_suppressed[indices[j]])
                    {
                        if (rectIou(boxes[indices[i]],boxes[indices[j]]) > overlapThreshold)
                        {
                            is_suppressed[indices[j]] = true;
                        }
                    }
                }
            }
        }
        return true;
    }


    //difference map pf two pictures
    cv::Mat diffImgs(cv::Mat bgImg,cv::Mat frontImg){
        cv::Mat bgGray,frontGray,binary,diff;
        if (bgImg.channels() == 3)
            cv::cvtColor(bgImg, bgGray, cv::COLOR_BGR2GRAY);
        else
            bgGray = bgImg.clone();
        if (frontImg.channels() == 3)
            cv::cvtColor(frontImg, frontGray, cv::COLOR_BGR2GRAY);
        else
            frontGray = frontImg.clone();
        cv::absdiff(bgGray, frontGray, diff);
        cv::threshold(diff, binary, 5, 255, 0);
        cv::Mat kernel5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::Mat kernel7 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
        cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel5);
        cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel7);
        return binary;
    }

    //sort
    void sort(std::vector<float>& scores,std::vector<float>& coordXs,std::vector<float>& coordYs)
    {
        for (int i = 0; i < scores.size(); i++)
            for (int j = i + 1; j < scores.size(); j++)
            {
                if(scores[j] < scores[i])
                {
                    float s = scores[i];
                    scores[i] = scores[j];
                    scores[j] = s;
                    float x = coordXs[i];
                    coordXs[i] = coordXs[j];
                    coordXs[j] = x;
                    float y = coordYs[i];
                    coordYs[i] = coordYs[j];
                    coordYs[j] = y;
                }
            }
    }
    
    //sort
    void sort(std::vector<double>& scores)
    {
        for (int i = 0; i < scores.size(); i++)
            for (int j = i + 1; j < scores.size(); j++)
            {
                if(scores[j] < scores[i])
                {
                    double s = scores[i];
                    scores[i] = scores[j];
                    scores[j] = s;
                }
            }
    }

    //draw boxes to img , and return the drawed img
    cv::Mat drawRectangle(cv::Mat img,std::vector<ObjBox>& boxes,cv::Scalar color){
        cv::Mat imgDraw = img.clone();
        for(int i=0;i<boxes.size();i++){
            ObjBox box = boxes[i];
            cv::Rect rect=cv::Rect(int(box.x),int(box.y),int(box.width),int(box.height));
            cv::rectangle(imgDraw,rect,color,1);
        }
        return imgDraw;
    }

    //Calculate the Euclidean distance between two vectors
    double calVectorDistance(std::vector<float> f1,std::vector<float> f2){
        double dDisTmp = 0.0;
        double eDisTmp1 = 0.0;
        double eDisTmp2 = 0.0;
        for(int i=0;i<f1.size();i++){
            eDisTmp1 += f1[i]*f1[i];
            eDisTmp2 += f2[i]*f2[i];
        }
        double eDis1 = sqrt(eDisTmp1);
        double eDis2 = sqrt(eDisTmp2);
        for(int i=0;i<f1.size();i++){
            dDisTmp += pow((f1[i]/eDis1-f2[i]/eDis2),2);
        }
        double dis = sqrt(dDisTmp);
        return dis;
    }

    //Batch calculation of Euclidean distance between two vectors
    std::vector<double> calVectorDistances(std::vector<std::vector<float>> fs1,std::vector<std::vector<float>> fs2){
        std::vector<double> distants;
        for(int i=0;i<fs1.size();i++){
            distants.push_back(calVectorDistance(fs1[i],fs2[i]));
        }
        return distants;
    }

    //Generate binary vector from distance value(score)
    void genBinaryVectorfromScore(vector<int> binaryScores,vector<double> scores,int imgW,int imgH,int downScale){
        binaryScores.clear();
        int gridW = int(imgW/downScale);
        int gridH = int(imgH/downScale);
        for(int i=0;i<gridH*gridW;i++){
            int score = std::min(255,int(scores[i]*255));
            if(score > 155){
                binaryScores.push_back(1);
            }else {
                binaryScores.push_back(0);
            }
        }
    }

    //Generate gray image from distance value(score)
    void genImgfromScore(cv::Mat& img,vector<double> scores,int imgW,int imgH,int downScale){
        img = img*0;
        int gridW = int(imgW/downScale);
        int gridH = int(imgH/downScale);
        for(int i=0;i<gridH;i++){
            for(int j=0;j<gridW;j++){
                int score = std::min(255,int(scores[i*gridW+j]*255));
                if(score > 100){
                    cv::Rect roi_rect = cv::Rect(j*downScale, i*downScale, downScale,downScale);
                    img(roi_rect).setTo(score);
                }
            }
        }
    }

    //filter rectangles through regions of interest region
    std::vector<ObjBox> selectRectByInterestMask(std::vector<ObjBox>& rects,cv::Mat interestedMask){
                    std::vector<ObjBox>  interestedRects;
                    interestedRects.clear();
                    for(int i=0;i<rects.size();i++){
                        ObjBox box = rects[i];
                        cv::Rect rect=cv::Rect(int(box.x),int(box.y),int(box.width),int(box.height));
                        cv::Mat imgSeg = interestedMask(rect);
                        int noZeroCount = cv::countNonZero(imgSeg);
                        float ratio = noZeroCount*1.0/(box.width*box.height);
                        if(ratio>0.5){
                            interestedRects.push_back(box);
                        }
                    }
                    return interestedRects;
                }

    //find adjacent grids
    int disInVector(GridCoord gridCoord,std::vector<std::vector<GridCoord>>& indicesAll){
        int index = -1;
        for(int i=0;i<indicesAll.size();i++){
            for(int j=0;j<indicesAll[i].size();j++){
                if(abs(gridCoord.x-indicesAll[i][j].x)<=1 && abs(gridCoord.y-indicesAll[i][j].y)<=1){
                    return i;
                }
            }
        }
        return index;
    }

    //generate rectangles using distance vector
    void genBoxesFromVector(std::vector<double>& dis,int gridW,int downScale,std::vector<ObjBox>* rects,std::vector<std::vector<GridCoord>>* indicesAll){
        for(int i=0; i<dis.size(); i++){
            if(dis[i] == 0){
                continue;
            }
            int x = i%gridW;
            int y = i/gridW;
            GridCoord gridCoord;
            gridCoord.x = x;
            gridCoord.y = y;
            gridCoord.i = i;
            int index = disInVector(gridCoord,*indicesAll);
            if(index >= 0){
                (*indicesAll)[index].push_back(gridCoord);
            }else{
                std::vector<GridCoord> indicesSingle;
                indicesSingle.push_back(gridCoord);
                (*indicesAll).push_back(indicesSingle);
            }
        }

        for(int i=0;i<indicesAll->size();i++){
            ObjBox rect;
            int minX=9999;
            int minY=9999;
            int maxX=-1;
            int maxY=-1;
            for(int j=0;j<(*indicesAll)[i].size();j++){
                if(minX>(*indicesAll)[i][j].x){
                    minX = (*indicesAll)[i][j].x;
                }
                if(minY>(*indicesAll)[i][j].y){
                    minY = (*indicesAll)[i][j].y;
                }
                if(maxX<(*indicesAll)[i][j].x){
                    maxX = (*indicesAll)[i][j].x;
                }
                if(maxY<(*indicesAll)[i][j].y){
                    maxY = (*indicesAll)[i][j].y;
                }
            }
            rect.x = minX*downScale;
            rect.y = minY*downScale;
            rect.width = (maxX-minX+1)*downScale;
            rect.height = (maxY-minY+1)*downScale;
            rects->push_back(rect);
        }

    }

    //generate rectangles using binary
    std::vector<ObjBox> genBoxesFromBinary(cv::Mat& binary){
        std::vector<vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarcy;

        cv::findContours(binary, contours, hierarcy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point());
        std::vector<ObjBox> rects;
        rects.clear();
        for( int i = 0 ; i < contours.size(); i++){
            cv::Rect rect = cv::boundingRect(cv::Mat(contours[i]));
            ObjBox r;
            r.x = rect.x;
            r.y = rect.y;
            r.width = rect.width;
            r.height = rect.height;
            rects.push_back(r);
        }
        return rects;
    }

    //compare the current image with the background image to find different areas
    bool compareAlarmRectToBg(ObjBox rectCurr,std::vector<ObjBox>& rects,ObjBox& rect){

        float iouMax = 0.0;
        int index = -1;
        for( int i = 0 ; i < rects.size(); i++){
            ObjBox rect = rects[i];
            float iou = rectIou(rectCurr,rect);
            if(iou > iouMax){
                iouMax = iou;
                index = i;
            }
        }
        if(iouMax > 0.25){
            rect = rects[index];
            rects.erase( rects.begin() + index);
            return true;
        }
        return false;
    }

    //filter rectangular boxes by scale / aspect ratio / region of interest / iou , etc
    void RectSelector::init(std::vector<ObjBox> rects){
        m_rects.clear();
        m_rects = rects;
        m_cursor = -1;
    }

    ObjBox RectSelector::next(){
        m_cursor = m_cursor+1;
        return m_rects[m_cursor];
    }

    bool RectSelector::hasNext(){
        return m_rects.size()> m_cursor+1;
    }

    void RectSelector::remove(){
        m_rects.erase( m_rects.begin() + m_cursor );
        m_cursor = m_cursor-1;
    }

    void RectSelector::remove_inside(ObjBox rect){
        for( int i = 0 ; i < m_rects.size(); i++){
            if(i == m_cursor){continue;}
            ObjBox r = m_rects[i];
            if(rect.y>=r.y && rect.y+rect.height<=r.y+r.height && rect.x>=r.x && rect.x+rect.width<=r.x+r.width){
                remove();
                break;
            }
        }
    }
    void RectSelector::remove_overlop(ObjBox rect,float iouThres){
        for( int i = 0 ; i < m_rects.size(); i++){
            if(i == m_cursor){continue;}
            ObjBox r = m_rects[i];
            float boxIou = rectIou(r, rect);
            if(boxIou>iouThres){
                remove();
                break;
            }
        }
    }

    void RectSelector::rmInside(){
        m_cursor = -1;
        while(hasNext()){
            ObjBox rect = next();
            remove_inside(rect);
        }
    }

    void RectSelector::rmOverlop(float iouThres){
        m_cursor = -1 ;
        while(hasNext()){
            ObjBox rect = next();
            remove_overlop(rect,iouThres);
        }
    }

    void RectSelector::rmLittle(int minW,int minH){
        m_cursor = -1 ;
        while(hasNext()){
            ObjBox rect = next();
            if (rect.width <= minW || rect.height <= minH)
            { remove();}
        }
    }

    void RectSelector::rmBig(int maxW,int maxH){
        m_cursor = -1 ;
        while(hasNext()){
            ObjBox rect = next();
            if (rect.width >= maxW || rect.height >= maxH)
            {
                remove();
            }
        }
    }

    void RectSelector::rmBigRatio(int maxW,int maxH,int imgH, float ratio){
        m_cursor = -1 ;
        while(hasNext()){
            ObjBox rect = next();
            int centerY = rect.y+rect.height/2;
            float currRatio = ratio + centerY*1.0/imgH*(1-ratio);
            if (rect.width >= maxW*currRatio || rect.height >= maxH*currRatio)
            {
                remove();
            }
        }
    }

    void RectSelector::rmBigWithRoadMask(cv::Mat& mask,std::vector<float>& roadNumsList,float carLeastWidth){
        m_cursor = -1 ;
        while(hasNext()){
            ObjBox rect = next();
            int centerX,centerY;
            centerX = rect.x+rect.width/2;
            centerY = rect.y+rect.height/2;
            cv::Rect rectRoi=cv::Rect(0, int(centerY-1), mask.size().width, 2);
            cv::Mat roiMask = mask(rectRoi);
            std::vector<vector<cv::Point>> contours;
            std::vector<cv::Vec4i> hierarcy;
            cv::findContours(roiMask, contours, hierarcy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point());
            int index = -1;
            int roadWidth = -1;
            std::vector<cv::Rect> rectMulti;

            for( int i = 0 ; i < contours.size(); i++){
                cv::Rect rectSingle = cv::boundingRect(cv::Mat(contours[i]));
                rectMulti.push_back(rectSingle);
            }

            for( int i = 0 ; i < rectMulti.size(); i++){
                for( int j = i+1 ; j < rectMulti.size(); j++){
                    if(rectMulti[i].x>rectMulti[j].x){
                        cv::Rect tmp = rectMulti[i];
                        rectMulti[i] = rectMulti[j];
                        rectMulti[j] = tmp;
                    }
                }
            }

            for( int i = 0 ; i < rectMulti.size(); i++){
                cv::Rect rectSingle = rectMulti[i];
                if(rectSingle.x<=centerX && centerX<=rectSingle.x+rectSingle.width){
                    index = i;
                    roadWidth = rectSingle.width;
                }
            }
            if(index>=roadNumsList.size() || index == -1){
                continue;
            }
            float carWidth = max(roadWidth/roadNumsList[index],carLeastWidth);
            if (rect.width > carWidth || rect.height > carWidth){
                remove();
            }
        }
    }

    void RectSelector::rmRatio(){
        m_cursor = -1 ;
        while(hasNext()){
            ObjBox rect = next();
            float ratio = rect.width*1.0/rect.height;
            float thres = 4.0;
            if (ratio >= thres || ratio <= 1.0/thres)
            { remove();}
        }
    }

    std::vector<ObjBox> RectSelector::getRect(){
        return m_rects;
    }

}
