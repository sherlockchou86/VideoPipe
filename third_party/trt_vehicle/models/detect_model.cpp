#include "detect_model.h"

namespace trt_vehicle{

    DetectModel::DetectModel(const std::string& modelPath,float scoreThres,float iouThres)
    {
        auto ret  = m_predictor.init(modelPath);
        assert(ret == 0);
        
        setSize();
        m_score = new float[m_batch*m_outputNum*m_classNum];
        m_box = new float[m_batch*m_outputNum*m_boxNum];
        m_scoreCuda = nullptr;
        cudaMalloc(&m_scoreCuda,m_batch*m_outputNum*m_classNum*sizeof(float));
        m_boxCuda = nullptr;
        cudaMalloc(&m_boxCuda,m_batch*m_outputNum*m_boxNum*sizeof(float));
        m_imageDataCuda= nullptr;
        cudaMalloc(&m_imageDataCuda, m_batch*m_inputSizeW*m_inputSizeH*m_inputSizeC*sizeof(float));
        m_scoreThres = scoreThres;
        m_iouThres = iouThres;
    }

    DetectModel::~DetectModel(){
        if(m_score){
            delete [] m_score;
        }
        if(m_box){
            delete [] m_box;
        }
        if(m_scoreCuda){
            cudaFree(m_scoreCuda);
        }
        if(m_boxCuda){
            cudaFree(m_boxCuda);
        }
        if(m_imageDataCuda){
            cudaFree(m_imageDataCuda);
        }
    }

    int DetectModel::setSize(){
        m_predictor.getSizeYolo(m_batch,m_inputSizeC,m_inputSizeH,m_inputSizeW,m_outputNum,m_classNum,m_boxNum);
        return 0;
    }

    int DetectModel::extractFeature(std::vector<cv::Mat> imgs)
    {
        vector<float> imageData;
        for(int i=0;i<imgs.size();i++){
			cv::Mat img ;// .clone();
            cv::cvtColor(imgs[i], img, cv::COLOR_BGR2RGB);
            imagePreprocess(img,imageData);
        }
        cudaMemcpy(m_imageDataCuda, imageData.data(), imageData.size()*sizeof(float), cudaMemcpyHostToDevice);
        vector<void *> buffers = { m_imageDataCuda, m_scoreCuda, m_boxCuda};
        m_predictor.infer(buffers, imgs.size());
        return 0;
    }

    int DetectModel::predict(std::vector<cv::Mat> imgs,std::vector<std::vector<ObjBox>>& outBoxes){
        outBoxes.clear();
        // infer img
        extractFeature(imgs);

        //Transfer data and score from gpu to cpu
        cudaMemcpy(m_score, m_scoreCuda,m_batch*m_outputNum*m_classNum*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(m_box, m_boxCuda,m_batch*m_outputNum*m_boxNum*sizeof(float), cudaMemcpyDeviceToHost);

        //save the boxes which's score more than scoreThres
        for(int b=0;b<m_batch;b++){
            std::vector<ObjBox> boxes;
            std::vector<float> scores,outScores;
            int boxLen = 4;
            for(int i=0;i<m_outputNum;i++){
                int index = -1;
                float scoreMax = -1;
                for(int j=0;j<m_classNum;j++){
                    if(scoreMax<m_score[b*m_outputNum*m_classNum + i*m_classNum+j]){
                        scoreMax = m_score[b*m_outputNum*m_classNum + i*m_classNum+j];
                        index = j;
                    }
                }
                if(scoreMax>m_scoreThres){
                    ObjBox box;
                    int bStart = b*m_outputNum*m_boxNum;
                    xyxy2objBox((m_box[bStart+boxLen*i]-m_wStarts[b])/m_ratioW,(m_box[bStart+boxLen*i+1]-m_hStarts[b])/m_ratioH,(m_box[bStart+boxLen*i+2]-m_wStarts[b])/m_ratioW,(m_box[bStart+boxLen*i+3]-m_hStarts[b])/m_ratioH,box);
                    box.score = scoreMax;
                    box.class_ = index;
                    boxes.push_back(box);
                    scores.push_back(scoreMax);
                }
            }
            std::vector<ObjBox> outBox;
            nonMaximumSuppression(boxes, scores, m_iouThres,outBox);
            outBoxes.push_back(outBox);
        }
        return 0;
    }

    int DetectModel::predictPadding(std::vector<cv::Mat> imgs,std::vector<std::vector<ObjBox>>& outBoxes, int paddingValue){
        std::vector<cv::Mat> imgPaddings;
        m_hStarts.clear();
        m_wStarts.clear();
        for(int i=0;i<imgs.size();i++){
            cv::Mat img = imgs[i];
            if(!img.empty()){
                //padding img
                int h = img.size().height;
                int w = img.size().width;
                float ratio = min(m_inputSizeW*1.0/w,m_inputSizeH*1.0/h);
                m_ratioH = ratio;
                m_ratioW = ratio;
                w = int(w*ratio);
                h = int(h*ratio);
                cv::resize(img,img,cv::Size(w, h),(0.0),(0.0),cv::INTER_LINEAR);
                cv::Mat imgPadding = cv::Mat::zeros(m_inputSizeH,m_inputSizeW,CV_8UC3);
                imgPadding.setTo(paddingValue);
                m_wStarts.push_back(int((m_inputSizeW-w)/2));
                m_hStarts.push_back(((m_inputSizeH-h)/2));
                cv::Rect roi = cv::Rect(m_wStarts[i],m_hStarts[i],w,h);
                img.copyTo(imgPadding(roi));
                imgPaddings.push_back(imgPadding);
            }
        }
        //infer
        if(imgPaddings.size()>0){
            predict(imgPaddings,outBoxes);
        }
        return 0;
    }
}