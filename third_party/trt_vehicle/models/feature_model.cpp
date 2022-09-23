#include "feature_model.h"
#include "mutex"

using namespace std;
namespace trt_vehicle {
    std::mutex mtx_feature;

	FeatureModel::FeatureModel(const std::string& modelPath)
    {
        m_inputType = NetworkInputType::CHW;
        auto ret = m_predictor.init(modelPath);
        assert(ret == 0);
        
        setSize();
        m_score = new float[m_batch*m_outputSizeC];
        m_scoreCuda = nullptr;
        cudaMalloc(&m_scoreCuda,m_batch*m_outputSizeC*sizeof(float));
        m_imageDataCuda= nullptr;
        cudaMalloc(&m_imageDataCuda, m_batch*m_inputSizeW*m_inputSizeH*m_inputSizeC*sizeof(float));
    }

	FeatureModel::~FeatureModel(){
        if(m_score){
            delete [] m_score;
        }
        if(m_scoreCuda){
            cudaFree(m_scoreCuda);
        }
        if(m_imageDataCuda){
            cudaFree(m_imageDataCuda);
        }
    }

    int FeatureModel::setSize(){
        int tmp1,tmp2;
        m_predictor.getSize(m_batch,m_inputSizeC,m_inputSizeH,m_inputSizeW,m_outputSizeC,tmp1,tmp2, m_inputType);
        return 0;
    }

    int FeatureModel::extractFeature(std::vector<cv::Mat> imgs)
    {
        vector<float> imageData;
        for(int i=0;i<imgs.size();i++){
			cv::Mat img = imgs[i];// .clone();
            imagePreprocess(img, imageData);
        }
        cudaMemcpy(m_imageDataCuda, imageData.data(), imageData.size()*sizeof(float), cudaMemcpyHostToDevice);
        vector<void *> buffers = { m_imageDataCuda, m_scoreCuda };
        m_predictor.infer(buffers, m_batch);
        return 0;
    }

    std::vector<std::vector<float>> FeatureModel::predictNoPadding(std::vector<cv::Mat> imgs){
        std::vector<std::vector<float>> scores;

        mtx_feature.lock();
        int ret = 0;
        try{
            ret = extractFeature(imgs);
        }catch(...){
            std::cout<<"errcode : "<<ret<<std::endl;
        }
        mtx_feature.unlock();

        cudaMemcpy(m_score, m_scoreCuda,m_batch*m_outputSizeC*sizeof(float), cudaMemcpyDeviceToHost);
        for(int b=0;b<m_batch;b++){
            std::vector<float> score;
            for(int i=0;i<m_outputSizeC;i++){
                score.push_back(m_score[i]);
            }
            scores.push_back(score);
        }
        return scores;
    }

    std::vector<std::vector<float>> FeatureModel::predictPadding(std::vector<cv::Mat> imgs,int paddingValue){
        std::vector<cv::Mat> imgPaddings;
        for(int i=0;i<imgs.size();i++){
            cv::Mat img = imgs[i];
            int h = img.size().height;
            int w = img.size().width;
            int s = h;
            if(w>h){
                s = w;
            }
            cv::Mat imgPadding = cv::Mat::zeros(s,s,CV_8UC3)+paddingValue;
            // imgPadding.setTo(paddingValue);
            int wStart = int((s-w)/2);
            int hStart = int((s-h)/2);
            cv::Rect roi = cv::Rect(wStart,hStart,w,h);
            img.copyTo(imgPadding(roi));
            imgPaddings.push_back(imgPadding);
        }
        return predictNoPadding(imgPaddings);
    }

}
