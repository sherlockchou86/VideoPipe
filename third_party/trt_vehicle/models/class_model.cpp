#include "class_model.h"

namespace trt_vehicle{

    ClassModel::ClassModel(const std::string& modelPath, bool isVehicle, std::string meanfile)
				:BaseModel(isVehicle ? NetworkInputType::CHW : NetworkInputType::HWC, meanfile)
    {
		isVehicleModel = isVehicle;
        auto ret  = m_predictor.init(modelPath);
        assert(ret == 0);
        
        setSize();
        m_score = new float[m_batch*m_outputSizeC];
        m_scoreCuda = nullptr;
        cudaMalloc(&m_scoreCuda,m_batch*m_outputSizeC*sizeof(float));
        m_imageDataCuda= nullptr;
        cudaMalloc(&m_imageDataCuda, m_batch*m_inputSizeW*m_inputSizeH*m_inputSizeC*sizeof(float));
    }

    ClassModel::~ClassModel(){
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

    int ClassModel::setSize(){
        int tmp1,tmp2;
		if(isVehicleModel)
			m_predictor.getSizeVehicle(m_batch, m_inputSizeC, m_inputSizeH, m_inputSizeW, m_outputSizeC, tmp1, tmp2);
		else
			m_predictor.getSize(m_batch,m_inputSizeC,m_inputSizeH,m_inputSizeW,m_outputSizeC,tmp1,tmp2);
        return 0;
    }

    int ClassModel::extractFeature(std::vector<cv::Mat> imgs)
    {
        vector<float> imageData;
        for(int i=0;i<imgs.size();i++){
			cv::Mat img;// .clone();
			if (!isVehicleModel)
			{
				cv::cvtColor(imgs[i], img, cv::COLOR_BGR2RGB);
			}
			else
			{
				img = imgs[i];
			}
            imagePreprocess(img,imageData);
        }
        cudaMemcpy(m_imageDataCuda, imageData.data(), imageData.size()*sizeof(float), cudaMemcpyHostToDevice);
        vector<void *> buffers = { m_imageDataCuda, m_scoreCuda };
        m_predictor.infer(buffers, imgs.size());
        return 0;
    }

    std::vector<std::vector<float>> ClassModel::predictNoPadding(std::vector<cv::Mat> imgs){
        std::vector<std::vector<float>> scores;
        extractFeature(imgs);
        cudaMemcpy(m_score, m_scoreCuda,m_batch*m_outputSizeC*sizeof(float), cudaMemcpyDeviceToHost);

        for(int i=0;i< imgs.size(); i++) {
            float sumScores = 0.0;
            float minScore = 999999999.0;
            for(int j=0;j<m_outputSizeC;j++){
                if(minScore>m_score[i*m_outputSizeC+j])
                    minScore = m_score[i*m_outputSizeC+j];
            }
            if(minScore<0){
                for(int j=0;j<m_outputSizeC;j++){
                        m_score[i*m_outputSizeC+j] = m_score[i*m_outputSizeC+j]-minScore+0.01;
                }
            }
            for(int j=0;j<m_outputSizeC;j++){
                sumScores += m_score[i*m_outputSizeC+j];
            }
            std::vector<float> score;
            for(int j=0;j<m_outputSizeC;j++){
                score.push_back(m_score[i*m_outputSizeC+j]/sumScores);
            }
            scores.push_back(score);
        }

        return scores;
    }

    std::vector<std::vector<float>> ClassModel::predictPadding(std::vector<cv::Mat> imgs,int paddingValue){
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