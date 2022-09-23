#include "base_model.h"

namespace trt_vehicle{

    std::mutex mtx;
    CudaPredictor::~CudaPredictor(){
        if (m_stream) cudaStreamDestroy(m_stream);
        mtx.lock();
        if (m_context) m_context->destroy();
        if (m_engine) m_engine->destroy();
		if (m_runtime) m_runtime->destroy();
        mtx.unlock();
    }

    int CudaPredictor::loadModel(const string &path){
        ifstream file(path, ios::in | ios::binary);
        if (!file)
        {
            return -3;
        }
        std::vector<char> trtModelStream;
        size_t size{0};
        if (file.good())
        {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream.resize(size);
            file.read(trtModelStream.data(), size);
            file.close();
        }
        try {
            m_engine = m_runtime->deserializeCudaEngine(trtModelStream.data(), size, nullptr);
            if(m_engine == nullptr){
                return -5;
            }
        }
        catch(...){
            return -6;
        }
        return 0;
    }

    int CudaPredictor::prepare() {
        m_context = m_engine->createExecutionContext();
        if (!m_context)
        {
            return -7;
        }
        cudaError_t ret = cudaStreamCreate(&m_stream);
        if(ret != cudaSuccess){
            return -8;
        }
        return 0;
    }

    int CudaPredictor::init(const std::string path,int deviceID , bool verbose) {
        int countGpus=0;
        cudaGetDeviceCount(&countGpus);
        if(deviceID > countGpus -1){
            return -1;
        }
        mtx.lock();
        cudaSetDevice(deviceID);
        static sample::Logger gLogger{sample::Logger::Severity::kINFO};
        m_runtime = createInferRuntime(gLogger);
        if(m_runtime == nullptr){
            mtx.unlock();
            return -2;
        }
        int ret = loadModel(path);
        if(ret != 0){
            mtx.unlock();
            return ret;
        }
        ret = prepare();
        mtx.unlock();
        return ret;
    }

    int CudaPredictor::getSizeYolo(int& batch,int& inputSizeC,int& inputSizeH,int& inputSizeW,int& outputNum,int& classNum,int& boxNum){
        auto dims0 = m_engine->getBindingDimensions(0);
        batch = dims0.d[0];
        inputSizeH = dims0.d[1];
        inputSizeW = dims0.d[2];
        inputSizeC = dims0.d[3];
        auto dims1 = m_engine->getBindingDimensions(1);
        auto dims2 = m_engine->getBindingDimensions(2);
        outputNum = dims1.d[1];
        classNum = dims1.d[2];
        boxNum = dims2.d[3];
        return 0;
    }

    int CudaPredictor::getSize(int& batch,int& inputSizeC,int& inputSizeH,int& inputSizeW,int& outputDim1,int& outputDim2,int& outputDim3, NetworkInputType networkInputType){
        auto dims0 = m_engine->getBindingDimensions(0);
        batch = dims0.d[0];
        if (networkInputType == NetworkInputType::CHW)
        {
            inputSizeC = dims0.d[1];
            inputSizeH = dims0.d[2];
            inputSizeW = dims0.d[3];
        }
        else
        {
            inputSizeH = dims0.d[1];
            inputSizeW = dims0.d[2];
            inputSizeC = dims0.d[3];
        }
        auto dims1 = m_engine->getBindingDimensions(1);
        outputDim1 = dims1.d[1];
        outputDim2 = dims1.d[2];
        outputDim3 = dims1.d[3];
        if(outputDim2 < 1){
            outputDim2 = 1;
        }
        if(outputDim3 < 1){
            outputDim3 = 1;
        }
        return 0;
    }

	int CudaPredictor::getSizeVehicle(int& batch, int& inputSizeC, int& inputSizeH, int& inputSizeW, int& outputDim1, int& outputDim2, int& outputDim3) {
		auto dims0 = m_engine->getBindingDimensions(0);
		batch = 1; // dims0.d[0];	//TODO 
		inputSizeH = dims0.d[1];
		inputSizeW = dims0.d[2];
		inputSizeC = dims0.d[0];
		auto dims1 = m_engine->getBindingDimensions(1);
		outputDim1 = dims1.d[0];
		outputDim2 = dims1.d[1];
		outputDim3 = dims1.d[2];
		if (outputDim2 < 1) {
			outputDim2 = 1;
		}
		if (outputDim3 < 1) {
			outputDim3 = 1;
		}
		return 0;
	}

    int CudaPredictor::infer(vector<void *> &buffers, int batch){
        bool ok = m_context->execute(batch, buffers.data());
        if(ok == false){
            return -1;
        }
        cudaError_t ret  = cudaStreamSynchronize(m_stream);
        if(ret != cudaSuccess){
            return -2;
        }
        return 0;
    }

    int BaseModel::getInputSize(int& inputW,int& inputH){
        inputH = m_inputSizeH;
        inputW = m_inputSizeW;
        return 0;
    }

    void BaseModel::imageResize(const cv::Mat &src, cv::Size size,std::vector<float>& data)
	{
		cv::Mat dst;
		cv::resize(src, dst, size, (0.0), (0.0), cv::INTER_LINEAR);
		int h = size.height;
		int w = size.width;

		if (m_inputType == NetworkInputType::CHW)
		{
			this->readMeanImageFile();
			data.resize(w * h * 3);
			std::vector<float> dataTmp = (std::vector<float>)(dst.reshape(1, 1));

			for (int row=0; row < h; row++)
			{
				for (int col = 0; col < w; col++)
				{
					for (int k = 0; k < 3; k++)
					{
						int dstIndex = w * h * k + row * w + col;
						int srcIndex = row * w * 3 + col * 3 + k;
						data[dstIndex] = (float)dataTmp[srcIndex];
						if(m_meanFileData)
							data[dstIndex] -= m_meanFileData[srcIndex];
					}
				}
			}
		}
		else
		{
			data = (std::vector<float>)(dst.reshape(1, 1));
		}
    }

	bool BaseModel::readMeanImageFile()
	{
		if (m_meanFileData != nullptr || m_MeanFile.empty())
			return false;

		std::ifstream infile(m_MeanFile, std::ifstream::binary);
		size_t size = m_inputSizeH * m_inputSizeW * m_inputSizeC;
		uint8_t tempMeanDataChar[size];

		if (!infile.good())
		{
			return false;
		}

		std::string magic, max;
		unsigned int h, w;
		infile >> magic >> w >> h >> max;

		if (magic != "P3" && magic != "P6")
		{
			return false;
		}

		if (w != m_inputSizeW || h != m_inputSizeH)
		{
			std:; cerr << "Mismatch between ppm mean image resolution and network resolution " << std::endl;
			return false;
		}

		infile.get();
		infile.read((char*)tempMeanDataChar, size);
		if (infile.gcount() != (int)size || infile.fail())
		{
			return false;
		}

		m_meanFileData = new float[size];
		for (size_t i = 0; i < size; i++)
		{
			m_meanFileData[i] = (float)tempMeanDataChar[i];
		}
		return true;
	}

    int BaseModel::imagePreprocess(cv::Mat& img,vector<float>& imageData)
    {
        cv::Mat src;
        if(img.channels() == 1){
            cv::cvtColor(img,src,cv::COLOR_GRAY2RGB);
        }
		else{
            //src = img.clone();
			src = img;
        }
		cv::Mat input;
        src.convertTo(input, CV_32FC3);
        vector<float> data;
        imageResize(input, cv::Size(m_inputSizeW, m_inputSizeH),data);
        imageData.insert(imageData.end(), data.begin(), data.end());
        return 0;
    }

    BaseModel::~BaseModel(){
		if (m_meanFileData != nullptr)
			delete[] m_meanFileData;
    }
    
}