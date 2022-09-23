#pragma once

#include <mutex>
#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <iostream>
#include <fstream>
#include <cuda/cuda_runtime.h>
#include <cuda/cuda_runtime_api.h>
#include <tensorrt/NvInfer.h>
#include <tensorrt/NvOnnxConfig.h>
#include <tensorrt/NvInferPlugin.h>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <sys/time.h>
#include <cuda/device_launch_parameters.h>

#include "../util/algorithm_util.h"
#include "logging.h"

using namespace nvinfer1;
using namespace nvonnxparser;
using namespace trt_vehicle;

//extern "C" void calGpuDistance(float** f1,float** f2,float** dis,int outN,int outC);

namespace trt_vehicle{

#define CheckCudaError(status) \
{ \
    if (status != 0) \
    { \
        return; \
    } \
}
	enum class NetworkInputType
	{
		HWC,
		CHW
	};

class CudaPredictor{
public:
    ~CudaPredictor();
public:
    // create engine from engine path
    int init(const string path,int deviceID = 0, bool verbose=false);

    //get input/output sizes of model
    int getSizeYolo(int& batch,int& inputSizeC,int& inputSizeH,int& inputSizeW,int& outputNum,int& classNum,int& boxNum); 

    //get input/output sizes of model
    int getSize(int& batch,int& inputSizeC,int& inputSizeH,int& inputSizeW,int& outputDim1,int& outputDim2,int& outputDim3, NetworkInputType networkInputType = NetworkInputType::HWC);

	// TODO 车辆分类模型
    int getSizeVehicle(int& batch,int& inputSizeC,int& inputSizeH,int& inputSizeW,int& outputDim1,int& outputDim2,int& outputDim3);
	
    //forward of neural network
    int infer(vector<void *> &buffers, int batch=1);

private:
    int loadModel(const string &path);
    int prepare();
private:
    nvinfer1::IRuntime* m_runtime = nullptr;
    nvinfer1::ICudaEngine* m_engine = nullptr;
    nvinfer1::IExecutionContext* m_context = nullptr;
    cudaStream_t m_stream = nullptr;
};

class BaseModel{
    public:
		BaseModel(NetworkInputType inputType = NetworkInputType::HWC, std::string meanFile = "")
				{ m_inputType = inputType; m_MeanFile = meanFile; }

        virtual ~BaseModel();

        //input dimensions of the model 
        int getInputSize(int& inputW,int& inputH);

        //image preprocessimg
        int imagePreprocess(cv::Mat& img,vector<float>& imageData);

		bool readMeanImageFile();
    protected:
        //image resize
        void imageResize(const cv::Mat &src,cv::Size size,std::vector<float>& data);

    protected:
        int m_inputSizeH;
        int m_inputSizeW;
        int m_inputSizeC;
        int m_batch;
        int m_downScale;

		NetworkInputType m_inputType;
		std::string m_MeanFile;
		float* m_meanFileData = nullptr;
    };
}