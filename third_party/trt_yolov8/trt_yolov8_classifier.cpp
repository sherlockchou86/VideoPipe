
#include "trt_yolov8_classifier.h"

namespace trt_yolov8 {
    using namespace nvinfer1;
    void trt_yolov8_classifier::batch_preprocess(std::vector<cv::Mat>& imgs, float* output, int dst_width, int dst_height) {
        for (size_t b = 0; b < imgs.size(); b++) {
        int h = imgs[b].rows;
        int w = imgs[b].cols;
        int m = std::min(h, w);
        int top = (h - m) / 2;
        int left = (w - m) / 2;
        cv::Mat img = imgs[b](cv::Rect(left, top, m, m));
        cv::resize(img, img, cv::Size(dst_width, dst_height), 0, 0, cv::INTER_LINEAR);
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        img.convertTo(img, CV_32F, 1/255.0);

        std::vector<cv::Mat> channels(3);
        cv::split(img, channels);

        // CHW format
        for (int c = 0; c < 3; ++c) {
        int i = 0;
        for (int row = 0; row < dst_height; ++row) {
            for (int col = 0; col < dst_width; ++col) {
            output[b * 3 * dst_height * dst_width + c * dst_height * dst_width + i] =
                channels[c].at<float>(row, col);
            ++i;
            }
        }
        }
        }
    }

    std::vector<float> trt_yolov8_classifier::softmax(float *prob, int n) {
        std::vector<float> res;
        float sum = 0.0f;
        float t;
        for (int i = 0; i < n; i++) {
            t = expf(prob[i]);
            res.push_back(t);
            sum += t;
        }
        for (int i = 0; i < n; i++) {
            res[i] /= sum;
        }
        return res;
    }

    std::vector<int> trt_yolov8_classifier::topk(const std::vector<float>& vec, int k) {
        std::vector<int> topk_index;
        std::vector<size_t> vec_index(vec.size());
        std::iota(vec_index.begin(), vec_index.end(), 0);

        std::sort(vec_index.begin(), vec_index.end(), [&vec](size_t index_1, size_t index_2) { return vec[index_1] > vec[index_2]; });

        int k_num = std::min<int>(vec.size(), k);

        for (int i = 0; i < k_num; ++i) {
            topk_index.push_back(vec_index[i]);
        }

        return topk_index;
    }

    void trt_yolov8_classifier::prepare_buffers(ICudaEngine* engine, float** gpu_input_buffer, float** gpu_output_buffer, float** cpu_input_buffer, float** output_buffer_host) {
        assert(engine->getNbBindings() == 2);
        // In order to bind the buffers, we need to know the names of the input and output tensors.
        // Note that indices are guaranteed to be less than IEngine::getNbBindings()
        const int inputIndex = engine->getBindingIndex(kInputTensorName);
        const int outputIndex = engine->getBindingIndex(kOutputTensorName);
        assert(inputIndex == 0);
        assert(outputIndex == 1);
        // Create GPU buffers on device
        CUDA_CHECK(cudaMalloc((void**)gpu_input_buffer, kBatchSize * 3 * kClsInputH * kClsInputW * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)gpu_output_buffer, kBatchSize * kOutputSize * sizeof(float)));

        *cpu_input_buffer = new float[kBatchSize * 3 * kClsInputH * kClsInputW];
        *output_buffer_host = new float[kBatchSize * kOutputSize];
    }

    void trt_yolov8_classifier::infer(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* input, float* output, int batchSize) {
        CUDA_CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * 3 * kClsInputH * kClsInputW * sizeof(float), cudaMemcpyHostToDevice, stream));
        context.enqueue(batchSize, buffers, stream, nullptr);
        CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * kOutputSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);
    }

    void trt_yolov8_classifier::serialize_engine(unsigned int max_batchsize, float& gd, float& gw, std::string& wts_name, std::string& engine_name) {
        // Create builder
        IBuilder* builder = createInferBuilder(gLogger);
        IBuilderConfig* config = builder->createBuilderConfig();
        // Create model to populate the network, then set the outputs and create an engine
        IHostMemory *serialized_engine = nullptr;
        //engine = buildEngineYolov8Cls(max_batchsize, builder, config, DataType::kFLOAT, gd, gw, wts_name);
        serialized_engine = buildEngineYolov8Cls(builder, config, DataType::kFLOAT, wts_name, gd, gw);
        assert(serialized_engine);
        // Save engine to file
        std::ofstream p(engine_name, std::ios::binary);
        if (!p) {
            std::cerr << "Could not open plan output file" << std::endl;
            assert(false);
        }
        p.write(reinterpret_cast<const char*>(serialized_engine->data()), serialized_engine->size());

        // Close everything down
        delete serialized_engine;
        delete config;
        delete builder;
    }

    void trt_yolov8_classifier::deserialize_engine(std::string& engine_name, IRuntime** runtime, ICudaEngine** engine, IExecutionContext** context) {
        std::ifstream file(engine_name, std::ios::binary);
        if (!file.good()) {
            std::cerr << "read " << engine_name << " error!" << std::endl;
            assert(false);
        }
        size_t size = 0;
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        char* serialized_engine = new char[size];
        assert(serialized_engine);
        file.read(serialized_engine, size);
        file.close();

        *runtime = createInferRuntime(gLogger);
        assert(*runtime);
        *engine = (*runtime)->deserializeCudaEngine(serialized_engine, size);
        assert(*engine);
        *context = (*engine)->createExecutionContext();
        assert(*context);
        delete[] serialized_engine;
    }

    trt_yolov8_classifier::trt_yolov8_classifier(std::string model_path) {
        if (model_path.empty()) {
            return;
        }

        cudaSetDevice(kGpuId);
        deserialize_engine(model_path, &runtime, &engine, &context);
        CUDA_CHECK(cudaStreamCreate(&stream));
    }

    trt_yolov8_classifier::~trt_yolov8_classifier() {
        cudaStreamDestroy(stream);
        delete context;
        delete engine;
        delete runtime;
    }

    void trt_yolov8_classifier::classify(std::vector<cv::Mat> images, std::vector<std::vector<Classification>>& classifications, int top_k) {
        // Prepare cpu and gpu buffers
        float* device_buffers[2];
        float* cpu_input_buffer = nullptr;
        float* output_buffer_host = nullptr;
        prepare_buffers(engine, &device_buffers[0], &device_buffers[1], &cpu_input_buffer, &output_buffer_host);

        // batch predict
        for (size_t i = 0; i < images.size(); i += kBatchSize) {
            // Get a batch of images
            std::vector<cv::Mat> img_batch;
            for (size_t j = i; j < i + kBatchSize && j < images.size(); j++) {
                img_batch.push_back(images[j]);
            }

            // Preprocess
            batch_preprocess(img_batch, cpu_input_buffer);

            // Run inference
            auto start = std::chrono::system_clock::now();
            infer(*context, stream, (void**)device_buffers, cpu_input_buffer, output_buffer_host, kBatchSize);
            auto end = std::chrono::system_clock::now();
            //std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

            // Postprocess and get top-k result
            for (size_t b = 0; b < img_batch.size(); b++) {
                float* p = &output_buffer_host[b * kOutputSize];
                auto res = softmax(p, kOutputSize);
                auto topk_idx = topk(res, top_k);

                std::vector<Classification> classification;
                for (auto idx: topk_idx) {
                    classification.push_back(Classification {idx, res[idx]});
                }

                classifications.push_back(classification);
            }
        }

        CUDA_CHECK(cudaFree(device_buffers[0]));
        CUDA_CHECK(cudaFree(device_buffers[1]));
        delete[] cpu_input_buffer;
        delete[] output_buffer_host;      
    }

    bool trt_yolov8_classifier::wts_2_engine(std::string wts_name, std::string engine_name, std::string sub_type) {
        int is_p = 0;
        float gd = 0.0f, gw = 0.0f;
        int max_channels = 0;

        if (sub_type[0] == 'n') {          // yolov8n
            gd = 0.33;
            gw = 0.25;
            max_channels = 1024;
        } else if (sub_type[0] == 's') {   // yolov8s
            gd = 0.33;
            gw = 0.50;
            max_channels = 1024;
        } else if (sub_type[0] == 'm') {   // yolov8m
            gd = 0.67;
            gw = 0.75;
            max_channels = 576;
        } else if (sub_type[0] == 'l') {   // yolov8l
            gd = 1.0;
            gw = 1.0;
            max_channels = 512;
        } else if (sub_type[0] == 'x') {   // yolov8x
            gd = 1.0;
            gw = 1.25;
            max_channels = 640;
        } else {
            return false;  // not support
        }
        
        if (sub_type.size() == 2 && sub_type[1] == '6') {         // yolov8n6/yolov8s6/yolov8m6/yolov8l6/yolov8x6
            is_p = 6;
        } else if (sub_type.size() == 2 && sub_type[1] == '2') {  // yolov8n2/yolov8s2/yolov8m2/yolov8l2/yolov8x2
            is_p = 2;
        }

        serialize_engine(kBatchSize, gd, gw, wts_name, engine_name);
        return true;
    }
}