

## VideoPipe
A framework for video structured. It could handle complex tasks such as stream reading(from local or network), video decoding, inference based on deeplearning models, osd(on screen display), message broker via middleware(like kafka), video encoding and stream pushing(rtmp or local file). It's Plugin-Oriented coding style, we can construct different types of pipeline using independent plugins namely `Node` in framework. 

VideoPipe works like DeepStream from Nvidia and MindX SDK from Huawei, but it is more simple to use, more portable and has few dependency on third-party modules such as gstreamer which is hard to learn(coding style or debug). The framework is written purely by native C++ STL, and depends on popular modules like OpenCV, so the code is more portable for different platforms.

The framework can be used in such situations:
1. Video Structure
2. Image Search
3. Face Recognition
4. Behaviour Analyse based on Video (Security and Safety)

> NOTE: <br/>
> VideoPipe is a framework aimed to make model-integration more simple in CV field, it is not a deeplearning related frameworks such as tensorflow, tensorrt.

## Key Features
- `Stream Reading`. Support popular protocals such as udp, rtsp, rtmp, file.
- `Video Decoding`. Support video decoding which is based on opencv/ffmpeg.
- `Inference based on dl`. Support multi-level inference based on deep learning models, such as Object-Detection, Image-Classification, Feature-Extraction. What you need is preparing models and know how to parse its outputs. Inference can be implemented based on different backends such as opencv::dnn(default), tensorrt, paddle_inference, onnx runtime.
- `On Screen Display(OSD)`. Support visualization, like drawing outputs from model onto frame.
- `Message Broker[not implemented yet]`. Support push structured data(via json) to cloud or other platforms.
- `Object Tracking[not implemented yet]`. Support object tracking such as iou, sort etc.
- `Behaviour Analyse[not implemented yet]`. Support behaviour analyse based on tracking.
- `Recording[not implemented yet]`. Support video recording for specific period, screenshots for specific frame.
- `Video Encoding`. Support video encoding which is based on opencv/ffmpeg.
- `Stream Pushing`. Support stream pushing via rtmp, rtsp, file.

## Highlights

1. Visualization for pipelines, which is useful when debugging. The running status of pipeline refresh automatically on screen, including fps, cache size, latency at each link in pipeline, We can figure out quickly where the bottleneck is based on these running information.
2. Data transfered between 2 nodes by smart pointer which is shallow-copy by default, no content copy operations needed when data flowing through the whole pipeline. But, we can specify deep-copy if we need, for example, when the pipeline has multi branches and we need operate on 2 different contents separately.
3. We can construct different types of pipeline, only 1 channel in a pipeline or multi channels in a piepline are both supported, channels in pipeline are independent. 
4. The pipeline support hooks, we can register callbacks to the pipeline to get the status notification(see the 1st item), such as fps.
5. Many node classes are already built-in in VideoPipe, but all nodes in framework can be re-implemented by yourself and also you can implement more based on your requirements.
6. The whole framework is written mainly by native C++ which is portable to all paltforms. 

## Project structure


## Dependency

Basicis
- ubuntu 18.04
- vscode (remote development on windows)
- c++ 17
- opencv 4.6
- ffmpeg 3.4 (required by opencv)
- gstreamer 1.20 (required by opencv)
- gcc 7.5

Optional, if you need implement(or use built-in) infer nodes based on other inference backends other than opencv::dnn.
- cuda, tensorrt
- paddle inference
- onnx runtime
- anything you like


## How to build and debug

- Build VideoPipe (via shell)
    - run `cd build/`
    - run `sh build.sh`
    - it will generate a library called libvp.so and copy it to /usr/local/lib automatically.


- Debug VideoPipe (via vscode)
    - select the cpp file you want to debug (keep it activated), like `./main/main.cpp`
    - press `run` button at debug menu in vscode
    - select a launch item at the top of window (something like `C/C++: g++ vp project`)


> All sub projects in `./third_party/` are independent projects and can be built and debug like above, please refer to readme.md in sub folder.

## How to use 

- build VideoPipe first and use shared library.
- referencing source code directly and build you whole application.

## how to contribute
The project is under development currently, any PRs would be appreciated.

note, the code, architecture may be not stable (2022/9/29)

## Compared to other similar sdk

VideoPipe is opensource totally and more portable for different soft/hard-ware platforms. DeepStream/MindX are platform-depended, maybe they can get better performance for some modules like decoding, inference, osd.
