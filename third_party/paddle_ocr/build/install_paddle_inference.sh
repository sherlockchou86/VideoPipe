

PADDLE_INFERENCE_PATH=/usr/local/paddle_inference


cp -r $PADDLE_INFERENCE_PATH/paddle/lib/* /usr/local/lib
cp -r $PADDLE_INFERENCE_PATH/paddle/include/* /usr/local/include

cp -r $PADDLE_INFERENCE_PATH/third_party/install/cryptopp/lib/* /usr/local/lib
cp -r $PADDLE_INFERENCE_PATH/third_party/install/cryptopp/include/* /usr/local/include

cp -r $PADDLE_INFERENCE_PATH/third_party/install/gflags/lib/* /usr/local/lib
cp -r $PADDLE_INFERENCE_PATH/third_party/install/gflags/include/* /usr/local/include

cp -r $PADDLE_INFERENCE_PATH/third_party/install/glog/lib/* /usr/local/lib
cp -r $PADDLE_INFERENCE_PATH/third_party/install/glog/include/* /usr/local/include

cp -r $PADDLE_INFERENCE_PATH/third_party/install/mkldnn/lib/* /usr/local/lib
cp -r $PADDLE_INFERENCE_PATH/third_party/install/mkldnn/include/* /usr/local/include

cp -r $PADDLE_INFERENCE_PATH/third_party/install/mklml/lib/* /usr/local/lib
cp -r $PADDLE_INFERENCE_PATH/third_party/install/mklml/include/* /usr/local/include

cp -r $PADDLE_INFERENCE_PATH/third_party/install/onnxruntime/lib/* /usr/local/lib
cp -r $PADDLE_INFERENCE_PATH/third_party/install/onnxruntime/include/* /usr/local/include

cp -r $PADDLE_INFERENCE_PATH/third_party/install/paddle2onnx/lib/* /usr/local/lib
cp -r $PADDLE_INFERENCE_PATH/third_party/install/paddle2onnx/include/* /usr/local/include

cp -r $PADDLE_INFERENCE_PATH/third_party/install/protobuf/lib/* /usr/local/lib
cp -r $PADDLE_INFERENCE_PATH/third_party/install/protobuf/include/* /usr/local/include

cp -r $PADDLE_INFERENCE_PATH/third_party/install/utf8proc/lib/* /usr/local/lib
cp -r $PADDLE_INFERENCE_PATH/third_party/install/utf8proc/include/* /usr/local/include

cp -r $PADDLE_INFERENCE_PATH/third_party/install/xxhash/lib/* /usr/local/lib
cp -r $PADDLE_INFERENCE_PATH/third_party/install/xxhash/include/* /usr/local/include

cp -r $PADDLE_INFERENCE_PATH/third_party/threadpool/* /usr/local/include