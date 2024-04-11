
## important tips when using node ##
Most nodes support multi-channel, which means that multiple channels of data can be used as its input. In addition, some nodes do not support multi-channel by default, that is, there can only be one channel of data as its input (the input channel index cannot be changed once it is determined), otherwise an error will occur. The following are common examples:

*support multi-channel*
- all infer nodes, broker nodes, because they work independently of the channel index.
- split node, it works independently of channel index too. it could be used to split pipeline into multi branches accordding to different channel index.
- some nodes which has been designed to support multi-channel, such as record nodes, track nodes, ba nodes, which are able to distinguish different channels inside logic code (`such as using std::map<int, ...> to maintain different data of channels`).

*do NOT support multi-channel*
- all src node and all des node, you MUST specify channel index when initializing instances of them.
- part of osd nodes, because they work dependently of channel index, you can do some work to let them support multi-channel, for example just use some data structure like `std::map<int, ...>` to maintain different data of channels inside code of nodes.


Note, although some nodes support multi-channel, you'd better be careful because using single instance of node to deal with multiple channels of data would have lower performance(process data serially
). On the contrary, one instance dealing with only one channel of data (multiple channels use multiple instances of the SAME node) have higher performance(process data in parallel). below is an example of 2 methods to create pipeline:
```
single instance of track/ba node works on 2 channels：
file_src_0                                                                                             --> osd_0 --> screen_des_0
           --> detector --> multi-classifiers --> tracker --> ba_crossline --> split(by channel index)
file_src_1                                                                                             --> osd_1 --> screen_des_1 

2 instances of track/ba node work on 2 channels:
file_src_0                                                                --> tracker_0 --> ba_crossline_0 --> osd_0 --> screen_des_0
           --> detector --> multi-classifiers --> split(by channel index) 
file_src_1                                                                --> tracker_0 --> ba_crossline_0 --> osd_1 --> screen_des_1 
```


## 节点目录 ##

<details open>
  <summary>ba</summary>

  - vp_ba_crossline_node：跨线判断
  - vp_ba_jam_node：拥堵判断
  - vp_ba_stop_node：停止判断
</details>

<details open>
  <summary>broker</summary>
  
  - vp_ba_socket_broker_node：使用udp转发行为分析结果
  - vp_embeddings_properties_socket_broker_node：使用udp转发目标特征、属性结果
  - vp_embeddings_socket_broker_node：使用udp转发目标特征结果
  - vp_expr_socket_broker_node：使用udp转发数学表达式检查结果
  - vp_json_console_broker_node：以json格式将结构化数据输出到控制台
  - vp_json_kafka_broker_node：以json格式将结构化数据通过kafka发送给第三方
  - vp_msg_broker_node：数据代理基类节点
  - vp_plate_socket_broker_node：使用udp转发车牌识别结果
  - vp_xml_file_broker_node：以xml格式将结构化数据存储到文件
  - vp_xml_socket_broker_node：以xml格式将结构化数据通过udp发送给第三方
</details>

<details open>
  <summary>infers</summary>
  
  - vp_classifier_node：基于resnet系列的图像分类节点（opencv::dnn）
  - vp_enet_seg_node：基于ENet网络的图像分割节点（opencv::dnn）
  - vp_face_swap_node：基于insightface的人脸替换节点（opencv::dnn）
  - vp_feature_encoder_node：基于resnet系列的目标特征提取节点（opencv::dnn）
  - vp_lane_detector_node：基于CenterNet的车道线检测节点（opencv::dnn）
  - vp_mask_rcnn_detector_node：基于maskrcnn的目标检测节点（opencv::dnn）
  - vp_openpose_detector_node：基于openpose的肢体检测节点（opencv::dnn）
  - vp_ppocr_text_detector_node：基于paddleocr的文字检测节点（paddleinference）
  - vp_restoration_node：基于real-esrgan的图像增强修复节点（opencv::dnn）
  - vp_sface_feature_encoder_node：基于sface网络的人脸特征提取节点（opencv::dnn）
  - vp_trt_vehicle_color_classifier：基于resnet18的车辆颜色分类节点（tensorrt）
  - vp_trt_vehicle_detector：基于yolov5s的车辆检测节点（tensorrt）
  - vp_trt_vehicle_feature_encoder：基于fastreid的车辆特征提取节点（tensorrt）
  - vp_trt_vehicle_plate_detector_v2：基于yolov5s的车牌检测识别节点（一级推理）（tensorrt）
  - vp_trt_vehicle_plate_detector：基于yolov5s的车牌检测识别节点（二级推理）（tensorrt）
  - vp_trt_vehicle_scanner：基于yolov5s的车身扫描节点（tensorrt）
  - vp_trt_vehicle_type_classifier：基于resnet18的车辆车型分类节点（tensorrt）
  - vp_yolo_detector_node：基于yolov3（含tiny）的目标检测节点（opencv::dnn）
  - yolo_yunet_face_detector_node：基于yunet网络的人脸检测节点（opencv::dnn）

</details>

<details open>
  <summary>osd</summary>
  
  - vp_ba_crossline_osd_node：跨线判断结果绘制节点
  - vp_ba_jam_osd_node：拥堵判断结果绘制节点
  - vp_ba_stop_osd_node：停止判断结果绘制节点
  - vp_cluster_node：目标聚类结果绘制节点
  - vp_expr_osd_node：数学表达式检查结果绘制节点
  - vp_face_osd_node_v2：人脸检测结果绘制节点（含相似度显示）
  - vp_face_osd_node：人脸检测结果绘制节点
  - vp_lane_osd_node：车道线检测结果绘制节点
  - vp_osd_node_v2：目标绘制节点（含子目标）
  - vp_osd_node_v3：目标绘制节点（含目标mask）
  - vp_osd_node：目标绘制节点
  - vp_plate_osd_node：车牌检测识别结果绘制节点
  - vp_pose_osd_node：肢体检测结果绘制节点
  - vp_seg_osd_node：图像分割结果绘制节点
  - vp_text_osd_node：文字检测识别结果绘制节点
</details>

<details open>
  <summary>proc</summary>
  
  - vp_expr_check_node：数学等式准确性判断节点
</details>

<details open>
  <summary>record</summary>
  
  - vp_record_node：视频/图片录制节点
</details>

<details open>
  <summary>track</summary>
  
  - vp_dsort_track_node：基于deepsort的跟踪节点
  - vp_sort_track_node：基于sort的跟踪节点
</details>

<details open>
  <summary>common</summary>
  
  - vp_app_des_node：将图片数据推送给application的目标节点
  - vp_app_src_node：从application接收图片数据的原始节点
  - vp_des_node：所有目标节点基类
  - vp_fake_des_node：虚拟目标节点（不做任何事）
  - vp_file_des_node：将视频数据存入文件的目标节点
  - vp_file_src_node：从文件读取视频数据的原始节点
  - vp_image_des_node：将数据以图片的形式发送给socket或者file的目标节点
  - vp_image_src_node：从file或者socket读取图片数据的原始节点
  - vp_infer_node：所有推理节点基类
  - vp_message_broker_node：所有数据代理节点基类
  - vp_node：所有节点基类
  - vp_placeholder_node：虚拟中间节点（不做任何事）
  - vp_primary_infer_node：所有一级推理节点基类
  - vp_rtmp_des_node：将视频数据以rtmp格式推送到rtmp服务器的目标节点
  - vp_rtsp_des_node：将视频数据以rtsp格式推送（无需rtsp服务器）的目标节点
  - vp_rtsp_src_node：以rtsp格式读取网络流的原始节点
  - vp_screen_des_node：将视频/图片显示到屏幕的目标节点
  - vp_secondary_infer_node：所有二级推理节点基类
  - vp_split_node：管道拆分节点
  - vp_src_node：所有原始节点基类
  - vp_sync_node：管道分支同步节点
  - vp_udp_src_node：以udp格式读取网络流的原始节点
</details>