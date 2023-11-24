
## 1-1-1_sample ##
1 video input, 1 infer task, and 1 output.
![](../doc/p10.png)

## 1-1-N_sample ##
1 video input, 1 infer task, and 2 outputs.
![](../doc/p11.png)


## 1-N-N_sample ##
1 video input and then split into 2 branches for different infer tasks, then 2 total outputs.
![](../doc/p12.png)


## N-1-N_sample ##
2 video input and merge into 1 branch automatically for 1 infer task, then resume to 2 branches for outputs again.
![](../doc/p13.png)


## N-N_sample ##
multi pipe exist separately and each pipe is 1-1-1 (can be any structure like 1-1-N, 1-N-N)
![](../doc/p14.png)


## paddle_infer_sample ##
ocr based on paddle (install paddle_inference first!), 1 video input and 2 outputs (screen, rtmp)
![](../doc/p15.png)


## src_des_sample ##
show how src nodes and des nodes work.
3 (file, rtsp, udp) input and merge into 1 infer task, then resume to 3 branches for outputs (screen, rtmp, fake)
![](../doc/p16.png)


## trt_infer_sample ##
vehicle and plate detector based on tensorrt (install tensorrt first!), 1 video input and 3 outputs (screen, file, rtmp)
![](../doc/p17.png)


## vp_logger_sample ##
show how `vp_logger` works.

## face_tracking_sample ##
tracking for multi faces.
![](../doc/p18.png)

## vehicle_tracking_sample ##
tracking for multi vehicles.
![](../doc/p22.png)

## interaction_with_pipe_sample ##
show how to interact with pipe, such as start/stop channel by calling api.

## record_sample ##
show how `vp_record_node` works.

## message_broker_sample & message_broker_sample2 ##
show how message broker nodes work.
![](../doc/p20.png)
![](../doc/p21.png)

## mask_rcnn_sample ##
show image segmentation by mask-rcnn.
![](../doc/p30.png)

## openpose_sample ##
show pose estimation by openpose network.
![](../doc/p31.png)

## enet_seg_sample ##
show semantic segmentation by enet network.
![](../doc/p32.png)

## multi_detectors_and_classifiers_sample ##
show multi infer node work together.
![](../doc/p33.png)

## image_des_sample ##
show save/push image to local file or remote via udp.
![](../doc/p34.png)

## image_src_sample ##
show read/receive image from local file or remote via udp.
![](../doc/p35.png)

## rtsp_des_sample ##
show push video stream via rtsp, no rtsp server needed, you can visit it directly.
![](../doc/p36.png)

## ba_crossline_sample ##
count for vehicle based on tracking, the simplest one of behaviour analysis.
![](../doc/p37.png)

## plate_recognize_sample ##
vehicle plate detect and recognize on the whole frame (no need to detect vechile first)
![](../doc/p38.png)