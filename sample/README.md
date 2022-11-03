
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
![](../doc/p19.png)

## interaction_with_pipe_sample ##
show how to interact with pipe, such as start/stop channel by calling api.

## record_sample ##
show how `vp_record_node` works.
