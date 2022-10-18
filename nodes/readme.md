
## common node list [`ctrl+f` to search] ##
-------

|  id  |  node                    | usage        | status | sample code  |
|  --  |  ----                    | -----------  | ----   | -----------  |
|  1   |  vp_ba_node              | ++++         | not implemented yet | on the way       |
|  2   |  vp_des_node             | base class for DES nodes | tested |               |
|  3   |  vp_fake_des_node        | do nothing just keep pipe complete | tested |               |
|  4   |  vp_file_des_node        | writting video to file | tested |               |
|  5   |  vp_file_src_node        | reading video from file | tested |               |
|  6   |  vp_infer_node           | base class for all infer nodes | tested |               |
|  7   |  vp_message_broker_node  | ++++ | not implemented yet |               |
|  8   |  vp_node                 | base class for all nodes | tested |               |
|  9   |  vp_primary_infer_node   | base class for all primary infer nodes, which infer on whole frame | tested |               |
|  10  |  vp_rtmp_des_node        | pushing video to server via rtmp | tested |               |
|  11  |  vp_rtsp_src_node        | reading video from rtsp server | tested |               |
|  12  |  vp_screen_des_node      | displaying video on screen | tested |               |
|  13  |  vp_secondary_infer_node | base class for all secondary infer nodes, which infer on small cropped image | tested |               |
|  14  |  vp_skip_node            | skip node which setting skipping rule for infer | not implemented yet |               |
|  15  |  vp_split_node           | split pipe into multi branches, support by channel and deep-copy | tested |               |
|  16  |  vp_src_node             | base class for all SRC nodes | tested |               |
|  17  |  vp_track_node           | ++++ | not implemented yet |
|  18  |  vp_udp_src_node         | reading video from network via udp/rtp | tested |               |
|  19  |  vp_record_node          | video/image recording node | on the way |               |
|  *   |  `more`                  | wait for your contribution | on the way |               |

## infers node list [with some customization] ##
-------
> `with some customization` means the node should be applied with other special nodes or depend on specific situation.

|  id  |  node                           | usage        | status | sample code  |
|  --  |  ----                           | -----------  | ----   | -----------  |
|  1   |  vp_classifier_node             | image classifier based opencv:dnn | tested         | on the way       |
|  2   |  vp_feature_encoder_node        | image feature encoder based opencv::dnn | on the way         |               |
|  3   |  vp_openpose_detector_node      | pose detector based on opencv::dnn(OpenPose) | tested         |               |
|  4   |  vp_ppocr_text_detector_node    | ocr based on paddlepaddle(paddle_ocr from badidu) | tested         |               |
|  5   |  vp_sface_feature_encoder_node  | face feature encoder based on opencv::dnn(sface) | tested         |               |
|  6   |  vp_trt_vehicle_detector        | vehicle detector based on tensorrt(yolov5s) | tested         |               |
|  7   |  vp_trt_vehicle_plate_detector  | vehicle plate detector based on tensorrt(yolov5s) | tested         |               |
|  8   |  vp_yolo_detector_node          | object detector based on opencv::dnn(yolo series) | tested         |               |
|  9   |  vp_yunet_face_detector_node    | face detector based on opencv::dnn(yunet) | tested         |               |
|  *   |  `more`                         | wait for your contribution | on the way         |               |


## osd node list [with some customization] ##
-------

|  id  |  node                    | usage        | status | sample code  |
|  --  |  ----                    | -----------  | ----   | -----------  |
|  1   |  vp_face_osd_node_v2     | face target display, including similarity at the bottom of frame | tested         | on the way       |
|  2   |  vp_face_osd_node        | face target display | tested         |               |
|  3   |  vp_osd_node_v2          | target display, including sub target at the bottom of frame | tested         |               |
|  4   |  vp_osd_node             | target display | tested         |               |
|  5   |  vp_pose_osd_node        | pose target display | tested         |               |
|  6   |  vp_text_osd_node        | text target display | tested         |               |
|  *   |  `more`                  | wait for your contribution | on the way         |               |