# can ONLY be called by parent CMakeLists.txt, because no environment settings for this script.#
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    message(FATAL_ERROR "can ONLY be called by parent CMakeLists.txt.")
endif()

message("start build for simple samples...")
# save all exe to 'build/bin'
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
add_executable(1-1-1_sample "1-1-1_sample.cpp")
target_link_libraries(1-1-1_sample ${PROJECT_NAME})

add_executable(1-1-N_sample "1-1-N_sample.cpp")
target_link_libraries(1-1-N_sample ${PROJECT_NAME})

add_executable(1-N-N_sample "1-N-N_sample.cpp")
target_link_libraries(1-N-N_sample ${PROJECT_NAME})

add_executable(ba_crossline_sample "ba_crossline_sample.cpp")
target_link_libraries(ba_crossline_sample ${PROJECT_NAME})

add_executable(enet_seg_sample "enet_seg_sample.cpp")
target_link_libraries(enet_seg_sample ${PROJECT_NAME})

add_executable(face_tracking_sample "face_tracking_sample.cpp")
target_link_libraries(face_tracking_sample ${PROJECT_NAME})

add_executable(image_des_sample "image_des_sample.cpp")
target_link_libraries(image_des_sample ${PROJECT_NAME})

add_executable(image_src_sample "image_src_sample.cpp")
target_link_libraries(image_src_sample ${PROJECT_NAME})

add_executable(interaction_with_pipe_sample "interaction_with_pipe_sample.cpp")
target_link_libraries(interaction_with_pipe_sample ${PROJECT_NAME})

add_executable(mask_rcnn_sample "mask_rcnn_sample.cpp")
target_link_libraries(mask_rcnn_sample ${PROJECT_NAME})

add_executable(message_broker_sample "message_broker_sample.cpp")
target_link_libraries(message_broker_sample ${PROJECT_NAME})

add_executable(multi_detectors_and_classifiers_sample "multi_detectors_and_classifiers_sample.cpp")
target_link_libraries(multi_detectors_and_classifiers_sample ${PROJECT_NAME})

add_executable(N-1-N_sample "N-1-N_sample.cpp")
target_link_libraries(N-1-N_sample ${PROJECT_NAME})

add_executable(N-N_sample "N-N_sample.cpp")
target_link_libraries(N-N_sample ${PROJECT_NAME})

add_executable(1-N-1_sample "1-N-1_sample.cpp")
target_link_libraries(1-N-1_sample ${PROJECT_NAME})

add_executable(1-N-1_sample2 "1-N-1_sample2.cpp")
target_link_libraries(1-N-1_sample2 ${PROJECT_NAME})

add_executable(1-N-1_sample3 "1-N-1_sample3.cpp")
target_link_libraries(1-N-1_sample3 ${PROJECT_NAME})

add_executable(openpose_sample "openpose_sample.cpp")
target_link_libraries(openpose_sample ${PROJECT_NAME})

add_executable(record_sample "record_sample.cpp")
target_link_libraries(record_sample ${PROJECT_NAME})

add_executable(rtsp_des_sample "rtsp_des_sample.cpp")
target_link_libraries(rtsp_des_sample ${PROJECT_NAME})

add_executable(vp_logger_sample "vp_logger_sample.cpp")
target_link_libraries(vp_logger_sample ${PROJECT_NAME})

# samples depend on PaddlePaddle
if(VP_WITH_PADDLE)
    add_executable(app_src_sample "app_src_sample.cpp")
    target_link_libraries(app_src_sample ${PROJECT_NAME})
    
    add_executable(paddle_infer_sample "paddle_infer_sample.cpp")
    target_link_libraries(paddle_infer_sample ${PROJECT_NAME})
endif()

# samples depend on TensorRT
if(VP_WITH_TRT)
    add_executable(ba_jam_sample "ba_jam_sample.cpp")
    target_link_libraries(ba_jam_sample ${PROJECT_NAME})

    add_executable(ba_stop_sample "ba_stop_sample.cpp")
    target_link_libraries(ba_stop_sample ${PROJECT_NAME})

    add_executable(body_scan_and_plate_detect_sample "body_scan_and_plate_detect_sample.cpp")
    target_link_libraries(body_scan_and_plate_detect_sample ${PROJECT_NAME})

    add_executable(dynamic_pipeline_sample "dynamic_pipeline_sample.cpp")
    target_link_libraries(dynamic_pipeline_sample ${PROJECT_NAME})

    add_executable(dynamic_pipeline_sample2 "dynamic_pipeline_sample2.cpp")
    target_link_libraries(dynamic_pipeline_sample2 ${PROJECT_NAME})    

    add_executable(message_broker_sample2 "message_broker_sample2.cpp")
    target_link_libraries(message_broker_sample2 ${PROJECT_NAME})

    add_executable(multi_trt_infer_nodes_sample "multi_trt_infer_nodes_sample.cpp")
    target_link_libraries(multi_trt_infer_nodes_sample ${PROJECT_NAME})

    add_executable(plate_recognize_sample "plate_recognize_sample.cpp")
    target_link_libraries(plate_recognize_sample ${PROJECT_NAME})

    add_executable(src_des_sample "src_des_sample.cpp")
    target_link_libraries(src_des_sample ${PROJECT_NAME})

    add_executable(trt_infer_sample "trt_infer_sample.cpp")
    target_link_libraries(trt_infer_sample ${PROJECT_NAME})

    add_executable(vehicle_body_scan_sample "vehicle_body_scan_sample.cpp")
    target_link_libraries(vehicle_body_scan_sample ${PROJECT_NAME})

    add_executable(vehicle_cluster_based_on_classify_encoding_sample "vehicle_cluster_based_on_classify_encoding_sample.cpp")
    target_link_libraries(vehicle_cluster_based_on_classify_encoding_sample ${PROJECT_NAME})

    add_executable(vehicle_tracking_sample "vehicle_tracking_sample.cpp")
    target_link_libraries(vehicle_tracking_sample ${PROJECT_NAME})
endif()

if(VP_BUILD_COMPLEX_SAMPLES)
    message("start build for complex sampels...")
    # save exe to 'face_recognize'
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/face_recognize)
    add_executable(face_recognize_pipeline "face_recognize/face_recognize_pipeline.cpp")
    target_link_libraries(face_recognize_pipeline ${PROJECT_NAME})

    if(VP_WITH_TRT) # samples depend on TensorRT   
        # save exe to 'similiarity_search'
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/similiarity_search)
        add_executable(face_encoding_pipeline "similiarity_search/face_encoding_pipeline.cpp")
        target_link_libraries(face_encoding_pipeline ${PROJECT_NAME})
        add_executable(vehicle_encoding_pipeline "similiarity_search/vehicle_encoding_pipeline.cpp")
        target_link_libraries(vehicle_encoding_pipeline ${PROJECT_NAME})

        # save exe to 'vehicle_behaviour_analysis'
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/vehicle_behaviour_analysis)
        add_executable(vehicle_ba_pipeline "vehicle_behaviour_analysis/vehicle_ba_pipeline.cpp")
        target_link_libraries(vehicle_ba_pipeline ${PROJECT_NAME})

        # save exe to 'vehicle_property_and_similiarity_search'
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/vehicle_property_and_similiarity_search)
        add_executable(vehicle_encoding_classify_pipeline "vehicle_property_and_similiarity_search/vehicle_encoding_classify_pipeline.cpp")
        target_link_libraries(vehicle_encoding_classify_pipeline ${PROJECT_NAME})

        # save exe to 'lpr_camera'
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lpr_camera)
        add_executable(plate_recognize_pipeline "lpr_camera/plate_recognize_pipeline.cpp")
        target_link_libraries(plate_recognize_pipeline ${PROJECT_NAME})
    endif()
endif()