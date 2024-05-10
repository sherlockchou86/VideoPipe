# rtsp_server #

> NOTE: building of VideoPipe is not dependent on this project. 

a simple rtsp server based on `GStreamer` which can accept udp stream and distribute them using `rtsp` protocal something like `rtsp://127.0.0.1:8554/stream1`/`rtsp://127.0.0.1:8554/stream2`.


## how to build ##

first install `GStreamer` refer to `VideoPipe`, and build using CMake:

```
mkdir build && cd build
cmake ..
make -j8

```

## how to run ##
first build `rtsp_server` successfully and run:

`./build/run_rtsp_server [-p] [rtsp_port] [-s] [stream_name1:inner_port1/stream_name2:inner_port2/...]`
```
for example:
./build/run_rtsp_server -p 8555 -s rtsp0:8000/rtsp1:9000/rtsp2:9005

rtsp server will listen at ports(8000/9000/9005) to receive udp stream and distribute them at port 8555 using rtsp protocal:
########## rtsp server info ###########
(1) rtsp0==>8000==>rtsp://127.0.0.1:8555/rtsp0
(2) rtsp1==>9000==>rtsp://127.0.0.1:8555/rtsp1
(3) rtsp2==>9005==>rtsp://127.0.0.1:8555/rtsp2
```

then you need push your stream to port `8000`, `9000`, `9005` using some multi-media tools like `gst-launch-1.0` or your application.

```
# push udp stream to port 8000
gst-launch-1.0 filesrc location=face2.mp4 ! qtdemux ! h264parse ! avdec_h264 ! x264enc ! rtph264pay ! udpsink host=localhost port=8000

# push udp stream to port 9000
gst-launch-1.0 filesrc location=face2.mp4 ! qtdemux ! h264parse ! avdec_h264 ! x264enc ! rtph264pay ! udpsink host=localhost port=9000

# push udp stream to port 9005
gst-launch-1.0 filesrc location=face2.mp4 ! qtdemux ! h264parse ! avdec_h264 ! x264enc ! rtph264pay ! udpsink host=localhost port=9005
```

now you can pull/play rtsp stream using `vlc` or `gst-launch-1.0` by rtsp urls:
```
rtsp://127.0.0.1:8555/rtsp0
rtsp://127.0.0.1:8555/rtsp1
rtsp://127.0.0.1:8555/rtsp2
```

## how rtsp_server serve VideoPipe? ##

`vp_udp_des_node` in VideoPipe will push udp streams to rtsp_server, which distribute them by rtsp protocal.