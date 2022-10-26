# rtsp-simple-server
ready-to-use RTSP / RTMP / LL-HLS server and proxy that allows to read, publish and proxy video and audio streams, refer to `https://github.com/aler9/rtsp-simple-server`.

# usage
```shell
docker run --name rtsp-simple-server -d --restart always  -e RTSP_PROTOCOLS=tcp -p 8554:8554 -p 1935:1935 -p 8888:8888 aler9/rtsp-simple-server
```

# push rtmp
```shell
ffmpeg -re -stream_loop -1 -i file.ts -c copy -f flv rtmp://localhost/mystream
```
