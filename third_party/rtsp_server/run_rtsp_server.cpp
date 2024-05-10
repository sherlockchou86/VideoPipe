
#include <iostream>
#include <thread>
#include <vector>
#include <sstream>
#include <utility>
#include <gst/gst.h>
#include <gst/rtsp-server/rtsp-server.h>

inline std::vector<std::string> string_split(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

/*
 * usage:
 * ./run_rtsp_server [-p] [rtsp_port] [-s] [stream_name1:inner_port1/stream_name2:inner_port2/...]
 * 
 * for example:
 * ./run_rtsp_server -p 8554 -s 1st_rtsp:8000/2nd_rtsp:8001/3rd_rtsp:8002
 * rtsp server will listen at ports(8000/8001/8002) to receive udp stream and distribute them at port 8554 using rtsp protocal:
 * (1) rtsp://127.0.0.1:8554/1st_rtsp
 * (2) rtsp://127.0.0.1:8554/2nd_rtsp
 * (3) rtsp://127.0.0.1:8554/3rd_rtsp
*/
bool parse_args(int argc, char** argv, std::string& rtsp_port, std::vector<std::pair<std::string, std::string>>& streams_info) {
    auto print_error = []() {
        std::cout << "invalid arguments!" << std::endl;
        std::cout << "./run_rtsp_server [-p] [rtsp_port] [-s] [stream_name1:inner_port1/stream_name2:inner_port2/...]" << std::endl;
    };

    if (argc != 5) {
        print_error();
        return false;
    }
    
    auto _p = std::string(argv[1]);
    rtsp_port = std::string(argv[2]);
    auto _s = std::string(argv[3]);
    auto streams = std::string(argv[4]);

    if (_p != "-p" || _s != "-s") {
        print_error();
        return false;
    }

    auto streams_list = string_split(streams, '/');
    for (auto& stream: streams_list) {
        auto name_port = string_split(stream, ':');
        if (name_port.size() != 2) {
            print_error();
            return false;
        }
        streams_info.push_back(std::pair<std::string, std::string>(name_port[0], name_port[1]));
    }
    return true;
}

int main(int argc, char** argv) {
    std::string rtsp_port;
    std::vector<std::pair<std::string, std::string>> streams_info;
    if (!parse_args(argc, argv, rtsp_port, streams_info)) {
        return -1;
    }
    
    auto rtsp_server_run = [&]() {
        GMainLoop* loop;
        GstRTSPServer* server;

        gst_init(&argc, &argv);
        server = gst_rtsp_server_new();
        g_object_set(server, "service", rtsp_port.c_str(), NULL);
        gst_rtsp_server_attach(server, NULL);
        auto mounts = gst_rtsp_server_get_mount_points(server);

        for(auto& stream: streams_info) {
            auto stream_name = stream.first;
            auto stream_port = stream.second;

            char udpsrc_pipeline[512];
            int udp_buffer_size = 512 * 1024;
            
            // receive stream data from udpsrc internally and push it via rtsp
            sprintf (udpsrc_pipeline,
                    "( udpsrc name=pay0 port=%s buffer-size=%lu caps=\"application/x-rtp, media=video, "
                    "clock-rate=90000, encoding-name=H264, payload=96 \" )",
                    stream_port.c_str(), udp_buffer_size);

            auto factory = gst_rtsp_media_factory_new();
            gst_rtsp_media_factory_set_launch(factory, udpsrc_pipeline);
            gst_rtsp_media_factory_set_shared(factory, true);
            gst_rtsp_mount_points_add_factory(mounts, ("/" + stream_name).c_str(), factory);
        }        
        g_object_unref(mounts);

        std::cout << "########## rtsp server info ###########" << std::endl;
        for (int i = 0; i < streams_info.size(); i++) {
            std::cout << "(" << i + 1 << ") " 
                << streams_info[i].first << "==>" << streams_info[i].second 
                << "==>rtsp://127.0.0.1:" << rtsp_port << "/" << streams_info[i].first << std::endl;
        }
        
        loop = g_main_loop_new(NULL, FALSE);
        g_main_loop_run(loop);
        g_main_loop_unref(loop);
        gst_object_unref(server);
    };
    std::thread rtsp_server(rtsp_server_run);

    // enter to exit
    std::string wait;
    std::getline(std::cin, wait);
    return 0;
}