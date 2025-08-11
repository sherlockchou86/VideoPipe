#include <iostream>
#include "httplib.h"

int main() {
    /**
     * GET html from website: http://www.videopipe.cool
    */
    httplib::Client get_cli("http://www.videopipe1.cool");
    if (auto res = get_cli.Get("/index.php/bloglist/")) {
        if (res->status == httplib::StatusCode::OK_200) {
            std::cout << res->body << std::endl;
        } 
        else {
            std::cout << res->status << std::endl;
        }
    } 
    else {
        auto err = res.error();
        std::cout << "HTTP error: " << httplib::to_string(err) << std::endl;
    }

    /**
     * POST data with headers to LLM service based on Ollama.
    */
    httplib::Client post_cli("http://192.168.77.219:11434");
    httplib::Headers headers = {{"myheader1", "1111"}, {"myheader2", "2222"}};
    auto payload = R"(
    {
        "model": "qwen2.5:7b",
        "prompt": "为什么坐火车要给钱？",
        "stream": false
    }
    )";
    if (auto res = post_cli.Post("/api/generate", headers, payload, "application/json")) {
        if (res->status == httplib::StatusCode::OK_200) {
            std::cout << res->body << std::endl;
        } 
        else {
            std::cout << res->body << std::endl;
            std::cout << res->status << std::endl;
        }
    } 
    else {
        auto err = res.error();
        std::cout << "HTTP error: " << httplib::to_string(err) << std::endl;
    }
}