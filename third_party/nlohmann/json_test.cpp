#include <iostream>
#include "json.hpp"
using json = nlohmann::json;

int main() {
    /**
     * 1. [OpenAI protocols]
     * simulate construct request payload and dump it to string based on OpenAI protocols
     * 
    */
    json openai_request_payload = {
        {"model", "qwen-vl-max"},
        {"messages", {{
            {"role", "system"},
            {"content", {
                {{"type", "text"}, {"text", "You are a helpful assistant."}}
            }}
        },
        {
            {"role", "user"},
            {"content", {
                {{"type", "text"}, {"text", "请描述这张图。"}},
                {
                    {"type", "image_url"},
                    {"image_url", {{"url", "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241022/emyrja/dog_and_girl.jpeg"}}}
                }
            }}
        }
        }},
        {"temperature", 2},
        {"top_k", 5},
        {"stream", false}
    };
    auto openai_request_str = openai_request_payload.dump(4);
    std::cout << "-------------------------------" << std::endl;
    std::cout << openai_request_str << std::endl;
    std::cout << "-------------------------------" << std::endl;

    /**
     * 2. [OpenAI protocols]
     * simulate parse response string to json object based on OpenAI protocols, and print content field
     * 
    */
    auto openai_response_str = R"(
    {
        "choices": [
            {
            "message": {
                "content": "这张图片展示了一位女士和一只狗在海滩上互动。女士坐在沙滩上，微笑着与狗握手。背景是大海和天空，阳光洒在她们身上，营造出温暖的氛围。狗戴着项圈，显得很温顺。",
                "role": "assistant"
            },
            "finish_reason": "stop",
            "index": 0,
            "logprobs": null
            }
        ],
        "object": "chat.completion",
        "usage": {
            "prompt_tokens": 1270,
            "completion_tokens": 54,
            "total_tokens": 1324
        },
        "created": 1725948561,
        "system_fingerprint": null,
        "model": "qwen-vl-max",
        "id": "chatcmpl-0fd66f46-b09e-9164-a84f-3ebbbedbac15"
    }
    )";
    auto openaij = json::parse(openai_response_str);
    std::cout << "-------------------------------" << std::endl;
    std::cout << openaij["choices"][0]["message"]["content"] << std::endl;
    std::cout << "-------------------------------" << std::endl;

    /**
     * 3. [Ollama protocols]
     * simulate construct request payload and dump it to string based on Ollama protocols
     * 
    */
    json ollama_request_payload = {
        {"model", "llava"},
        {"messages", {{
            {"role", "user"},
            {"content", "what is in this image?"},
            {"images", {"base64"}}
        }}},
        {"options", {
            {"temperature", 1},
            {"top_k", 2}
        }},
        {"stream", false}
    };
    auto ollama_request_str = ollama_request_payload.dump(4);
    std::cout << "-------------------------------" << std::endl;
    std::cout << ollama_request_str << std::endl;
    std::cout << "-------------------------------" << std::endl;

    /**
     * 4. [Ollama protocols]
     * simulate parse response string to json object based on Ollama protocols, and print content field
     * 
    */
    auto ollama_response_str = R"(
    {
        "model": "llava",
        "created_at": "2023-12-13T22:42:50.203334Z",
        "message": {
            "role": "assistant",
            "content": " The image features a cute, little pig with an angry facial expression. It's wearing a heart on its shirt and is waving in the air. This scene appears to be part of a drawing or sketching project.",
            "images": null
        },
        "done": true,
        "total_duration": 1668506709,
        "load_duration": 1986209,
        "prompt_eval_count": 26,
        "prompt_eval_duration": 359682000,
        "eval_count": 83,
        "eval_duration": 1303285000
    }
    )";
    auto ollamaj = json::parse(ollama_response_str);
    std::cout << "-------------------------------" << std::endl;
    std::cout << ollamaj["message"]["content"] << std::endl;
    std::cout << "-------------------------------" << std::endl;

    /**
     * 5.
     * simulate parse json string to json object using `_json`.
    */
    
    auto j = R"(
        {
            "name": "jack",
            "sex": "male",
            "age": 18,
            "p1": {
                "aaa": 0.5,
                "bbb": [1, 2, 3]
            }
        }
    )"_json;
    std::cout << "-------------------------------" << std::endl;
    std::cout << j["p1"]["bbb"] << std::endl;
    std::cout << "-------------------------------" << std::endl;
}