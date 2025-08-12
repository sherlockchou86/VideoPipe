#define CPPHTTPLIB_OPENSSL_SUPPORT
#include <string>
#include <map>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "../cpp_httplib/httplib.h"
#include "../cpp_base64/base64.h"
#include "../nlohmann/json.hpp"
using json = nlohmann::json;

namespace llmlib {
    /**
     * LLM backends:
     * 1. OpenAI: OpenAI-compatible LLM service API providers
     * 2. Ollama: LLM services hosted locally by Ollama framework
     * 
    */
    enum class LLMBackendType {
        OpenAI,
        Ollama
    };


    /**
     * LLM Client:
     * 
     * support chat with LLM with text & image as input.
    */
    class LLMClient {
        public:
            LLMClient() {}
            LLMClient(const std::string& api_base_url, 
                    const std::string& api_key,
                    LLMBackendType backend_type = LLMBackendType::OpenAI):
                    __api_base_url(api_base_url),
                    __api_key(api_key),
                    __backend_type(backend_type)
            {}

            /**
             * set default headers for all requests.
            */
            void set_default_header(const std::string& key, const std::string& value) {
                __default_headers[key] = value;
            }

            /**
             * set connection timeout(seconds) for all requests.
            */
            void set_connection_timeout(int connection_timeout) {
                __connection_timeout = connection_timeout;
            }

            /**
             * simple chat interface.
             * support single text prompt and multi images as input, history messages not Supported.
             * 
             * @param model_name LLM to be used.
             * @param prompt input text prompt.
             * @param images input images.
             * @param options LLM parameters: temperature, top_k, etc.
            */
            std::string simple_chat(const std::string& model_name, 
                                    const std::string& prompt,
                                    const std::vector<cv::Mat>& images,
                                    const json& options) {
                // construct json payload
                json payload;
                payload["model"] = model_name;
                payload["stream"] = false;

                if (__backend_type == LLMBackendType::OpenAI) {                
                    json message;
                    message["role"] = "user";
                    json content = json::array();
                    content.push_back({{"type", "text"}, {"text", prompt}});
                    if (images.size() != 0) {
                        for(auto& image: images) {
                            auto base64_img = mat_to_base64(image);
                            content.push_back({{"type", "image_url"}, {"image_url", {{"url", "data:image/jpg;base64," + base64_img}}}});
                        }
                    }
                    
                    message["content"] = content;
                    payload["messages"] = {message};
                    if (!options.is_null()) {
                        payload.update(options);
                    }
                } else if (__backend_type == LLMBackendType::Ollama) {
                    json message;
                    message["role"] = "user";
                    message["content"] = prompt;
                    if (images.size() != 0) {
                        json imgs = json::array();
                        for(auto& image: images) {
                            auto base64_img = mat_to_base64(image);
                            imgs.push_back(base64_img);
                        }
                        message["images"] = imgs;
                    }
                    payload["messages"] = {message};
                    payload["options"] = options;
                }

                // send request
                auto res = request(payload.dump(4));
                auto res_json = json::parse(res);

                // std::cout << payload.dump(4) << std::endl;
                // std::cout << res << std::endl;

                // try to get result extracted from response
                try
                {
                    if (__backend_type == LLMBackendType::OpenAI) {
                        return res_json["choices"][0]["message"]["content"];
                    } else if (__backend_type == LLMBackendType::Ollama) {
                        return res_json["message"]["content"];
                    } else {
                        return "";
                    }
                }
                catch(const std::exception& e)
                {
                    std::cerr << e.what() << '\n';
                }
                return "";
            }

            /**
             * chat interface.
             * support messages(json array) as input, history messages Supported.
             * 
             * @param model_name LLM to be used.
             * @param messages a json array containing history messages.
             * @param options LLM parameters: temperature, top_k, etc.
             * 
             * NOTE:
             * messages MUST be constructed according to appropriate protocols by caller outside of chat.
            */
            std::string chat(const std::string& model_name,
                            const json& messages,
                            const json& options) {
                // construct json payload
                json payload;
                payload["model"] = model_name;
                payload["messages"] = messages;
                payload["stream"] = false;

                if (__backend_type == LLMBackendType::OpenAI) {
                    if (!options.is_null()) {
                        payload.update(options);
                    }
                } else if (__backend_type == LLMBackendType::Ollama) {
                    payload["options"] = options;
                }
                
                // send request
                auto res = request(payload.dump(4));
                auto res_json = json::parse(res);

                //std::cout << payload.dump(4) << std::endl;
                //std::cout << res << std::endl;

                // try to get result extracted from response
                try
                {
                    if (__backend_type == LLMBackendType::OpenAI) {
                        return res_json["choices"][0]["message"]["content"];
                    } else if (__backend_type == LLMBackendType::Ollama) {
                        return res_json["message"]["content"];
                    } else {
                        return "";
                    }
                }
                catch(const std::exception& e)
                {
                    std::cerr << e.what() << '\n';
                }
                return "";
            }

            /**
             * convert base64 string to cv::Mat
            */
            cv::Mat base64_to_mat(const std::string& base64_data) {
                std::string decoded_data = base64_decode(base64_data);
                std::vector<uchar> data(decoded_data.begin(), decoded_data.end());
                cv::Mat img = cv::imdecode(data, cv::IMREAD_UNCHANGED);
                return img;
            }

            /**
             * convert cv::Mat to base64 string
            */
            std::string mat_to_base64(const cv::Mat& img, const std::string& ext = ".jpg") {
                std::vector<uchar> buf;
                cv::imencode(ext, img, buf);
                std::string encoded = base64_encode(buf.data(), buf.size());
                return encoded;
            }
        private:
            std::string __api_base_url;
            std::string __api_key;
            LLMBackendType __backend_type = LLMBackendType::OpenAI;
            std::map<std::string, std::string> __default_headers;
            int __connection_timeout = 1;
            /**/
            std::string __openai_chat_completions_path = "/chat/completions";
            std::string __ollama_chat_completions_path = "/api/chat";

            std::string request(const std::string& payload) {
                auto parsed_base_url = split_base_url(__api_base_url);
                httplib::Client cli(parsed_base_url.base_host.c_str());
                cli.set_connection_timeout(__connection_timeout, 0);

                httplib::Headers headers;
                auto chat_path = __backend_type == LLMBackendType::OpenAI ? __openai_chat_completions_path : __ollama_chat_completions_path;
                if (__backend_type == LLMBackendType::OpenAI) {
                    headers = {
                        {"Content-Type", "application/json"},
                        {"Authorization", "Bearer " + __api_key}
                    };
                } else if (__backend_type == LLMBackendType::Ollama) {
                    headers = {
                        {"Content-Type", "application/json"}
                    };
                } else {

                }
                for (const auto& kv : __default_headers) {
                    headers.emplace(kv.first, kv.second);
                }

                auto final_path = parsed_base_url.base_path + chat_path;
                auto res = cli.Post(final_path, headers, payload, "application/json");
                if (res) {
                    if (res->status == httplib::StatusCode::OK_200) {
                        return res->body;
                    } 
                    else {
                        std::cout << "[llmlib] HTTP status code: " << res->status << std::endl;
                        return "{}";
                    }
                } else {
                    auto err = res.error();
                    std::cout << "[llmlib] HTTP error: " << httplib::to_string(err) << std::endl;
                    return "{}";
                }
            }

            struct ParsedBaseUrl {
                std::string base_host; // protocols + host + port（if exists）
                std::string base_path; // start with '/', not end with '/'
            };

            ParsedBaseUrl split_base_url(const std::string& url) {
                ParsedBaseUrl result;

                std::regex re(R"(^(https?)://([^:/\s]+)(?::(\d+))?(/.*|$))", std::regex::icase);
                std::smatch match;

                if (!std::regex_search(url, match, re)) {
                    throw std::invalid_argument("Invalid URL format: " + url);
                }

                std::string protocol = match[1].str();
                std::string host = match[2].str();
                std::string port_str = match[3].str();
                std::string path = match[4].str();

                int port = (protocol == "https") ? 443 : 80;
                if (!port_str.empty()) {
                    port = std::stoi(port_str);
                }

                if (port_str.empty()) {
                    result.base_host = protocol + "://" + host;
                } else {
                    result.base_host = protocol + "://" + host + ":" + port_str;
                }

                result.base_path = path.empty() ? "/" : path;
                if (result.base_path.length() > 1 && result.base_path.back() == '/') {
                    result.base_path.pop_back();
                }
                if (result.base_path == "/") {
                    result.base_path.clear();
                }

                return result;
            }
    };
}