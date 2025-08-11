#include <iostream>
#include "llmlib.hpp"
using namespace llmlib;

int main() {
    // create LLMClient
    LLMClient llmcli("http://192.168.77.219:11434", "", LLMBackendType::Ollama);

    // define parameters
    auto model_name = "qwen2.5:7b";
    json options = {{"temperature", 0.1}, {"top_k", 1}};
    json messages = {{{"role", "system"}, {"content", "你是一个非常专业的聊天助手，你的名字叫小王。"}}};

    // chat loop runing on console
    std::string input = "";
    std::cout << "##let's chat with LLM hosted by Ollama##" << std::endl;
    std::cout << "YOU: ";
    do
    {
        std::cin >> input;
        if (input != "quit") {
            // append user's input to history messages
            messages.push_back({{"role", "user"}, {"content", input}});

            // call chat()
            auto output = llmcli.chat(model_name, messages, options);
            std::cout << "-----------------------------------------" << std::endl << "LLM: " << output << std::endl;
            std::cout << "-----------------------------------------" << std::endl << "YOU: ";

            // append assistant's output to history messages
            messages.push_back({{"role", "assistant"}, {"content", output}});
        }
    } while (input != "quit");
    std::cout << "##finish chat with LLM hosted by Ollama##" << std::endl;
}