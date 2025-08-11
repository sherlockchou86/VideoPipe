 
 a lightweight LLM Client SDK using modern C++ Language, supports calling OpenAI-compatible online services — such as Alibaba Cloud, OpenAI, and other API providers — as well as locally hosted inference backends like Ollama, vLLM, and Hugging Face TGI.

> NOTE: OpenSSL >= 3.0 required for llmlib.hpp.
```
mkdir build
cd build
cmake ..
make -j8

./llmlib_ollama_test # test chat REST API for Ollama
./llmlib_openai_test # test chat REST API for OpenAI-compatible protocol(aliyun LLM API provider/vLLM/...)
```