#pragma once

#ifdef VP_WITH_LLM
#include "../vp_primary_infer_node.h"
#include "../../third_party/cpp_llmlib/llmlib.hpp"

namespace vp_nodes {
    // image(frame) analyser based on Multimodal Large Language Model
    class vp_mllm_analyser_node: public vp_primary_infer_node 
    {
    private:
        /* data */
        llmlib::LLMClient cli;
        std::string llm_prompt;
        std::string llm_model_name;
    protected:
        // we need a totally new logic for the whole infer combinations
        // no separate step pre-defined needed in base class
        virtual void run_infer_combinations(const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) override;
        // override pure virtual method, for compile pass
        virtual void postprocess(const std::vector<cv::Mat>& raw_outputs, const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) override;
    public:
        vp_mllm_analyser_node(std::string node_name,
                              std::string model_name,
                              std::string prompt,
                              std::string api_base_url,
                              std::string api_key = "",
                              llmlib::LLMBackendType backend_type = llmlib::LLMBackendType::Ollama);
        ~vp_mllm_analyser_node();
    };
}
#endif