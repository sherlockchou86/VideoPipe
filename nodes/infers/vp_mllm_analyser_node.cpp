#include "vp_mllm_analyser_node.h"

#ifdef VP_WITH_LLM
namespace vp_nodes {
    vp_mllm_analyser_node::vp_mllm_analyser_node(std::string node_name,
                              std::string model_name,
                              std::string prompt,
                              std::string api_base_url,
                              std::string api_key,
                              llmlib::LLMBackendType backend_type):
                              vp_primary_infer_node(node_name, ""),
                              llm_model_name(model_name),
                              llm_prompt(prompt) {
        cli = llmlib::LLMClient(api_base_url, api_key, backend_type);
        this->initialized();
    }

    vp_mllm_analyser_node::~vp_mllm_analyser_node() {
        deinitialized();
    }

    void vp_mllm_analyser_node::run_infer_combinations(const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) {
        assert(frame_meta_with_batch.size() == 1);
        auto& frame_meta = frame_meta_with_batch[0];

        auto output = cli.simple_chat(llm_model_name, llm_prompt, {frame_meta->frame}, {});
        frame_meta->description = output;
    }

    void vp_mllm_analyser_node::postprocess(const std::vector<cv::Mat>& raw_outputs, const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) {

    }
}
#endif