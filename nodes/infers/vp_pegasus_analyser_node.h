#pragma once

#ifdef VP_WITH_LLM
#include "../vp_primary_infer_node.h"

namespace vp_nodes {
    // video analyser based on TwelveLabs Pegasus video understanding model.
    //
    // unlike vp_mllm_analyser_node (which sends a single frame/image per request to a
    // multimodal LLM), Pegasus analyses the WHOLE video on the TwelveLabs cloud and
    // returns a semantic, structured description of it (summary, chapters, highlights,
    // open-ended Q&A, ...). this is well suited to video structuring scenarios.
    //
    // because Pegasus reasons over the entire video rather than individual frames, the
    // analysis is performed ONCE (against a video url or a pre-indexed video id) and the
    // resulting text is cached, then written into frame_meta->description for frames
    // flowing through this node.
    class vp_pegasus_analyser_node: public vp_primary_infer_node {
    private:
        // TwelveLabs model name, e.g. "pegasus1.5" (accepts a video url) or
        // "pegasus1.2" (accepts a pre-indexed video id).
        std::string model_name;
        // prompt guiding the analysis, e.g. "Summarize this video in 3 sentences.".
        std::string prompt;
        // direct http(s) url to a raw media file. mutually exclusive with video_id.
        std::string video_url;
        // pre-indexed TwelveLabs video id (pegasus1.2 only). mutually exclusive with video_url.
        std::string video_id;
        // TwelveLabs api key (x-api-key header).
        std::string api_key;
        // max output tokens (TwelveLabs requires >= 512 for pegasus1.5).
        int max_tokens;
        // api host, e.g. "https://api.twelvelabs.io".
        std::string api_host;

        // cached analysis result; Pegasus reasons over the whole video so we only call once.
        std::string cached_description;
        bool analysed = false;

        // call the TwelveLabs /v1.3/analyze endpoint and return the generated text.
        std::string analyse_video();
    protected:
        // whole-video analysis needs custom logic, no per-step inference from base class.
        virtual void run_infer_combinations(const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) override;
        // override pure virtual method, for compile pass.
        virtual void postprocess(const std::vector<cv::Mat>& raw_outputs, const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) override;
    public:
        // analyse a video by direct url (use model_name "pegasus1.5").
        // for a pre-indexed video, pass video_id and leave video_url empty (use "pegasus1.2").
        vp_pegasus_analyser_node(std::string node_name,
                                 std::string api_key,
                                 std::string prompt,
                                 std::string video_url,
                                 std::string video_id = "",
                                 std::string model_name = "pegasus1.5",
                                 int max_tokens = 2048,
                                 std::string api_host = "https://api.twelvelabs.io");
        ~vp_pegasus_analyser_node();
    };
}
#endif
