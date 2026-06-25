#include "vp_pegasus_analyser_node.h"

#ifdef VP_WITH_LLM
#ifndef CPPHTTPLIB_OPENSSL_SUPPORT
#define CPPHTTPLIB_OPENSSL_SUPPORT
#endif
#include "../../third_party/cpp_httplib/httplib.h"
#include "../../third_party/nlohmann/json.hpp"

using json = nlohmann::json;

namespace vp_nodes {
    vp_pegasus_analyser_node::vp_pegasus_analyser_node(std::string node_name,
                              std::string api_key,
                              std::string prompt,
                              std::string video_url,
                              std::string video_id,
                              std::string model_name,
                              int max_tokens,
                              std::string api_host):
                              vp_primary_infer_node(node_name, ""),
                              api_key(api_key),
                              prompt(prompt),
                              video_url(video_url),
                              video_id(video_id),
                              model_name(model_name),
                              max_tokens(max_tokens),
                              api_host(api_host) {
        this->initialized();
    }

    vp_pegasus_analyser_node::~vp_pegasus_analyser_node() {
        deinitialized();
    }

    std::string vp_pegasus_analyser_node::analyse_video() {
        // build request payload for the TwelveLabs /v1.3/analyze endpoint.
        json payload;
        payload["model_name"] = model_name;
        payload["prompt"] = prompt;
        payload["max_tokens"] = max_tokens;
        payload["stream"] = false;
        // provide exactly one video source: a direct url (preferred, works with both
        // pegasus1.5 and pegasus1.2) or a pre-indexed video id (pegasus1.2 only).
        if (!video_url.empty()) {
            payload["video"] = {{"type", "url"}, {"url", video_url}};
        } else {
            payload["video_id"] = video_id;
        }

        httplib::Client cli(api_host.c_str());
        // Pegasus reasons over the whole video and may take a while; allow a generous timeout.
        cli.set_connection_timeout(10, 0);
        cli.set_read_timeout(300, 0);

        httplib::Headers headers = {
            {"x-api-key", api_key},
            {"Content-Type", "application/json"}
        };

        auto res = cli.Post("/v1.3/analyze", headers, payload.dump(), "application/json");
        if (!res) {
            VP_WARN(vp_utils::string_format("[%s] TwelveLabs request failed: %s",
                    node_name.c_str(), httplib::to_string(res.error()).c_str()));
            return "";
        }
        if (res->status != httplib::StatusCode::OK_200) {
            VP_WARN(vp_utils::string_format("[%s] TwelveLabs HTTP status %d: %s",
                    node_name.c_str(), res->status, res->body.c_str()));
            return "";
        }

        try {
            auto res_json = json::parse(res->body);
            // non-streaming response shape: {"id":..., "data":"<text>", "finish_reason":...}
            return res_json.value("data", std::string());
        } catch (const std::exception& e) {
            VP_WARN(vp_utils::string_format("[%s] failed to parse TwelveLabs response: %s",
                    node_name.c_str(), e.what()));
            return "";
        }
    }

    void vp_pegasus_analyser_node::run_infer_combinations(const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) {
        assert(frame_meta_with_batch.size() == 1);
        auto& frame_meta = frame_meta_with_batch[0];

        // Pegasus analyses the whole video, so we only need to call the api once and
        // then reuse the cached structuring for every frame flowing through this node.
        if (!analysed) {
            cached_description = analyse_video();
            analysed = true;
            VP_INFO(vp_utils::string_format("[%s] Pegasus analysis: %s",
                    node_name.c_str(), cached_description.c_str()));
        }
        frame_meta->description = cached_description;
    }

    void vp_pegasus_analyser_node::postprocess(const std::vector<cv::Mat>& raw_outputs, const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) {

    }
}
#endif
