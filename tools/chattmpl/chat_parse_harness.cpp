// N14 (gap 6 of docs/reassessment/CORRECTNESS.md): llama.cpp's own chat code, without a model, on a chat template.
//
// For each request of a JSON spec it builds the same common_chat_session llama-server builds for /v1/chat/completions
// (tools/server/server-common.cpp oaicompat_chat_params_parse @79e2e74: common_chat_msgs_parse_oaicompat,
// common_chat_tools_parse_oaicompat, kwargs merged over the default preserve_reasoning=true of common/arg.cpp:964-966,
// reasoning_format deepseek = common/common.h:657) and prints the detected format, the rendered prompt, the generation
// prompt, the thinking tags, the message delimiters the server uses for checkpoints, and, for every candidate model
// output, the message the server's parser makes of it (common_chat_session::finish, i.e. the end of a non-streamed
// completion). The vocabulary is null: the generation prompt is matched as text, as the server does when its
// tokenization round trip fails (common/chat.cpp common_chat_input_tokenize).
//
// Spec (JSON file, argv[1]):
//   {"template_file": "...", "requests": [{"id": "...", "messages": [...OpenAI form...], "tools": [...],
//     "kwargs": {"name": <json value>}, "add_generation_prompt": true, "parallel_tool_calls": true,
//     "tool_choice": "auto", "reasoning_format": "deepseek", "outputs": ["<model output>", ...]}]}
// Output: one JSON object per request, one per line, on stdout. With CHATTMPL_DEBUG=1 in the environment the common
// log runs at debug verbosity on stderr: it names the handler, "Using specialized template: ..." or "using
// differential autoparser" (common/chat.cpp common_chat_template_detect_params_init, common_chat_templates_apply_jinja).
//
// Build: tools/chattmpl/CMakeLists.txt (out of tree against a llama.cpp checkout), or tools/chattmpl/run_n14.sh.

#include "chat.h"
#include "json.h"
#include "log.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>

static std::string read_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("cannot read " + path);
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static const char * role_name(common_chat_role r) {
    return common_chat_role_to_string(r);
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s spec.json\n", argv[0]);
        return 2;
    }
    if (getenv("CHATTMPL_DEBUG")) {
        common_log_set_verbosity_thold(LOG_LEVEL_DEBUG);
    }
    const common_json spec = common_json::parse(read_file(argv[1]));
    const std::string src = read_file(spec.at("template_file").get<std::string>());

    common_chat_templates_ptr tmpls = common_chat_templates_init(nullptr, src, "", "");
    const bool supports_thinking = common_chat_templates_support_enable_thinking(tmpls.get());
    const auto caps = common_chat_templates_get_caps(tmpls.get());

    for (const auto & req : spec.at("requests")) {
        common_json out = common_json::object();
        out["id"] = req.at("id").get<std::string>();
        try {
            common_chat_templates_inputs inputs;
            inputs.messages              = common_chat_msgs_parse_oaicompat(req.at("messages"));
            inputs.tools                 = common_chat_tools_parse_oaicompat(req.value("tools", common_json()));
            inputs.tool_choice           = common_chat_tool_choice_parse_oaicompat(req.value("tool_choice", std::string("auto")));
            inputs.use_jinja             = true;
            inputs.add_generation_prompt = req.value("add_generation_prompt", true);
            inputs.parallel_tool_calls   = req.value("parallel_tool_calls", caps.at("supports_parallel_tool_calls"));
            inputs.reasoning_format      = common_reasoning_format_from_name(req.value("reasoning_format", std::string("deepseek")));
            inputs.enable_thinking       = supports_thinking;
            inputs.chat_template_kwargs["preserve_reasoning"] = "true";
            if (req.contains("kwargs")) {
                for (const auto & item : req.at("kwargs").items()) {
                    inputs.chat_template_kwargs[item.key()] = item.value().dump();
                }
            }
            auto it = inputs.chat_template_kwargs.find("enable_thinking");
            if (it != inputs.chat_template_kwargs.end()) {
                inputs.enable_thinking = it->second == "true";
            }

            common_chat_session session(tmpls.get(), nullptr, inputs);
            out["format"]             = std::string(common_chat_format_name(session.format()));
            out["supports_thinking"]  = supports_thinking;
            out["prompt"]             = session.prompt();
            out["generation_prompt"]  = session.generation_prompt();
            out["thinking_start_tag"] = session.thinking_start_tag();
            common_json ends = common_json::array();
            for (const auto & e : session.thinking_end_tags()) {
                ends.push_back(e);
            }
            out["thinking_end_tags"] = ends;
            out["grammar_nonempty"]  = !session.grammar().empty();
            common_json delims = common_json::array();
            for (const auto & d : session.message_delimiters().delimiters) {
                common_json jd = common_json::object();
                jd["role"]      = std::string(role_name(d.role));
                jd["delimiter"] = d.delimiter;
                delims.push_back(jd);
            }
            out["delimiters"] = delims;

            common_json parsed = common_json::array();
            if (req.contains("outputs")) {
                for (const auto & o : req.at("outputs")) {
                    common_json po = common_json::object();
                    try {
                        common_chat_session s(tmpls.get(), nullptr, inputs);
                        const auto & msg = s.finish(common_chat_input(o.get<std::string>()));
                        po["message"] = msg.to_json_oaicompat();
                    } catch (const std::exception & e) {
                        po["error"] = std::string(e.what());
                    }
                    parsed.push_back(po);
                }
            }
            out["parsed"] = parsed;
        } catch (const std::exception & e) {
            out["error"] = std::string(e.what());
        }
        printf("%s\n", out.dump().c_str());
        fflush(stdout);
    }
    return 0;
}
