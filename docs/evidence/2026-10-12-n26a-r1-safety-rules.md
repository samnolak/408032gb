# N26a: the R1 safety rules of PHASE-CLOSE section (v) as configuration guards (scripts/r1, run_r1, gateway)

Date 2026-10-12. Role: engine-integrator. Branch `claude/n26a-r1-safety`, cut from the N26 integration head `c1ffb80`.
Hardware-free sandbox (no GPU): every run below is a CPU build, a mock server or a unit test. `<S>` stands for the session
scratch directory `/tmp/claude-0/-home-user-408032gb/83634621-ad4f-5747-933c-0d86b3c4b300/scratchpad`; outputs are pasted
with only that substitution.

Task: the owner accepted rules 1-11 of `docs/reassessment/PHASE-CLOSE-2026-10-11.md` section (v) (risk matrix: section
(b)). They are applied to the R1 configuration (`scripts/r1/configs/*.json`, `scripts/r1/run_r1.py`,
`tools/gateway/gateway.py`) as **configuration guards only**: no patch of `patches/llama.cpp/*` is applied to the R1
build (owner's instruction; `scripts/engines/build_engine.py` is unchanged).

## Result

1. **`scripts/r1/safety_rules.py` (stdlib)**, `check(cfg, repo_root) -> [Violation(rule, message)]` and a CLI (exit 1 on a
   violation, 2 on an unreadable file). It encodes rules 1, 2, 4, 5, 8 and 10 (ids below); rule 1's recurrent-rollback
   part is `scripts/r1/rs_rollback_rule.check()` itself (loaded from the file, unchanged). CONFIRMED by the tests and the
   mutation check (sections 5, 6).
2. **`run_r1.py` refuses a violating config before any step**: the check runs on the fully substituted config (with
   `--set`) inside `Cell.__init__`, before the plan is built; a violation is a configuration error, exit 2, no run
   directory, nothing runs, `--plan` included. The report is written to `artifacts/orchestrator/safety_rules.json` (with
   the sha256 of every `--chat-template-file`), a summary to `run.json` (`safety_rules`). CONFIRMED (section 3, tests
   `RunR1Refuses`).
3. **Weights gate (rule 8), the least invasive variant of the step model**: `weights_verify` stays the first step;
   `engine_start` gets one more precondition next to its dependency on `engine_build_check` (same `blocked_by`
   mechanism). A non-mock engine starts only if `weights_verify` passed in this run, or the config has no manifest and
   marks its runs unverified with `model.weights_unverified_reason`; a failed, disabled or unselected `weights_verify`
   blocks it; `--only ... engine_start ...` adds `weights_verify` (as it already added `engine_build_check`). The
   `engine_start` journal record carries `weights` (`verified by weights_verify in this run` / `UNVERIFIED: <reason>` /
   `MOCK (the weights gate does not apply)`). The mock (`"mock": true`) is exempt: its engine is
   `tools/bench/mock_server.py` and its pack is NOT RUN by construction; `R-SYNTH` refuses `"mock": true` on a config
   that starts a llama-server. Why this variant: the existing orchestration tests (and the documented model "a failing
   step is recorded and every later step that does not depend on it still runs") use the mock with failing or unselected
   weights; making the gate apply to non-mock configs only keeps all 24 of them unchanged. CONFIRMED (tests
   `WeightsGate`). `quant_audit` is unchanged.
4. **Configs**: `llamacpp-server-moecache.json` and `ik-llamacpp-server.json` pass `--chat-template-file
   ${REPO}/templates/glm-5.3-flash/chat_template.jinja` after `--jinja` and record `model.chat_template_sha256`;
   both have `weights_manifest: null` and now say why in `model.weights_unverified_reason`; every config has
   `_notes.safety_rules` with the operational rules (3, 6, 7, 9, 11, and 10 for the mock). `${REPO}` already existed as
   a built-in variable of `run_r1.py` (repo root, cannot be redefined: `FIXED_VARS`). The N11 derived CPU config
   (`tools/srvcheck/r1_tools_smoke.py`) gets a `weights_unverified_reason` (its model is SYNTHETIC; without it run_r1
   would now refuse it).
5. **Template sha256** `5d47ac8e48510b5c6543db4000ef145201823bf46bae9cf51ac6a9553a16ef5a`, CONFIRMED (`sha256sum`,
   section 4) = the value N23 reports for this file.
6. **ik_llama.cpp accepts the template**: its CPU build @5194a9e starts with `--chat-template-file <ours> --jinja` and
   `/apply-template` renders a two-assistant-turn conversation without the reasoning leak, byte-equal to jinja2's render
   of the same template. CONFIRMED for that one conversation on a tiny random-weight glm5next GGUF (section 4); the
   N23 corpus through ik: NOT RUN.
7. **Gateway (rule 3)** drops `"reasoning_content": null` members of chat messages, textually, keeping every other byte;
   against the real CPU llama-server @79e2e74 the same request is HTTP 500 direct and HTTP 200 through the gateway.
   CONFIRMED (section 4, tests `NullReasoningContent`).
8. **CI**: `tests/test_r1_safety_rules.py` is collected by `unittest discover` in `.github/workflows/ci.yml` (step
   `unit tests`, every push) and runs explicitly in `.github/workflows/server-cpu.yml` (step "N11 derived R1 config and
   N26a R1 safety rules (no server)", with `tests.test_gateway`), which fails if its log reports any skip. Not wrapped by
   `tools/ci/check_unittest.py`: `tests/test_check_unittest.py::test_real_modules_without_their_environment_fail_at_every_pin`
   requires every wrapped module to skip without its environment, and these tests have no skip path, so
   `tools/ci/parity_expectations.json` is unchanged. Local replay CONFIRMED (section 5); on GitHub NOT RUN (nothing pushed).

## 1. What each rule checks and where

| rule (section (v)) | id | what is checked | where |
|---|---|---|---|
| 1 `-np 1` with speculation | `R-NP` | every llama-server argv (the elements after the one whose basename is `llama-server`: llama.cpp and ik_llama.cpp) gives `-np`/`--parallel` explicitly (llama.cpp's server default is auto: `common/arg.cpp:1406@79e2e74` `params.n_parallel = -1;     // auto by default`, CONFIRMED, printed); the value is an integer; `rs_rollback_rule.check()` holds (its message is kept); and with ANY speculation (`--spec-type` other than `none`, ik's `TYPE:k=v` reduced to `TYPE`; a draft model `-md`/`--spec-draft-model`/`--model-draft`/`-hfd`/`-hfrd`/`--spec-draft-hf`/`--hf-repo-draft` or their env variables, even without `--spec-type`; the presets; `--mtp`/`--dflash`; ik's `-mtp`) the slot count is exactly 1 | `safety_rules.rule_np` |
| 2 GLM template | `R-GLM-TEMPLATE` | GLM = `glm` in `name` or `model.desc` (any case); with `--jinja` in `engine.argv`: `--chat-template-file` present, equal to `${REPO}/templates/glm-5.3-flash/chat_template.jinja` (normalised absolute path), after `--jinja`, no `--chat-template`; any config with `--chat-template-file` records `model.chat_template_sha256` (64 lower-case hex) equal to the sha256 of the file on disk | `safety_rules.rule_glm_template`; configs moe-cache and ik |
| 3 no `reasoning_content: null` | (gateway) | `tools/gateway/gateway.py drop_null_reasoning()`: for `/v1/chat/completions` only, a member `"reasoning_content": null` of `messages[i]` is cut out with its comma; the result must parse to the normalised object (else the body is re-serialised from it); request log `dropped_null_fields`; `--keep-null-reasoning-content` turns it off | gateway; `_notes` of every config |
| 4 Qwen MTP files | `R-MTP-FILES` | Qwen = `qwen` in name/desc; each llama-server draft file (`-md`/`--spec-draft-model`/`--model-draft`, or `LLAMA_ARG_SPEC_DRAFT_MODEL`) has basename `mtp-Qwen3.8-Flash-Next-(Q8_0|BF16|Q4_K_M).gguf`, a `-shared-` basename gets its own message (N18); a draft file needs an explicit `--spec-type draft-mtp`; HF draft flags/env, `--mtp`, `--dflash` and the draft presets are refused | `safety_rules.rule_mtp_files` |
| 5 MiMo + MTP | `R-MIMO-MTP` | MiMo = `mimo` in name/desc; any speculation is refused (the server signs of rule 1, `mode.spec_k` > 0, `--spec N>0` / `--mtp` in `engine.argv` of a non-llama-server engine or in `engine.config_files.*.args`) because the N21 patch is not in the R1 builds | `safety_rules.rule_mimo_mtp` |
| 6 slot restore via session_store | — | operational: `_notes.safety_rules` of the llama.cpp templates (they use `--slot-save-path`) and of ik; README | configs, README |
| 7 own engine rollback | — | operational (our own engine, not llama-server): `_notes.safety_rules` of the Strata template, README | config, README |
| 8 weights | `R-WEIGHTS` + gate | non-mock: `model.weights_manifest` or a non-empty string `model.weights_unverified_reason`, not both; `steps.weights_verify` not disabled when a manifest is set; at run time the weights gate above | `safety_rules.rule_weights`; `run_r1.Run.weights_gate` |
| 9 ik not a parity reference | — | operational: `_notes.safety_rules` of ik (SwiGLU "before SiLU" default semantics until the experiment, PROVISIONAL N24) | config |
| 10 synthetic pack runs | `R-SYNTH` | the pack side was already in `tools/verify/pack.py` (unchanged). Here: a criteria file with `"synthetic_run": true` needs the SYNTHETIC marker (pack's own `synthetic_marked`) in `model.desc`; `"mock": true` needs the marker and must not start a llama-server. Test: among the committed configs only `mock.json` declares a synthetic model | `safety_rules.rule_synth`; test `test_only_the_mock_declares_a_synthetic_model` |
| 11 Qwen `reasoning_effort` | — | operational: `_notes.safety_rules` of the Qwen templates (llama.cpp, Strata) | configs |

Speculation of the Strata template (`--spec ${SPEC_K}`, `--mtp mtp/rt`, `mode.spec_k`) is seen by `speculation()` but only
rule 5 acts on it (rule 1 and 4 are llama-server rules). Residual risk (PROVISIONAL, by construction): a config of a
non-llama-server engine marked `"mock": true` with a SYNTHETIC model description would pass without the weights gate; its
artifacts then say SYNTHETIC.

## 2. Sources checked for this change (lines printed before citing)

- `common/chat.cpp:422-423@79e2e74`: `if (message.contains("reasoning_content")) {` /
  `msg.reasoning_content = message.at("reasoning_content");` CONFIRMED (the HTTP 500 itself: section 4).
- `common/arg.cpp:1406@79e2e74`: `params.n_parallel = -1;     // auto by default` CONFIRMED.
- `common/arg.cpp:3097@79e2e74`: `{"--mtp"},` ("also download the multi-token prediction (MTP) head") CONFIRMED.
- ik_llama.cpp @5194a9e1451b885e24dd5a065f650864e8807c59: `common/common.cpp:2935`
  `if (arg == "--chat-template-file") {`, `:3454` the `--chat-template-file file_with_JINJA_TEMPLATE` help line,
  `common/common.h:323` `int32_t n_parallel = 1;`, `common/common.cpp:2353` `-mtp` (legacy, it throws) CONFIRMED.

## 3. The configs before and after; run_r1 refusals

Before the config changes (the new check against the committed templates of `c1ffb80`):

```
$ python3 scripts/r1/safety_rules.py scripts/r1/configs/*.json   # before the config changes
scripts/r1/configs/ik-llamacpp-server.json: VIOLATES R-GLM-TEMPLATE: GLM with --jinja needs --chat-template-file ${REPO}/templates/glm-5.3-flash/chat_template.jinja (the built-in GGUF template leaks the previous turn's reasoning: risk matrix item 1, N15/N23)
scripts/r1/configs/ik-llamacpp-server.json: VIOLATES R-WEIGHTS: model.weights_manifest is null and there is no model.weights_unverified_reason: give the sha256 manifest (scripts/offline/weights.py) or mark every run of this config unverified with a reason
scripts/r1/configs/llamacpp-server-moecache.json: VIOLATES R-GLM-TEMPLATE: GLM with --jinja needs --chat-template-file ${REPO}/templates/glm-5.3-flash/chat_template.jinja (the built-in GGUF template leaks the previous turn's reasoning: risk matrix item 1, N15/N23)
scripts/r1/configs/llamacpp-server-moecache.json: VIOLATES R-WEIGHTS: model.weights_manifest is null and there is no model.weights_unverified_reason: give the sha256 manifest (scripts/offline/weights.py) or mark every run of this config unverified with a reason
scripts/r1/configs/llamacpp-server.json: ok (R-NP, R-GLM-TEMPLATE, R-MTP-FILES, R-MIMO-MTP, R-WEIGHTS, R-SYNTH)
scripts/r1/configs/mock.json: ok (R-NP, R-GLM-TEMPLATE, R-MTP-FILES, R-MIMO-MTP, R-WEIGHTS, R-SYNTH)
scripts/r1/configs/strata-serve-qwen38fn.json: ok (R-NP, R-GLM-TEMPLATE, R-MTP-FILES, R-MIMO-MTP, R-WEIGHTS, R-SYNTH)
exit 1
```

After:

```
$ python3 scripts/r1/safety_rules.py scripts/r1/configs/*.json; echo "exit $?"   # after (HEAD 856afb4)
scripts/r1/configs/ik-llamacpp-server.json: ok (R-NP, R-GLM-TEMPLATE, R-MTP-FILES, R-MIMO-MTP, R-WEIGHTS, R-SYNTH)
scripts/r1/configs/llamacpp-server-moecache.json: ok (R-NP, R-GLM-TEMPLATE, R-MTP-FILES, R-MIMO-MTP, R-WEIGHTS, R-SYNTH)
scripts/r1/configs/llamacpp-server.json: ok (R-NP, R-GLM-TEMPLATE, R-MTP-FILES, R-MIMO-MTP, R-WEIGHTS, R-SYNTH)
scripts/r1/configs/mock.json: ok (R-NP, R-GLM-TEMPLATE, R-MTP-FILES, R-MIMO-MTP, R-WEIGHTS, R-SYNTH)
scripts/r1/configs/strata-serve-qwen38fn.json: ok (R-NP, R-GLM-TEMPLATE, R-MTP-FILES, R-MIMO-MTP, R-WEIGHTS, R-SYNTH)
exit 0
```

run_r1 on two violating copies (a GLM template without `--chat-template-file`; Qwen with a `-shared-` MTP file,
`--spec-type draft-mtp` and `-np 4`):

```
$ python3 scripts/r1/run_r1.py --config <S>/n26a/glm-no-template.json --out <S>/n26a/runs --plan; echo "exit $?"
run_r1: config error: the R1 safety rules (scripts/r1/safety_rules.py) refuse this config; nothing runs:
  [R-GLM-TEMPLATE] GLM with --jinja needs --chat-template-file ${REPO}/templates/glm-5.3-flash/chat_template.jinja (the built-in GGUF template leaks the previous turn's reasoning: risk matrix item 1, N15/N23)
exit 2
$ python3 scripts/r1/run_r1.py --config <S>/n26a/qwen-shared-np4.json --out <S>/n26a/runs --plan; echo "exit $?"
run_r1: config error: the R1 safety rules (scripts/r1/safety_rules.py) refuse this config; nothing runs:
  [R-NP] --spec-type draft-mtp gives the target recurrent rollback snapshots (n_rs_seq > 0); without the N20 patch run it with -np 1, not 4 slot(s) (scripts/r1/rs_rollback_rule.py)
  [R-MTP-FILES] Qwen MTP file mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf is a -shared- file: it does not load on either pin (token_embd created with flags 0, N18)
exit 2
$ ls <S>/n26a/runs
ls: cannot access '<S>/n26a/runs': No such file or directory
```

## 4. CPU probes: ik_llama.cpp with the template, gateway against llama-server

ik_llama.cpp CPU build of N12 (@5194a9e), tiny random-weight glm5next GGUF of N12 (`<S>/n12/work-ci/ik.gguf`); "gguf-builtin"
(no `--chat-template-file`) rendered chatml, so this GGUF carries no GLM template (PROVISIONAL: inferred from the output). The assistant turn A2 (no reasoning) after A1 (with
reasoning R1) renders `<think></think>A2`: no leak.

```
$ llama-server --version
version: 1 (5194a9e)
built with cc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0 for x86_64-linux-gnu
$ sha256sum chat_template.jinja
5d47ac8e48510b5c6543db4000ef145201823bf46bae9cf51ac6a9553a16ef5a  /home/user/wt-n26a/templates/glm-5.3-flash/chat_template.jinja
$ llama-server -m ik.gguf --port 18591 -c 2048 --parallel 1 --jinja --chat-template-file /home/user/wt-n26a/templates/glm-5.3-flash/chat_template.jinja -t 2   # mode template-file
GET /health -> 200
$ POST /apply-template {"messages":[{"role":"user","content":"Q1"},{"role":"assistant","content":"A1","reasoning_content":"R1"},{"role":"user","content":"Q2"},{"role":"assistant","content":"A2"},{"role":"user","content":"Q3"}]}
{"prompt":"[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>Q1<|assistant|><think>R1</think>A1<|user|>Q2<|assistant|><think></think>A2<|user|>Q3<|assistant|><think>"}
server log lines with "error": 0
$ llama-server -m ik.gguf --port 18591 -c 2048 --parallel 1 --jinja  -t 2   # mode gguf-builtin
GET /health -> 200
$ POST /apply-template {"messages":[{"role":"user","content":"Q1"},{"role":"assistant","content":"A1","reasoning_content":"R1"},{"role":"user","content":"Q2"},{"role":"assistant","content":"A2"},{"role":"user","content":"Q3"}]}
{"prompt":"<|im_start|>user\nQ1<|im_end|>\n<|im_start|>assistant\nA1<|im_end|>\n<|im_start|>user\nQ2<|im_end|>\n<|im_start|>assistant\nA2<|im_end|>\n<|im_start|>user\nQ3<|im_end|>\n<|im_start|>assistant\n"}
server log lines with "error": 0
```

The same conversation rendered by jinja2 (`tools/chattmpl/hf_render.make_env()`, the template file of this tree,
`add_generation_prompt=True`) gave
`"[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>Q1<|assistant|><think>R1</think>A1<|user|>Q2<|assistant|><think></think>A2<|user|>Q3<|assistant|><think>"`
and `jinja2 render == ik render: True` (literal output of the comparison).

Rule 3 against the real llama-server @79e2e74 (CPU build, tiny qwen4exp with the real Qwen vocabulary of N23; random
weights, the 2 generated tokens mean nothing):

```
$ git -C /home/user/wt-n26a log --oneline -1 -- tools/gateway/gateway.py
6305ef3 [engine-integrator] gateway: drop "reasoning_content": null from chat messages (R1 safety rule 3)
$ git -C /home/user/wt-n26a status --short tools/gateway/gateway.py   # (empty = the committed file)
$ llama-server --version
version: 0.6.0-dev (build 1, commit 79e2e74)
llama-server -m qwen4exp_realvocab.gguf -c 2048 -np 1 --jinja: GET /health -> 200
$ B='{"messages":[{"role":"user","content":"Q1"},{"role":"assistant","content":"A1","reasoning_content":null},{"role":"user","content":"Q2"}],"max_tokens":2,"temperature":0}'
$ curl -s http://127.0.0.1:18601/v1/chat/completions -d "$B"     # direct
HTTP 500
{"error":{"code":500,"message":"Failed to parse messages: [json.exception.type_error.302] type must be string, but is null","type":"server_error"}}
$ curl -s http://127.0.0.1:18602/v1/chat/completions -d "$B"     # through the gateway
HTTP 200
{"object": "chat.completion", "usage": {"completion_tokens": 2, "prompt_tokens": 24, "total_tokens": 26, "prompt_tokens_details": {"cached_tokens": 0}}}
$ python3 -c '...' gw.jsonl   # fields of the gateway's request log
{"path": "/v1/chat/completions", "status": 200, "dropped_null_fields": ["messages[1].reasoning_content"], "injected_defaults": [], "error": null}
```

## 5. Tests (literal)

`LLAMA_CPP_GGUF_PY=<S>/llama-ec7630a/gguf-py python3 -m unittest tests.test_r1_orchestration tests.test_r1_rs_rollback_rule tests.test_gateway tests.test_eval_via_gateway tests.test_r1_tools_smoke tests.test_r1_safety_rules -v`
at `856afb4` (the code commits of this branch; this file is added after):

```
test_engine_is_killed_even_when_the_regular_stop_fails (tests.test_r1_orchestration.FailurePaths.test_engine_is_killed_even_when_the_regular_stop_fails) ... ok
test_engine_that_exits_early_fails_health_at_once (tests.test_r1_orchestration.FailurePaths.test_engine_that_exits_early_fails_health_at_once) ... ok
test_exception_inside_a_step_is_recorded_and_the_run_continues (tests.test_r1_orchestration.FailurePaths.test_exception_inside_a_step_is_recorded_and_the_run_continues) ... ok
test_failing_steps_are_recorded_and_the_engine_is_still_stopped (tests.test_r1_orchestration.FailurePaths.test_failing_steps_are_recorded_and_the_engine_is_still_stopped) ... ok
test_health_timeout_fails_cleanly_and_kills_a_stubborn_engine (tests.test_r1_orchestration.FailurePaths.test_health_timeout_fails_cleanly_and_kills_a_stubborn_engine) ... ok
test_help_exit_code_is_checked_against_help_ok_codes (tests.test_r1_orchestration.FailurePaths.test_help_exit_code_is_checked_against_help_ok_codes) ... ok
test_keyboard_interrupt_stops_the_engine_and_skips_the_rest (tests.test_r1_orchestration.FailurePaths.test_keyboard_interrupt_stops_the_engine_and_skips_the_rest) ... ok
test_signal_in_a_shielded_section_is_deferred_not_lost (tests.test_r1_orchestration.FailurePaths.test_signal_in_a_shielded_section_is_deferred_not_lost) ... ok
test_signal_to_the_orchestrator_process_stops_engine_and_tool (tests.test_r1_orchestration.FailurePaths.test_signal_to_the_orchestrator_process_stops_engine_and_tool) ... ok
test_engine_recorded_and_stopped (tests.test_r1_orchestration.FullMockRun.test_engine_recorded_and_stopped) ... ok
test_exit_code_and_layout (tests.test_r1_orchestration.FullMockRun.test_exit_code_and_layout) ... ok
test_journal_has_every_item_in_order_with_outputs (tests.test_r1_orchestration.FullMockRun.test_journal_has_every_item_in_order_with_outputs) ... ok
test_pack_is_not_run_because_of_the_mock (tests.test_r1_orchestration.FullMockRun.test_pack_is_not_run_because_of_the_mock) ... ok
test_prediction_is_written_before_the_engine_starts (tests.test_r1_orchestration.FullMockRun.test_prediction_is_written_before_the_engine_starts) ... ok
test_binary_changed_after_the_build_is_caught (tests.test_r1_orchestration.PinnedBuild.test_binary_changed_after_the_build_is_caught) ... ok
test_binary_outside_the_build_and_missing_build_info_are_caught (tests.test_r1_orchestration.PinnedBuild.test_binary_outside_the_build_and_missing_build_info_are_caught) ... ok
test_engine_templates_are_pinned_and_checked (tests.test_r1_orchestration.PinnedBuild.test_engine_templates_are_pinned_and_checked) ... ok
test_every_mismatch_fails_the_check_and_the_engine_is_not_started (tests.test_r1_orchestration.PinnedBuild.test_every_mismatch_fails_the_check_and_the_engine_is_not_started) ... ok
test_matching_build_is_checked_and_the_engine_starts (tests.test_r1_orchestration.PinnedBuild.test_matching_build_is_checked_and_the_engine_starts) ... ok
test_pin_config_errors_exit_2 (tests.test_r1_orchestration.PinnedBuild.test_pin_config_errors_exit_2) ... ok
test_source_checkout_must_be_at_the_pin (tests.test_r1_orchestration.PinnedBuild.test_source_checkout_must_be_at_the_pin) ... ok
test_config_errors_exit_2 (tests.test_r1_orchestration.Plans.test_config_errors_exit_2) ... ok
test_every_template_substitutes_every_variable_and_preregisters_first (tests.test_r1_orchestration.Plans.test_every_template_substitutes_every_variable_and_preregisters_first) ... ok
test_plan_runs_nothing (tests.test_r1_orchestration.Plans.test_plan_runs_nothing) ... ok
test_a_draft_model_without_a_type_may_infer_one (tests.test_r1_rs_rollback_rule.Rule.test_a_draft_model_without_a_type_may_infer_one)
common/arg.cpp:546-573@79e2e74 infers draft-mtp / dspark / dflash / eagle3 from the draft model. ... ok
test_environment_when_the_flag_is_absent (tests.test_r1_rs_rollback_rule.Rule.test_environment_when_the_flag_is_absent) ... ok
test_no_rollback_snapshots_no_constraint (tests.test_r1_rs_rollback_rule.Rule.test_no_rollback_snapshots_no_constraint) ... ok
test_r1_llamacpp_configs_follow_the_rule (tests.test_r1_rs_rollback_rule.Rule.test_r1_llamacpp_configs_follow_the_rule) ... /home/user/wt-n26a/scripts/r1/configs/ik-llamacpp-server.json: not covered (engine pin 'ik_llama.cpp')
/home/user/wt-n26a/scripts/r1/configs/llamacpp-server-moecache.json: ok (spec types none, draft model no, slots 1)
/home/user/wt-n26a/scripts/r1/configs/llamacpp-server.json: ok (spec types none, draft model no, slots 1)
/home/user/wt-n26a/scripts/r1/configs/mock.json: not covered (engine pin None)
/home/user/wt-n26a/scripts/r1/configs/strata-serve-qwen38fn.json: not covered (engine pin 'strata')
ok
test_snapshot_types_need_one_slot (tests.test_r1_rs_rollback_rule.Rule.test_snapshot_types_need_one_slot) ... ok
test_the_default_slot_count_is_auto_four (tests.test_r1_rs_rollback_rule.Rule.test_the_default_slot_count_is_auto_four) ... ok
test_auth_off_warns_and_allows (tests.test_gateway.Auth.test_auth_off_warns_and_allows) ... ok
test_empty_keys_file_refuses_to_start (tests.test_gateway.Auth.test_empty_keys_file_refuses_to_start) ... ok
test_keys_file_good_bad_missing_and_never_logged (tests.test_gateway.Auth.test_keys_file_good_bad_missing_and_never_logged) ... ok
test_apply_defaults_unit (tests.test_gateway.BodyFidelity.test_apply_defaults_unit) ... ok
test_byte_identical_forwarding_and_same_prefix_cache_hits (tests.test_gateway.BodyFidelity.test_byte_identical_forwarding_and_same_prefix_cache_hits) ... ok
test_set_default_injected_only_when_absent (tests.test_gateway.BodyFidelity.test_set_default_injected_only_when_absent) ... ok
test_help_exits_zero (tests.test_gateway.CLIHelp.test_help_exits_zero) ... ok
test_chat_guard_uses_apply_template (tests.test_gateway.ContextGuard.test_chat_guard_uses_apply_template) ... ok
test_chat_without_apply_template_passes_unchecked (tests.test_gateway.ContextGuard.test_chat_without_apply_template_passes_unchecked) ... ok
test_pass_through_when_backend_has_no_tokenize (tests.test_gateway.ContextGuard.test_pass_through_when_backend_has_no_tokenize) ... ok
test_reject_completion_over_limit_before_engine (tests.test_gateway.ContextGuard.test_reject_completion_over_limit_before_engine) ... ok
test_disconnect_while_engine_prefills_closes_upstream (tests.test_gateway.Disconnect.test_disconnect_while_engine_prefills_closes_upstream) ... ok
test_stream_disconnect_closes_upstream (tests.test_gateway.Disconnect.test_stream_disconnect_closes_upstream) ... ok
test_stream_disconnect_detected_by_failed_write_when_watch_is_off (tests.test_gateway.Disconnect.test_stream_disconnect_detected_by_failed_write_when_watch_is_off) ... ok
test_dead_backend_502_and_health_503 (tests.test_gateway.Failures.test_dead_backend_502_and_health_503) ... ok
test_http500_before_first_byte_maps_to_502 (tests.test_gateway.Failures.test_http500_before_first_byte_maps_to_502) ... ok
test_truncated_stream_ends_with_error_event_and_no_done (tests.test_gateway.Failures.test_truncated_stream_ends_with_error_event_and_no_done) ... ok
test_jsonl_fields_and_no_prompt_text (tests.test_gateway.LogAndMetrics.test_jsonl_fields_and_no_prompt_text) ... ok
test_log_prompts_opt_in (tests.test_gateway.LogAndMetrics.test_log_prompts_opt_in) ... ok
test_metrics_text_parses (tests.test_gateway.LogAndMetrics.test_metrics_text_parses) ... ok
test_completions_untouched_and_switch_off (tests.test_gateway.NullReasoningContent.test_completions_untouched_and_switch_off) ... ok
test_drop_unit (tests.test_gateway.NullReasoningContent.test_drop_unit) ... ok
test_null_is_cut_out_and_every_other_byte_kept (tests.test_gateway.NullReasoningContent.test_null_is_cut_out_and_every_other_byte_kept) ... ok
test_requests_without_null_are_forwarded_byte_for_byte (tests.test_gateway.NullReasoningContent.test_requests_without_null_are_forwarded_byte_for_byte) ... ok
test_models_and_health (tests.test_gateway.Passthrough.test_models_and_health) ... ok
test_non_stream_chat_and_completions_equal_direct (tests.test_gateway.Passthrough.test_non_stream_chat_and_completions_equal_direct) ... ok
test_stream_passthrough_equals_direct_and_is_not_buffered (tests.test_gateway.Passthrough.test_stream_passthrough_equals_direct_and_is_not_buffered) ... ok
test_429_when_queue_full (tests.test_gateway.Queueing.test_429_when_queue_full) ... ok
test_fifo_order (tests.test_gateway.Queueing.test_fifo_order) ... ok
test_queued_client_disconnect_leaves_queue (tests.test_gateway.Queueing.test_queued_client_disconnect_leaves_queue) ... ok
test_drain_timeout_reports_false (tests.test_gateway.Shutdown.test_drain_timeout_reports_false) ... ok
test_graceful_shutdown_drains_in_flight (tests.test_gateway.Shutdown.test_graceful_shutdown_drains_in_flight) ... ok
test_sigterm_cli_drains_and_exits_zero (tests.test_gateway.Shutdown.test_sigterm_cli_drains_and_exits_zero) ... ok
test_auth_failure_is_a_request_error_not_a_model_failure (tests.test_eval_via_gateway.ViaGateway.test_auth_failure_is_a_request_error_not_a_model_failure) ... ok
test_stand_through_gateway_equals_direct (tests.test_eval_via_gateway.ViaGateway.test_stand_through_gateway_equals_direct) ... ok
test_wrong_answers_still_fail_through_gateway (tests.test_eval_via_gateway.ViaGateway.test_wrong_answers_still_fail_through_gateway) ... ok
test_both_lengths_plan_with_the_template_engine (tests.test_r1_tools_smoke.Derive.test_both_lengths_plan_with_the_template_engine) ... ok
test_cli (tests.test_r1_tools_smoke.Derive.test_cli) ... ok
test_cli (tests.test_r1_safety_rules.CommittedConfigs.test_cli) ... ok
test_every_committed_config_passes (tests.test_r1_safety_rules.CommittedConfigs.test_every_committed_config_passes) ... ok
test_every_llama_server_argv_gives_np_1 (tests.test_r1_safety_rules.CommittedConfigs.test_every_llama_server_argv_gives_np_1) ... ok
test_non_mock_configs_have_a_manifest_or_an_unverified_reason (tests.test_r1_safety_rules.CommittedConfigs.test_non_mock_configs_have_a_manifest_or_an_unverified_reason) ... ok
test_only_the_mock_declares_a_synthetic_model (tests.test_r1_safety_rules.CommittedConfigs.test_only_the_mock_declares_a_synthetic_model)
Rule 10: the SYNTHETIC marker of tools/verify/pack.py, "mock": true or a synthetic_run criteria file. ... ok
test_operational_rules_are_written_into_the_configs (tests.test_r1_safety_rules.CommittedConfigs.test_operational_rules_are_written_into_the_configs) ... ok
test_resolved_by_run_r1_they_pass_too_and_the_template_sha_is_recorded (tests.test_r1_safety_rules.CommittedConfigs.test_resolved_by_run_r1_they_pass_too_and_the_template_sha_is_recorded) ... ok
test_template_sha256_is_the_file_on_disk (tests.test_r1_safety_rules.CommittedConfigs.test_template_sha256_is_the_file_on_disk) ... ok
test_glm_template (tests.test_r1_safety_rules.Mutations.test_glm_template) ... ok
test_glm_template_changed_on_disk (tests.test_r1_safety_rules.Mutations.test_glm_template_changed_on_disk) ... ok
test_mimo_speculation (tests.test_r1_safety_rules.Mutations.test_mimo_speculation) ... ok
test_np (tests.test_r1_safety_rules.Mutations.test_np) ... ok
test_qwen_mtp_files (tests.test_r1_safety_rules.Mutations.test_qwen_mtp_files) ... ok
test_synthetic (tests.test_r1_safety_rules.Mutations.test_synthetic) ... ok
test_weights (tests.test_r1_safety_rules.Mutations.test_weights) ... ok
test_the_resolved_config_is_checked (tests.test_r1_safety_rules.RunR1Refuses.test_the_resolved_config_is_checked)
--set values are part of what is checked: a variable that turns speculation on with -np 2 is refused. ... ok
test_violating_configs_are_refused_before_anything (tests.test_r1_safety_rules.RunR1Refuses.test_violating_configs_are_refused_before_anything) ... ok
test_disabled_verification_is_refused_and_unselected_is_added (tests.test_r1_safety_rules.WeightsGate.test_disabled_verification_is_refused_and_unselected_is_added) ... ok
test_failed_verification_blocks_the_engine (tests.test_r1_safety_rules.WeightsGate.test_failed_verification_blocks_the_engine) ... ok
test_the_mock_is_exempt (tests.test_r1_safety_rules.WeightsGate.test_the_mock_is_exempt) ... ok
test_unverified_reason_lets_the_engine_start_and_is_recorded (tests.test_r1_safety_rules.WeightsGate.test_unverified_reason_lets_the_engine_start_and_is_recorded) ... ok
test_verified_weights_start_the_engine (tests.test_r1_safety_rules.WeightsGate.test_verified_weights_start_the_engine) ... ok

----------------------------------------------------------------------
Ran 90 tests in 17.527s

OK
```

Local replay of the new `server-cpu.yml` step (script `<S>/n26a/ci_step.sh` = the step's three lines):

```
$ bash <S>/n26a/ci_step.sh <S>/n26a/r1-safety.log; echo "rc=$?"   # local replay of the server-cpu.yml step (LLAMA_CPP_GGUF_PY unset)
::notice::R1 safety rules: Ran 57 tests in 4.287s OK 
rc=0
```

Also run (unchanged modules that read the configs or the CI files):
`python3 -m unittest tests.test_ik_glm5next_parity tests.test_glm5next_f16fa_parity tests.test_srvcheck tests.test_bench tests.test_check_unittest tests.test_chat_templates tests.test_glm_template_n23`
-> `Ran 245 tests in 74.021s` / `OK (skipped=136)` (the skips are the dump-parity tests without their environment, by
design). The full `unittest discover`: NOT RUN in this session.

## 6. Mutation check

`<S>/n26a/mutate.py` (copy: `docs/evidence/mutation-scripts/2026-10-12-n26a-mutations.py`): one mutation at a time in the worktree file, the six modules above, the file restored in `finally`
(`git status` clean afterwards). Round 1 (the first version of the tests):

```
M1 R-NP: explicit -np not required                           rc=1 Ran 90 tests in 18.016s FAILED (failures=10)
      test_cli (tests.test_r1_safety_rules.CommittedConfigs.test_cli)
      test_np (tests.test_r1_safety_rules.Mutations.test_np)
      test_violating_configs_are_refused_before_anything (tests.test_r1_safety_rules.RunR1Refuses.test_violating_configs_are_refused_before_anything)
M2 R-NP: rs_rollback_rule.check() not used                   rc=0 Ran 90 tests in 17.983s OK
M3 R-NP: 'any speculation' part dropped                      rc=1 Ran 90 tests in 18.433s FAILED (failures=1)
      test_np (tests.test_r1_safety_rules.Mutations.test_np)
M4 R-GLM-TEMPLATE: --chat-template-file not required         rc=1 Ran 90 tests in 18.541s FAILED (failures=7)
      test_glm_template (tests.test_r1_safety_rules.Mutations.test_glm_template)
      test_violating_configs_are_refused_before_anything (tests.test_r1_safety_rules.RunR1Refuses.test_violating_configs_are_refused_before_anything)
M5 R-GLM-TEMPLATE: path not compared                         rc=0 Ran 90 tests in 18.138s OK
M6 R-GLM-TEMPLATE: sha256 not compared                       rc=1 Ran 90 tests in 17.964s FAILED (failures=2)
      test_glm_template (tests.test_r1_safety_rules.Mutations.test_glm_template)
      test_glm_template_changed_on_disk (tests.test_r1_safety_rules.Mutations.test_glm_template_changed_on_disk)
M7 R-MTP-FILES: -shared- accepted                            rc=0 Ran 90 tests in 17.988s OK
M8 R-MTP-FILES: any basename accepted                        rc=1 Ran 90 tests in 17.762s FAILED (failures=1)
      test_qwen_mtp_files (tests.test_r1_safety_rules.Mutations.test_qwen_mtp_files)
M9 R-MTP-FILES: -hfd accepted                                rc=1 Ran 90 tests in 18.213s FAILED (failures=1)
      test_qwen_mtp_files (tests.test_r1_safety_rules.Mutations.test_qwen_mtp_files)
M10 R-MIMO-MTP: rule off                                     rc=1 Ran 90 tests in 18.197s FAILED (failures=1)
      test_mimo_speculation (tests.test_r1_safety_rules.Mutations.test_mimo_speculation)
M11 R-WEIGHTS: rule off                                      rc=1 Ran 90 tests in 18.572s FAILED (failures=4)
      test_disabled_verification_is_refused_and_unselected_is_added (tests.test_r1_safety_rules.WeightsGate.test_disabled_verification_is_refused_and_unselected_is_added)
      test_violating_configs_are_refused_before_anything (tests.test_r1_safety_rules.RunR1Refuses.test_violating_configs_are_refused_before_anything)
      test_weights (tests.test_r1_safety_rules.Mutations.test_weights)
M12 R-SYNTH: rule off                                        rc=1 Ran 90 tests in 18.417s FAILED (failures=1)
      test_synthetic (tests.test_r1_safety_rules.Mutations.test_synthetic)
M13 run_r1: violations not refused                           rc=1 Ran 90 tests in 19.394s FAILED (failures=10)
      test_disabled_verification_is_refused_and_unselected_is_added (tests.test_r1_safety_rules.WeightsGate.test_disabled_verification_is_refused_and_unselected_is_added)
      test_the_resolved_config_is_checked (tests.test_r1_safety_rules.RunR1Refuses.test_the_resolved_config_is_checked)
      test_violating_configs_are_refused_before_anything (tests.test_r1_safety_rules.RunR1Refuses.test_violating_configs_are_refused_before_anything)
M14 run_r1: weights gate open                                rc=1 Ran 90 tests in 17.856s FAILED (failures=1)
      test_failed_verification_blocks_the_engine (tests.test_r1_safety_rules.WeightsGate.test_failed_verification_blocks_the_engine)
M15 run_r1: --only engine_start does not add weights_verify  rc=1 Ran 90 tests in 17.263s FAILED (failures=4)
      test_disabled_verification_is_refused_and_unselected_is_added (tests.test_r1_safety_rules.WeightsGate.test_disabled_verification_is_refused_and_unselected_is_added)
      test_failed_verification_blocks_the_engine (tests.test_r1_safety_rules.WeightsGate.test_failed_verification_blocks_the_engine)
      test_unverified_reason_lets_the_engine_start_and_is_recorded (tests.test_r1_safety_rules.WeightsGate.test_unverified_reason_lets_the_engine_start_and_is_recorded)
      test_verified_weights_start_the_engine (tests.test_r1_safety_rules.WeightsGate.test_verified_weights_start_the_engine)
M16 gateway: null not dropped                                rc=1 Ran 90 tests in 17.627s FAILED (failures=1)
      test_null_is_cut_out_and_every_other_byte_kept (tests.test_gateway.NullReasoningContent.test_null_is_cut_out_and_every_other_byte_kept)
M17 gateway: whole body re-serialised (bytes not kept)       rc=1 Ran 90 tests in 17.872s FAILED (failures=6)
      test_drop_unit (tests.test_gateway.NullReasoningContent.test_drop_unit)
      test_null_is_cut_out_and_every_other_byte_kept (tests.test_gateway.NullReasoningContent.test_null_is_cut_out_and_every_other_byte_kept)
```

Survivors of round 1 and what was added to `tests/test_r1_safety_rules.py`:
- M2 (rs_rollback_rule.check() not called): the broader "any speculation" part caught every case; added: the message of a
  `--spec-type draft-mtp -np 2` config must be the rs_rollback_rule one (`n_rs_seq > 0`, `(scripts/r1/rs_rollback_rule.py)`).
- M5 (template path not compared): the wrong paths tested did not exist, so the sha check refused them; added: a byte-equal
  copy of the template at another path must still be refused with the path message.
- M7 (`-shared-` accepted): the basename pattern caught it; added: the `-shared-` message (N18 reason) must be present.

Round 2 (final tests), 17/17 mutations killed:

```
M1 R-NP: explicit -np not required                           rc=1 Ran 90 tests in 18.049s FAILED (failures=10)
      test_cli (tests.test_r1_safety_rules.CommittedConfigs.test_cli)
      test_np (tests.test_r1_safety_rules.Mutations.test_np)
      test_violating_configs_are_refused_before_anything (tests.test_r1_safety_rules.RunR1Refuses.test_violating_configs_are_refused_before_anything)
M2 R-NP: rs_rollback_rule.check() not used                   rc=1 Ran 90 tests in 17.553s FAILED (failures=1)
      test_np (tests.test_r1_safety_rules.Mutations.test_np)
M3 R-NP: 'any speculation' part dropped                      rc=1 Ran 90 tests in 17.730s FAILED (failures=1)
      test_np (tests.test_r1_safety_rules.Mutations.test_np)
M4 R-GLM-TEMPLATE: --chat-template-file not required         rc=1 Ran 90 tests in 17.986s FAILED (failures=7)
      test_glm_template (tests.test_r1_safety_rules.Mutations.test_glm_template)
      test_violating_configs_are_refused_before_anything (tests.test_r1_safety_rules.RunR1Refuses.test_violating_configs_are_refused_before_anything)
M5 R-GLM-TEMPLATE: path not compared                         rc=1 Ran 90 tests in 17.750s FAILED (failures=1)
      test_glm_template (tests.test_r1_safety_rules.Mutations.test_glm_template)
M6 R-GLM-TEMPLATE: sha256 not compared                       rc=1 Ran 90 tests in 17.423s FAILED (failures=2)
      test_glm_template (tests.test_r1_safety_rules.Mutations.test_glm_template)
      test_glm_template_changed_on_disk (tests.test_r1_safety_rules.Mutations.test_glm_template_changed_on_disk)
M7 R-MTP-FILES: -shared- accepted                            rc=1 Ran 90 tests in 17.617s FAILED (failures=1)
      test_qwen_mtp_files (tests.test_r1_safety_rules.Mutations.test_qwen_mtp_files)
M8 R-MTP-FILES: any basename accepted                        rc=1 Ran 90 tests in 17.877s FAILED (failures=1)
      test_qwen_mtp_files (tests.test_r1_safety_rules.Mutations.test_qwen_mtp_files)
M9 R-MTP-FILES: -hfd accepted                                rc=1 Ran 90 tests in 17.651s FAILED (failures=1)
      test_qwen_mtp_files (tests.test_r1_safety_rules.Mutations.test_qwen_mtp_files)
M10 R-MIMO-MTP: rule off                                     rc=1 Ran 90 tests in 17.691s FAILED (failures=1)
      test_mimo_speculation (tests.test_r1_safety_rules.Mutations.test_mimo_speculation)
M11 R-WEIGHTS: rule off                                      rc=1 Ran 90 tests in 18.142s FAILED (failures=4)
      test_disabled_verification_is_refused_and_unselected_is_added (tests.test_r1_safety_rules.WeightsGate.test_disabled_verification_is_refused_and_unselected_is_added)
      test_violating_configs_are_refused_before_anything (tests.test_r1_safety_rules.RunR1Refuses.test_violating_configs_are_refused_before_anything)
      test_weights (tests.test_r1_safety_rules.Mutations.test_weights)
M12 R-SYNTH: rule off                                        rc=1 Ran 90 tests in 17.810s FAILED (failures=1)
      test_synthetic (tests.test_r1_safety_rules.Mutations.test_synthetic)
M13 run_r1: violations not refused                           rc=1 Ran 90 tests in 19.375s FAILED (failures=10)
      test_disabled_verification_is_refused_and_unselected_is_added (tests.test_r1_safety_rules.WeightsGate.test_disabled_verification_is_refused_and_unselected_is_added)
      test_the_resolved_config_is_checked (tests.test_r1_safety_rules.RunR1Refuses.test_the_resolved_config_is_checked)
      test_violating_configs_are_refused_before_anything (tests.test_r1_safety_rules.RunR1Refuses.test_violating_configs_are_refused_before_anything)
M14 run_r1: weights gate open                                rc=1 Ran 90 tests in 17.748s FAILED (failures=1)
      test_failed_verification_blocks_the_engine (tests.test_r1_safety_rules.WeightsGate.test_failed_verification_blocks_the_engine)
M15 run_r1: --only engine_start does not add weights_verify  rc=1 Ran 90 tests in 17.668s FAILED (failures=4)
      test_disabled_verification_is_refused_and_unselected_is_added (tests.test_r1_safety_rules.WeightsGate.test_disabled_verification_is_refused_and_unselected_is_added)
      test_failed_verification_blocks_the_engine (tests.test_r1_safety_rules.WeightsGate.test_failed_verification_blocks_the_engine)
      test_unverified_reason_lets_the_engine_start_and_is_recorded (tests.test_r1_safety_rules.WeightsGate.test_unverified_reason_lets_the_engine_start_and_is_recorded)
      test_verified_weights_start_the_engine (tests.test_r1_safety_rules.WeightsGate.test_verified_weights_start_the_engine)
M16 gateway: null not dropped                                rc=1 Ran 90 tests in 17.571s FAILED (failures=1)
      test_null_is_cut_out_and_every_other_byte_kept (tests.test_gateway.NullReasoningContent.test_null_is_cut_out_and_every_other_byte_kept)
M17 gateway: whole body re-serialised (bytes not kept)       rc=1 Ran 90 tests in 17.876s FAILED (failures=6)
      test_drop_unit (tests.test_gateway.NullReasoningContent.test_drop_unit)
      test_null_is_cut_out_and_every_other_byte_kept (tests.test_gateway.NullReasoningContent.test_null_is_cut_out_and_every_other_byte_kept)
```

| # | mutation | killed by (round 2) |
|---|---|---|
| M1 | R-NP: explicit -np not required | 10 failures (test_cli, test_np, refusal) |
| M2 | R-NP: rs_rollback_rule.check() not used | test_np |
| M3 | R-NP: "any speculation" part dropped | test_np |
| M4 | R-GLM-TEMPLATE: flag not required | 7 failures |
| M5 | R-GLM-TEMPLATE: path not compared | test_glm_template |
| M6 | R-GLM-TEMPLATE: sha256 not compared | test_glm_template, test_glm_template_changed_on_disk |
| M7 | R-MTP-FILES: -shared- accepted | test_qwen_mtp_files |
| M8 | R-MTP-FILES: any basename accepted | test_qwen_mtp_files |
| M9 | R-MTP-FILES: -hfd accepted | test_qwen_mtp_files |
| M10 | R-MIMO-MTP off | test_mimo_speculation |
| M11 | R-WEIGHTS off | 4 failures |
| M12 | R-SYNTH off | test_synthetic |
| M13 | run_r1 does not refuse violations | 10 failures |
| M14 | weights gate open | test_failed_verification_blocks_the_engine |
| M15 | --only engine_start does not add weights_verify | 4 failures |
| M16 | gateway does not drop the null | test_null_is_cut_out_and_every_other_byte_kept |
| M17 | gateway re-serialises the whole body | 6 failures |

## 7. NOT RUN / UNKNOWN

- Any GPU run: no template was started on the 4x4080 server; `--chat-template-file` with the CUDA builds of llama.cpp
  @79e2e74 and ik_llama.cpp @5194a9e (and their `--help`, step `engine_version`) NOT RUN.
- The N23 corpus (21 conversations x 7 switches) through ik_llama.cpp with our template: NOT RUN (one conversation above).
- The `server-cpu.yml` and `ci.yml` workflows on GitHub with these commits: NOT RUN (the branch is not pushed by this
  task).
- Whether ik_llama.cpp honours `LLAMA_ARG_*` environment variables for `-np` / draft models: UNKNOWN; `R-NP` therefore
  requires the flag in argv.
- Speculation inside engines other than llama-server is recognised only by `--spec N` / `--mtp` arguments and
  `mode.spec_k` (Strata's syntax); other engines' flags: UNKNOWN to the check.
