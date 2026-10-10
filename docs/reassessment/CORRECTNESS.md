# CORRECTNESS — инвентарь инфраструктуры проверки корректности (2026-10-10)

Роль: parity-qa (deliverable F). Документ перечисляет **все** существующие сегодня проверки по шести уровням, которые задал
заказчик, — что именно сравнивается, с каким допуском, на какой архитектуре, в какой ветке и каков текущий статус с
доказательством; затем — проверки, ждущие GPU/сервер, и пробелы, которые можно закрыть на CPU уже сейчас.
Метки — по `AGENTS.md` «Evidence rules»: **CONFIRMED** (файл:строка@коммит или литеральный вывод из раздела 4), **PROVISIONAL**
(как получено), **UNKNOWN**, **NOT RUN**. Ничего не исполнялось на GPU.

## 0. Источники и как проверялось

| ветка | рабочее дерево | HEAD при проверке | что в ней сверх других |
|---|---|---|---|
| `claude/research-landscape` | `/home/user/wt-research` | `a2c1815` (первый прогон набора — на `7e534c9` + тогда ещё не закоммиченный `tests/test_merge_matrix.py`; второй и шаги `cpu-parity` — на `a2c1815`) | всё PR #1 + исследования, `tools/ceilings/`, `scripts/traces/smoke_tiny.sh` (шаг CI с qwen3moe), `tools/research/` |
| `claude/n7-server-hybrid-cpu` | `/home/user/408032gb` | `090ca71` (первый прогон набора — на `986519a`, до слияния PR #1) | N7: `tools/srvcheck/`, `.github/workflows/server-cpu.yml`, `glm5next_tiny.py --mtp` |
| `claude/n8-session-store` | `/home/user/wt-n8` | `43c2171` (первый прогон набора и шаги `server-cpu` — на `ebecad4`; затем влиты N7 и PR #1) | N7 + N8: `tools/kv/session_store.py`, `slot_corruption_probe.py`, шаг N8 в `server-cpu` |

Общая база research и n7/n8 — `b6f4fe3` (коммит PR #1); n7 и n8 во время инвентаризации влили голову PR #1 `259081e`, которой в
research нет (CONFIRMED: `git merge-base`, `git merge-base --is-ancestor`, раздел 4.1).
Ссылки вида `ci.yml:NN` — на `.github/workflows/ci.yml`@a2c1815 (на n7/n8 файл тот же, кроме шага qwen3moe, `ci.yml:36-39`);
`server-cpu.yml` есть только на n7/n8, ссылки — @ebecad4.

Статусы CI взяты из GitHub API (`gh run list`, `gh run view --json jobs`, аннотации check-run; литерально — 4.2). Логи заданий
через прокси песочницы недоступны (`failed to get run log: … Forbidden`), поэтому из CI цитируются только статусы шагов и аннотации.
Локальные прогоны (4.3–4.5) — песочница: Intel Xeon 4 ядра (AVX2 и AVX-512 есть, сборка ggml с `GGML_NATIVE=OFF` даёт путь AVX2
без AVX-512), Python 3.13.16, numpy 2.5.3, без GPU, без nvcc, без torch.

**Итог по уровням** (число проверок — строк таблиц раздела 1; одна строка = один шаг CI или один ручной инструмент):

| уровень | проверок | из них исполняются в CI | исполнено на целевом железе | статус |
|---|---|---|---|---|
| 1. математика (оракулы против эталона) | 12 | 11 (1 — ручной эталон DeepSeek) | — | CI зелёный; повторено локально 10 из 11 автоматических (L1-11 — частично, без torch) |
| 2. CPU-функциональные (настоящий граф/движок на CPU) | 9 | 9 (L2-3 — workflow `traces-real`, последний завершённый прогон — failure, повтор идёт) | — | CI зелёный, кроме L2-3; повторено локально L2-1, L2-2, L2-4, L2-5 |
| 3. компиляция CUDA | 6 | 5 (L3-1 пустой по построению; L3-5 в CI пропускается) | — | CI зелёный |
| 4. исполнение CUDA | 0 | 0 | 0 | **NOT RUN** (GPU нет нигде, где мы исполняем) |
| 5. производительность GPU | 0 | 0 | 0 | **NOT RUN** |
| 6. производительность на целевой машине | 0 | 0 | 0 | **NOT RUN** (R0 не начат) |

Плюс опорные тесты инструментов (mock/синтетика, не уровень). Полный набор `unittest` на HEAD при написании: research 522, n7 461,
n8 479 — везде OK, по 86 пропусков (76 — тесты паритета без дампов, их гоняют отдельные шаги; причины — 4.3).

**Главное.**
1. Проверки уровней 1–3 в CI зелёные (кроме `traces-real`, L2-3); автоматические проверки уровня 1 и L2-1, L2-2, L2-4, L2-5
   повторены здесь же на CPU с теми же числами, что в evidence (CONFIRMED, 4.3–4.5).
   Уровни 4–6 — **NOT RUN**: GPU нет ни в песочнице, ни на раннерах, к серверу доступа нет; собственного GPU-кода у проекта нет
   (шаг `cuda-compile` пуст по построению).
2. Все сверки оракулов с графом llama.cpp сделаны на пине `ec7630a`, а продукт пойдёт на `79e2e74`. Проба этой инвентаризации:
   на `79e2e74` G1/MiMo/N2 проходят, N6 — 52/54 (падают только сравнения строк-заполнителей индексера; всё значимое совпадает)
   (CONFIRMED, 4.7). На момент инвентаризации в CI такой проверки не было — пробел 1; закрыт в тот же день (workflow
   `parity-r1-pin.yml`, N6 54/54 на `79e2e74`, `docs/evidence/2026-10-10-parity-r1-pin.md`).
3. Ни одна проверка не касается квантованных весов, KV f16 и flash attention — а именно так будет работать сервер (пробелы 2–3);
   для GLM нет сверки декода, отката и MTP (пробел 4).

## 1. Шесть уровней проверки

Обозначения: «CI» — последний прогон GitHub Actions, который виден через `gh run list`/`gh run view` (раздел 4.2); «локально» —
прогон этой инвентаризации в песочнице (раздел 4.3–4.5). Числа тестов — `def test_` в файле (статический подсчёт на research `a2c1815` и n8 `ebecad4`; CONFIRMED `grep -cE '^\s+def test_'`),
если не сказано иначе. Метрика паритета во всех тестах G1/N2/N6 —
нормированная относительная ошибка `max|got − want| / max|want|` по тензору (`tests/test_glm_tiny_parity.py:68`@a2c1815,
`tests/test_mimo_tiny_parity.py:32`@a2c1815, `TOL = 1e-4` в `tests/test_qwen4exp_tiny_parity.py:38`@a2c1815). Эталонный
прогон llama.cpp — всегда f32-веса, f32-кэши, flash attention выключен (`-fa off -ctk f32 -ctv f32`:
`tools/tinygen/run_parity.sh:16`, `run_parity_mimo.sh:13`, `run_parity_qwen4exp.sh:17`@a2c1815).

Последние прогоны CI, на которые ссылаются колонки «статус» (CONFIRMED: `gh run list`/`gh run view`, раздел 4.2; «in_progress» —
на момент проверки 2026-10-10T20:16Z):

| ветка / workflow | `ci` | `server-cpu` | прочее |
|---|---|---|---|
| research `a2c1815` | 38081953672 (push) и 38081957679 (PR) — success, все 6 джобов и все шаги success | — (workflow в ветке нет) | `traces-real`: 38076798213 (`8981351`) — failure; 38079738013 (`7a0bf02`) — in_progress (шаг 8 из 9) |
| n7 `986519a` → `090ca71` | 38076450758 (`986519a`) и 38082292627 (`090ca71`, после слияния PR #1) — success | 38076450727 (`986519a`, push) и 38082295327 (`090ca71`, PR) — success | — |
| n8 `ebecad4` → `43c2171` | 38081791581 (`ebecad4`) и 38082626725 (`43c2171`) — success | 38081791575 (`ebecad4`, push; job `n7` 114300045445: все шаги success) и 38082629418 (`43c2171`, PR) — success; 38082585007 (`750fe3d`, PR) — cancelled (вытеснен следующим push) | — |
| `engines-sm89` | — | — | последний: 38080354794 (`claude/relaxed-dijkstra-1d7i5o` = голова PR #1 `259081e`) — success, 4 из 4 движков; перед ним 38076515851 (`ca1d2a7`) — success |

### Уровень 1 — математическая корректность (оракулы numpy/torch против эталона)

| id | проверка | файл / CI (workflow → job → step) | что с чем сравнивается, допуск | архитектура | ветки | статус |
|---|---|---|---|---|---|---|
| L1-1 | свойства оракула KDA/GDN | `tests/test_kda.py` (7); `ci` → `cpu-parity` → `unit tests` (`ci.yml:17-18`) | `ref/kda.py` против частных случаев: KDA с постоянным по каналам гейтом = скалярный GDN, разрез последовательности с переносом состояния = целая, матричная форма, ассоциативная память, плиточное отображение голов `h % H_k`, l2-норма = форма llama.cpp; rtol 1e-12…1e-6 | KDA (glm5-next), GDN (qwen4exp) | все три | CI зелёный; локально в составе полного набора (4.3) |
| L1-2 | оракул KDA против **настоящей** операции ggml | `tests/test_kda_vs_ggml.py` (3) + `tests/ggml_parity/gdn_harness.c`; `ci` → `cpu-parity` → `ggml delta-net parity (CPU)` (`ci.yml:19-25`) | выход и конечное состояние `ggml_gated_delta_net` (CPU, llama.cpp `ec7630a`) против `ref/kda.py`: гейт по каналам, скалярный гейт, геометрия голов GLM; `rtol 1e-4, atol 1e-5` (`tests/test_kda_vs_ggml.py:54-55`) | KDA/GDN | все три | CI зелёный; локально 18/18 OK (вместе с `test_fitplan`, 4.4) |
| L1-3 | свойства роутера, SwiGLU-clamp, mHC | `tests/test_router_mhc.py` (6); `unit tests` | bias только выбирает, но не взвешивает; сумма весов = 2.5; clamp до активации; Sinkhorn: по src точно (1e-5), по dst ≤ 0.05 | glm5-next | все три | CI зелёный; локально (4.3) |
| L1-4 | **G1 GLM**: оракулы против дампов графа `glm5-next` | `tests/test_glm_tiny_parity.py` (8) ← `tools/tinygen/run_parity.sh`; `ci` → `cpu-parity` → `G1 tiny-model parity against the real glm5-next graph (CPU)` (`ci.yml:26-27`) | крошечная модель 12 слоёв (KDA на 9, DSA на 3, MoE на 11, 8 экспертов top-2, clamp 0.5), 64 токена, **один ubatch префилла**. Тензоры (имена `cb()`): `hc_post`, `hc_comb`, `hc_attn_pre`, `hc_attn_post` (1e-5); `kda_g1`, `kda_{q,k,v}_conv` (1e-5); `kda_scan_out` (1e-4); `indexer_k`, `indexer_pool_k_new`, `indexer_q`, `indexer_score` (1e-4); `kq_mask_dsa` — выбор с допуском ничьих; `q_resid`, `q_absorbed`, `kv_cmpr` (1e-5); `kqv_out`, `attn_out` (1e-4); `ffn_moe_topk` (равенство множеств), `ffn_moe_weights_scaled` (Δ ≤ 1e-5, `:137`), `ffn_moe_swiglu_limited` (1e-5, clamp срабатывает). Сквозных логитов, общего эксперта, `ffn_out`, выхода — нет | glm5-next @ec7630a | все три | CI зелёный; локально 8/8 OK; тест ошибок не печатает — сняты вспомогательным скриптом: худшая rel 5.323e-07 (`hc_comb`, слой 10), все ≥ 18× внутри допусков (4.4) |
| L1-5 | **G1 MiMo**: оракулы против графа `mimo2` | `tests/test_mimo_tiny_parity.py` (5) ← `run_parity_mimo.sh`; `ci` → `cpu-parity` → `MiMo G1 tiny-model parity against the real mimo2 graph (CPU)` (`ci.yml:28-29`) | 6 слоёв (глобальные 0 и 5, SWA 1–4 с окном 8), 40 токенов: `Qcur`/`Kcur` до и после частичного NEOX RoPE (1e-5); `kqv_out`, `attn_out` со SWA + sinks + GQA (1e-5); страж «без sinks ошибка > 1e-3»; `ffn_moe_topk`, `ffn_moe_weights_norm` (Δ ≤ 1e-5); `ffn_moe_swiglu` (1e-5) | mimo2 @ec7630a | все три | CI зелёный; локально 5/5 OK (4.4) |
| L1-6 | свойства оракула qwen4exp | `tests/test_qwen4exp_ref.py` (16); `unit tests` | сведение к известным случаям и алгебраические тождества (`ref/qwen4exp.py`) | qwen4exp | все три | CI зелёный; локально (4.3) |
| L1-7 | **N2 qwen4exp**: компоненты + сквозной прогон | `tests/test_qwen4exp_tiny_parity.py` (17) ← `run_parity_qwen4exp.sh`; `ci` → `cpu-parity` → `N2 qwen4exp (candidate 1) tiny-model parity …` (`ci.yml:30-31`) | 8 слоёв (GDN 0–2, 4–6; PLE на 1; QSA 3; плотное внимание 7; MoE + общий эксперт везде), 64 токена с EOS; каждый компонент из дампов + оракул от id токенов: `l_last` всех слоёв, `result_norm`, логиты; TOL 1e-4, копии (эмбеддинг, строки PLE) и id — точно; таблица max abs/rel по слоям печатается | qwen4exp @ec7630a | все три | CI зелёный; локально 17/17 OK, худшая — логиты 7.439e-07 (4.4) |
| L1-8 | инкрементальный путь оракула qwen4exp | `tests/test_qwen4exp_chunked_ref.py` (8); `unit tests` и шаг N6 | куски (несколько ubatch, по одному токену) = целая последовательность; откат `cache_truncate` = состояние префикса; допуск 1e-10 | qwen4exp | все три | CI зелёный; локально (4.3, 4.4) |
| L1-9 | **N6 qwen4exp**: несколько ubatch, декод, откат | `tests/test_qwen4exp_decode_parity.py` (10 × 4 прогона = 40) ← `run_parity_qwen4exp_decode.sh`; `ci` → `cpu-parity` → `N6 qwen4exp several ubatches, token-by-token decode, rollback and MTP …` (`ci.yml:32-33`) | прогоны `ub16`, `ub10`, `decode` (48 + 16×1), `rollback` (`n_rs_seq` 3, черновик 4, откат 2); каждая строка каждого графа против полного оракула [full] и инкрементального [chunk], плюс переносимое состояние: `conv_state_at`, `state_predelta`, `new_state`, `indexer_k`; TOL 1e-4 | qwen4exp @ec7630a | все три | CI зелёный; локально 54/54 OK (вместе с L1-8, L1-10), худшая 5.422e-06 (`rollback final_output`, 4.4) |
| L1-10 | **N6 MTP-блок** | `tests/test_qwen4exp_mtp_parity.py` (6), тот же шаг | MTP-контекст (`nextn_predict_layers` = 1) как в `common/speculative.cpp` против `mtp_forward_chunk`; цепочка черновика = argmax оракула; TOL 1e-4 | qwen4exp @ec7630a | все три | CI зелёный; локально OK, худшая 1.577e-06 (`hc_inject`, 4.4) |
| L1-11 | **DFlash2**: оракул против исполняемого кода vLLM | `tests/test_dflash2_ref.py` (7: 5 по определению + 2 против vLLM); `ci` → `ref-torch-cpu` → `DFlash2 oracle against the vendored vLLM functions` (`ci.yml:42-58`; шаг падает, если тесты пропущены, `:57`) | `grouped_conv`, `score_edges` против `_grouped_conv`, `_score_edges` из `third_party/vllm-dflash2` (vLLM `bc21cba`), torch на CPU, 1e-12; `selector_walk`, `context_kv` — только против своего определения (PROVISIONAL, `docs/evidence/2026-10-03-dflash2-oracle.md:46-47`) | драфтер DFlash2 для GLM | все три | CI зелёный (аннотация `Ran 7 tests … OK torch 2.14.1+cpu`, 4.2); локально 2 из 7 пропущены — torch в песочнице нет |
| L1-12 | эталон DeepSeek-V4.1 на CPU | `tools/ds41/run_reference.py`, `check_decode_consistency.py`; теста в `tests/` и шага CI **нет** | неизменённый `model.py` (HF `2cba9e4`) на крошечной модели с CPU-оверлеем ядер; numpy-оракул `ref/ds41.py` НЕ НАПИСАН | deepseek-v4.1 | все три (файлы) | разовый ручной прогон 2026-10-02 (`docs/evidence/2026-10-02-deepseek-ref-cpu.md`); сейчас NOT RUN (нет torch) |

### Уровень 2 — CPU-функциональные проверки (настоящий граф llama.cpp или движок на CPU)

Прогоны L1-2, L1-4, L1-5, L1-7, L1-9, L1-10 тоже исполняют настоящий граф llama.cpp на CPU, но их критерий — числа оракула,
поэтому они учтены на уровне 1.

| id | проверка | файл / CI | что проверяется, критерий | архитектура | ветки | статус |
|---|---|---|---|---|---|---|
| L2-1 | трассировщик маршрутизации + симулятор кэша | `tools/g0/run_trace_smoke.sh`; `ci` → `cpu-parity` → `G0 per-token routing tracer + cache simulator on the tiny glm5-next model (CPU smoke)` (`ci.yml:34-35`) | 5 промптов × 300 id + 40 сгенерированных; часть трассы для промпта = трасса без генерации; `cachesim.py` отрабатывает | glm5-next @ec7630a | все три | CI зелёный; локально `trace checks OK: 5 files …` (4.4) |
| L2-2 | трассировка и проба на qwen3moe | `scripts/traces/smoke_tiny.sh`; `ci` → `cpu-parity` → `G0 trace + probe on a tiny qwen3moe …` (`ci.yml:36-39`); `traces-real` → step 7 | промпты длиннее батча; самопроверки `predict_eval`: логиты роутера пересчитаны из GGUF, правило выбора, строки пробы = строки трассы; маршрутизация `-b 64` = `-b 1` на ≥ 99% строк | qwen3moe @ec7630a | только research | CI зелёный (`ci` на research); локально `router_logits_max_rel_err` 2.7e-4, `-b 64 vs -b 1: 2334/2340` (4.4) |
| L2-3 | трассы реальных прокси-моделей на CPU | `.github/workflows/traces-real.yml`, `scripts/traces/run_real_traces.sh` | трассы и пробы для GGUF Qwen3-Coder-30B-A3B и GLM-4.7-Flash (sha256 против LFS-хэша HF); функциональная, не числовая проверка | qwen3moe, deepseek2 | research | run 38076798213 — failure (трасса не записана; причина и исправление — `docs/evidence/2026-10-10-traces-real-fix.md`); run 38079738013 — in_progress на момент проверки (шаг 7 smoke — success, шаг 8 — идёт) |
| L2-4 | **N7: настоящий llama-server пина R1** | `tools/srvcheck/run_n7.sh` → `hybrid_server_checks.py`; `server-cpu` → `n7` → `N7 llama-server checks on tiny qwen4exp and glm5-next (CPU)` (`server-cpu.yml:34-35`@ebecad4) | llama.cpp `79e2e74`, CPU; крошечные qwen4exp и glm5-next (+ MTP-блок): детерминизм, продолжение промпта, расхождение внутри кэша, `--cache-reuse`, сохранение/восстановление слота (+ рестарт), два слота одновременно (сервер сообщает оба занятыми), MTP = обычный жадный декод; PASS = те же токены и \|Δ log-prob\| ≤ 1e-3 по общим top-5 (`--tol`, `hybrid_server_checks.py:386`@ebecad4). Гейт (f32, FA off) обязан пройти; прогон с настройками по умолчанию — характеризация | qwen4exp, glm5-next @79e2e74 | n7, n8 | CI зелёный (n8 `ebecad4`, n7 `986519a`); локально воспроизведено (4.5): гейт `all_pass: true` на обеих архитектурах; с настройками по умолчанию glm5-next — `all_pass: false` (`prefix_extend` 0.0267, `isolation` 1.1e-3/1.9e-3 при тех же токенах), как в evidence N7 находка 7 |
| L2-5 | **N8: повреждённые файлы слотов** | `tools/kv/slot_corruption_probe.py`; `server-cpu` → `n7` → `N8 corrupted slot files - raw llama-server vs tools/kv/session_store.py (CPU)` (`server-cpu.yml:36-42`@ebecad4) | 13 повреждений на модель (флип бита в байте / в экспоненте f32, обрезка): сырой restore против `session_store.py`; ожидание — хранилище отвергает каждый повреждённый файл и после отказа даёт эталонные токены | qwen4exp, glm5-next @79e2e74 | n8 | CI зелёный (`ebecad4`); локально воспроизведено (4.5): сырой сервер принял 26/26, у 5 изменились токены (у одного log-prob = null), хранилище отвергло 26/26 (24 по sha256, 2 по размеру) и после отказа дало эталонные токены |
| L2-6 | дисковый ярус базы A на Linux | `tests/strata_glm/run_diskreader_test.sh`; `ci` → `strata-glm-sm89` → `Linux disk tier (patch 0003) functional test, CPU` (`ci.yml:99-101`) | `DiskReader` (патч 0003, O_DIRECT) на настоящих файлах | база A (strata-glm) | все три | CI зелёный (оба arch); локально NOT RUN (нет исходников strata-glm в песочнице) |
| L2-7 | бинарник базы A (sm_89) стартует без GPU и проверяет флаги | `ci` → `strata-glm-sm89` → `new flags are validated before any CUDA call (no GPU needed)` (`ci.yml:138-152`) | 4 неверных сочетания `--layer-split`/`--gpus`/неизвестный флаг → код 2 и сообщение до первого вызова CUDA | база A | все три | CI зелёный (аннотации `ok: … -> 2`, 4.2) |
| L2-8 | CPU-тесты базы B | `ci` → `lt-strata-sm89` → `CPU-only tests of the fork (GGUF reader, model description)` (`ci.yml:209-223`) | `gguf_reader_test`, `model_test` форка | база B (lt-Strata) | все три | CI зелёный (аннотации `100% tests passed`, 4.2) |
| L2-9 | открывает ли база B нашу крошечную GLM-GGUF | `ci` → `lt-strata-sm89` → `probe - does base B open our tiny GLM GGUF …` (`ci.yml:224-241`, никогда не роняет джоб) | `strata-model-inspect` на `glm5next`/`glm5-next` | glm5next, glm5-next | все три | CI: exit 1 для обоих диалектов (`missing or invalid integer glm5next.expert_group_count`, `unsupported architecture glm5-next`) — несовместимость задокументирована, проверкой корректности не является |

### Уровень 3 — компиляция CUDA

| id | проверка | файл / CI | что проверяется | ветки | статус |
|---|---|---|---|---|---|
| L3-1 | свои ядра под sm_89 | `ci` → `cuda-compile` → `compile kernels for sm_89` (`ci.yml:59-72`) | `nvcc -arch=sm_89 -c kernels/*.cu`; каталога `kernels/` нет → `no kernels yet`, exit 0 (`ci.yml:70-71`) — **зелёный по построению**, своего GPU-кода нет | все три | CI зелёный (пусто) |
| L3-2 | база A: strata-glm `ed37419` + патчи 0001–0006 | `ci` → `strata-glm-sm89` (матрица 89, 86; `ci.yml:73-137`) | конфигурация, сборка, `cuobjdump --list-elf` содержит `sm_89`/`sm_86` | все три | CI зелёный: `strata-glm build OK, 69 CUDA objects`, `cubin archs: 26 sm_89` (4.2) |
| L3-3 | база B: lighttransport/Strata `e486a95` | `ci` → `lt-strata-sm89` (`ci.yml:154-208`) | сборка `strata-glm-decode` и тестов, cubin sm_89 | все три | CI зелёный: `lt-strata build OK, 62 CUDA objects`, `cubin archs: 18 sm_89` (4.2) |
| L3-4 | движки R1 по пинам | `engines-sm89` → `build` (матрица `llama.cpp`, `ik_llama.cpp`, `strata`, `exllamav3`; `engines-sm89.yml:20-64`) | `scripts/engines/build_engine.py --arch 89`, проверка cubin sm_89, `build-info.json` | запускается только на PR/`main` при изменении путей движков (`engines-sm89.yml:8-14`) — на ветках research/n7/n8 не запускался | последний зелёный: run 38080354794 (`claude/relaxed-dijkstra-1d7i5o`, `259081e`): `llama.cpp 79e2e74eb110 sm_89 ok`, `ik_llama.cpp 5194a9e1451b sm_89 ok`, `strata fb58e0dbc839 sm_89 ok`, `exllamav3 151539c77abc sm_89 ok` (4.2) |
| L3-5 | CUDA-программы диагностики R0 | `tests/test_diag_kit.py` `CudaCompile.test_compile_sm89` (`:557`) | `nvcc -arch=sm_89` для `scripts/diag/gpu_p2p_matrix.cu`, `gpu_vram_check.cu` | все три | **пропускается** без nvcc: в джобе `cpu-parity` nvcc нет, в песочнице тоже; последний литеральный `ok` — nvcc 13.4 (`docs/evidence/2026-10-09-tool-mutations.md`, раздел «scripts/diag: сборка CUDA-программ») |
| L3-6 | ручные сборки в песочнице | evidence | llama.cpp `79e2e74`, CUDA 13.4, 144 cubin sm_89 (`docs/evidence/2026-10-10-engines-build.md`); ExLlamaV3 из офлайн-набора, 167 cubin sm_89 (`docs/evidence/2026-10-10-offline-engine-kit.md:120-125`); `src/glm/*.cu` базы A, nvcc 13.4 (`docs/evidence/2026-10-02-strata-glm-sm89.md:5-16`) | — | CONFIRMED по evidence; сейчас не повторялось (nvcc в песочнице нет) |

### Уровень 4 — исполнение CUDA: NOT RUN

Ни одна проверка не исполнялась на sm_89: GPU нет ни в песочнице (`/dev/nvidia*` отсутствует, `nvidia-smi`/`nvcc` не найдены —
раздел 4.6), ни на раннерах GitHub (во всех workflow — «no GPU on the runner», например `ci.yml:63`, `engines-sm89.yml:2-3`), а
доступа к серверу из облака нет (`PROJECT_STATE.md:27`@a2c1815). Единственные исполнения на GPU — у оператора стенда
3×RTX 3090 (sm_86): strata-glm на синтетическом паке, `--layer-split 14,25,35 --gpus 0,0,0,0` побайтно равен единому движку,
ярус RAM побайтно равен дисковому, ik_llama.cpp на реальном UD-Q4_K_XL — **PROVISIONAL** (замеры чужого хоста, не повторены;
`docs/evidence/2026-10-03-3090-review.md:23-31`); к целевым картам и к нашим проверкам отношения не имеют. Что готово для уровня 4 —
раздел 2.2.

### Уровень 5 — производительность GPU: NOT RUN

Своих замеров скорости на GPU нет. Числа 3090-стенда (декод 19.7 tok/s на синтетическом паке, ik_llama.cpp tg 15.0 / pp 166.8,
база A 13.89 tok/s) — PROVISIONAL/REPORTED (`docs/evidence/2026-10-03-3090-review.md:30-31`, `EVIDENCE_MATRIX.md` E29, E32);
синтетический пак как ориентир снят (E54). Инструменты (`tools/bench/`, `tools/roofline.py`, `vs_roofline.py`) проверены только на
mock — ниже, «Опорные тесты».

### Уровень 6 — производительность на целевой машине: NOT RUN

R0–R6 на сервере не начинались: ни одна строка `ROADMAP.md:15-70`@a2c1815 не выполнена на сервере (статусы — NOT RUN, «сделано на mock / в песочнице», «ожидает»),
`HARDWARE_FACTS.md` F1–F20 — UNKNOWN, verification pack не собран (`VERIFICATION_PACK.md:3`). Что нужно запустить — раздел 2.1 и 2.3.

### Опорные тесты инструментов (не проверяют модель; mock, заглушки, синтетика)

Эти тесты гарантируют, что измерительные и эксплуатационные инструменты уровней 5–6 сами не врут (в том числе мутационными
проверками в `docs/evidence/2026-10-09-tool-mutations.md` и последующих), но ни один из них не исполнял настоящий движок на GPU.
Все входят в шаг `unit tests` (`ci.yml:17-18`); тесты N7/N8 — ещё и в `server-cpu` → `harness tests against a stand-in server`
(`server-cpu.yml:26-27`@ebecad4).

| область | файлы (число `def test_`) | против чего |
|---|---|---|
| стенд бенчмарков, пробы 262K, замер vs граница | `test_bench` (38), `test_vs_roofline` (3) | `tools/bench/mock_server.py` |
| сравнение распределений токена, слоты | `test_logprob_compare` (6), `test_slot_persist` (5) | заглушки OpenAI / llama-server |
| стенд N7 и хранилище сессий N8 | `test_srvcheck` (18 на n7, 19 на n8), `test_n7_report` (3, n7/n8), `test_session_store` (17 на `ebecad4`, n8) | `tests/fixtures/srvcheck/stub_llama_server.py` с внесёнными неисправностями |
| шлюз, оценочный стенд | `test_gateway` (29), `test_eval` (34), `test_eval_via_gateway` (3) | mock-сервер, скриптованный агент |
| оркестрация R1, verification pack | `test_r1_orchestration` (23), `test_verify_pack` (19) | mock-движок, синтетические артефакты (вердикт mock = NOT RUN) |
| модели скорости и ёмкости | `test_roofline` (26), `test_fitplan` (15), `test_tier_cost` (26, только research), `test_arch_ceilings` (30, только research) | опубликованные размеры, свойства формул |
| маршрутизация и кэш (инструменты) | `test_cachesim` (14), `test_g0_analyze` (4), `test_predict_eval` (12, только research) | синтетические трассы |
| веса, движки, диагностика | `test_gguf_quant_audit` (7), `test_offline_weights` (5), `test_offline_engines` (9 на research, 10 на n7/n8), `test_engines_build` (13), `test_diag_kit` (33) | заголовки GGUF, локальный Range-сервер, фейковые репозитории, заглушки системных утилит |
| процесс | `test_hooks` (21), `test_team_config` (5), `test_merge_matrix` (19, только research), `test_repo_backup` (4, n7/n8 после слияния PR #1) | временные репозитории, файлы `.claude/` |

## 2. Проверки, которые ждут GPU или сервер

Ниже — всё, что нельзя выполнить без целевых карт или самого сервера (диск, RAM, драйвер, веса на NVMe). Колонка «инструмент»
называет только то, что лежит в репозитории; где его нет — «tool missing». Все строки — **NOT RUN** (на сервере не запускалось
ничего: `ROADMAP.md:16-18`, `PROJECT_STATE.md:27`@a2c1815 — CONFIRMED чтением). Пороги взяты из документов-источников; их метки
(CONFIRMED / PROVISIONAL) — как в источнике.

### 2.1 Факты о железе (R0) — уровень 6

| # | что проверит | готовый инструмент / команда | входы | критерий PASS | гейт / строка |
|---|---|---|---|---|---|
| B1 | F1–F20: CPU/AVX, NUMA, DIMM, PCIe, GPU, NVMe, ПО, сеть | `scripts/diag/collect.sh --sudo` → `scripts/diag/summarize.py --json` | доступ к серверу, `sudo -n` | каждая строка `HARDWARE_FACTS.md` CONFIRMED или UNKNOWN с причиной; в pack: `synthetic=false`, обязательные факты `tools/verify/criteria.default.json` `hw_required_facts` (V8) | R0.2, R0.3 (`ROADMAP.md:16-17`); H1, H8 (`EXPERIMENTS.md:30,37`) |
| B2 | каналы DDR4 и полоса RAM | `scripts/diag/bench_mem.sh` (`membw.c`) через `bench_all.sh` | — | факт; ожидание 8 каналов ≈ 140–170 ГБ/с triad, < 90 ГБ/с → 4 канала или NPS (PROVISIONAL, `EXPERIMENTS.md:31`) | H2; E3, E43 |
| B3 | реальные 32 GB VRAM на каждой карте | `scripts/diag/bench_gpu.sh --run --vram-check` (`gpu_vram_check.cu`) | 4 GPU без нагрузки | 0 ошибок шаблонов на 4 картах (`EXPERIMENTS.md:32`) | H3; F10 |
| B4 | полоса памяти карты (D2D) | `bench_gpu.sh --run` (`gpu_p2p_matrix.cu`) | — | ожидание ≥ 600 ГБ/с (PROVISIONAL, `EXPERIMENTS.md:33`) | H4; F11 |
| B5 | PCIe Gen/ширина, H2D/D2H по одной и всем картам сразу | `bench_gpu.sh --run [--numa-node N]` + опрос `nvidia-smi` | — | x16 Gen4 ≈ 24–26 ГБ/с; одновременная H2D 4 карт падает ≤ 20% (`EXPERIMENTS.md:34`) | H5; F12, F13 |
| B6 | P2P: разрешён ли, полоса и латентность D2D | `gpu_p2p_matrix` (`cudaDeviceCanAccessPeer`, `enable_peer`, `d2d_*`, `lat_d2d_peer`) | — | факт (есть/нет); без P2P — цена host-staged обмена (`EXPERIMENTS.md:35`) | H6; F14; условие R4.1 |
| B7 | NVMe: модель, SMART, линк, p99 4K QD1, IOPS QD32, ГБ/с, масштаб 1→4 | `scripts/diag/bench_nvme.sh` (fio, O_DIRECT, только чтение) | тестовый каталог на каждом NVMe | факт (`EXPERIMENTS.md:36`) | H7; F15, F16 |
| B8 | настройки BIOS по шагам с повтором F4–F6, F9, F12–F14 | tool missing (только `BIOS_PLAN.md`, ручной журнал) | физический доступ | журнал «до/после» | R0.4 (`ROADMAP.md:18`) |
| B9 | all-reduce 8–64 КБ на 4 картах, с P2P и без | tool missing (в репозитории нет NCCL-теста; колесо `nvidia-nccl-cu13` есть только в офлайн-наборе ExLlamaV3, `scripts/offline/engines.py:38`) | — | латентность p50/p99; гипотеза «≤ 30 мкс на слой» (`EXPERIMENTS.md:90`) | R4.1 (`ROADMAP.md:52`); P2 |

### 2.2 Движок на GPU: запуск и корректность (уровень 4)

| # | что проверит | готовый инструмент / команда | входы | критерий PASS | гейт / строка |
|---|---|---|---|---|---|
| B10 | движки собираются **на сервере** его драйвером и стартуют | `python3 scripts/offline/engines.py verify <kit>`; `python3 scripts/engines/build_engine.py --engine <e> --arch 89 --offline-kit <kit>`; `<bin> --version` | офлайн-набор (`docs/evidence/2026-10-10-offline-engine-kit.md`) | `build-info.json` sha = `tools/engines/pins.json`, cubin sm_89, бинарник запускается с драйвером сервера (совместимость CUDA 13 с драйвером — UNKNOWN, `HW_DIAG_RUNBOOK.md:224-226`) | R1.4 (`ROADMAP.md:27`); DEPLOY P2 |
| B11 | функциональные проверки N7 на CUDA-сборке llama-server: детерминизм, префикс, чекпойнты, `--cache-reuse`, слоты с рестартом, два слота, MTP = обычный декод; падение первого MTP-графа (N6 находка 12) на GPU | `python3 tools/srvcheck/hybrid_server_checks.py --server <CUDA llama-server> --model … --mtp-model … --extra "<флаги> -ngl 999"` (`hybrid_server_checks.py:379-389`@ebecad4 принимает любой бинарник и флаги); драйвер `run_n7.sh` собирает только CPU-версию — GPU-вариант драйвера missing | крошечные GGUF из `tools/tinygen` | гейт N7: те же токены и \|Δ log-prob\| ≤ 1e-3 по общим top-5 при f32-кэшах и FA off; MTP без падения | R1.4/R1.5 (предусловие); S1; `docs/evidence/2026-10-10-n7-server-hybrid-cpu.md` «Not covered» |
| B12 | оракулы `ref/` против дампов **CUDA-бэкенда** (тензоры `cb(…)` как в G1/N2/N6) | `tools/dump/dump_tensors.cpp` (параметры через `common_params_parse`, `-ngl` работает: `dump_tensors.cpp:219`) + сборка `tools/g0` с `-DGGML_CUDA=ON` (`tools/g0/CMakeLists.txt:2-3`); скрипты `run_parity*.sh` конфигурируют только CPU, допуски для CUDA (f16/bf16-аккумуляция, ничьи top-k: G1 находка 5) не заданы — tool partially missing | крошечные GGUF | допуск на GPU — UNKNOWN, задать до прогона; выбор DSA/QSA — «любой допустимый top-k» | нет строки в ROADMAP (старые G2/G3, `docs/PLAN.md:13-14`) — пробел, см. разд. 3 |
| B13 | шум «тот же GGUF: CPU против GPU» в llama.cpp — калибровка порога точной маршрутизации | `python3 tools/quality/logprob_compare.py --ref <CPU llama-server> --cand <GPU llama-server> --corpus … --prefixes 200 --top-k 20` | реальный GGUF, корпус | заменяет PROVISIONAL порог top-1 ≥ 0.97 измеренным (`criteria.default.json` `_notes.top1_agreement_min`) | V6; R2.1 |
| B14 | ядра базы A (strata-glm `src/glm/*.cu`) против эталонов и FP32-эталона автора `tools/glm_ref.py` | tool missing у нас (`scripts/gpu_day.sh` check1–check3 сравнивает только токены 1 GPU / 4 стадии / 4 GPU; помечен SUPERSEDE: `LEGACY_AUDIT.md:35`, NVFP4-чекпойнт отменён ADR-010, шаг build требует интернет) | NVFP4-пак (конвертера GGUF→pack нет) | токены `--layer-split 14,25,35 --gpus 0,1,2,3` = единый движок; логиты — UNKNOWN | B1a (`EXPERIMENTS.md:45`); база A заморожена с условием разморозки (ADR-012 D4) |
| B15 | GPU-тесты базы B (`glm_parity`, `glm_prefill_parity` форка) | ctest форка; в CI собираются только CPU-тесты (`ci.yml:193`) | GPU + модель | по тестам форка | база B заморожена (E30) — справочно |
| B16 | ExLlamaV3 `151539c` грузится и считает на sm_89 | колесо из `engines-sm89` / офлайн-набора; шаблона R1 для ExLlamaV3/TabbyAPI нет — tool missing | EXL3 ≥ 4.0 bpw (квантование на сервере) | см. B5 эксперимента | B5 (`EXPERIMENTS.md:49`) |

### 2.3 Измерения R1–R5 на сервере (уровни 5–6) и сдача (R6)

| # | что проверит | готовый инструмент / команда | входы | критерий PASS | гейт / строка |
|---|---|---|---|---|---|
| B17 | целостность весов | `scripts/offline/weights.py verify <manifest> --dest <nvme>` | веса шорт-листа на NVMe | sha256 и размер каждого файла = манифест ревизии (V2) | R1.3 (`ROADMAP.md:26`) |
| B18 | ≥ 4 бит у экспертов и прочих матриц | `python3 tools/gguf_quant_audit.py --json <shard-1>` | реальные GGUF | ни одного тензора < 4.0 bpw, нет неизвестных типов (V1) | R1.2 (`ROADMAP.md:25`); Q1 |
| B19 | ячейка R1 одной командой (старт движка, матрица, пробы, аудит, pack, стоп) | `python3 scripts/r1/run_r1.py --config scripts/r1/configs/{llamacpp-server,llamacpp-server-moecache,ik-llamacpp-server,strata-serve-qwen38fn}.json --out …` (все шаблоны PROVISIONAL, NOT RUN); шаблонов для `-sm tensor`/ik `-sm graph`/ExLlamaV3/базы A нет — tool missing | движок по пину, веса | вердикт pack по V1–V9 | R1.5 (`ROADMAP.md:28`); B0, B1, B1a, B5 |
| B20 | декод single-stream при заполненном 262K ≥ 50 tok/s (цель 80–100+) | `tools/bench/run_matrix.py --inputs 262144 --length-mode total --outputs 512,2048 --states cold,warm --concurrency 1 --repeats 3` | сервер движка | p50 ≥ 50, 0 отказов, ≥ 3 повтора (V4, `BENCH_PROTOCOL.md:126-128`) | R1.5; критерий заказчика (ADR-012 D1) |
| B21 | 262K реально используемых токенов | `tools/bench/probes.py --contexts 262144 --length-mode total` | сервер | needle ≥ 0.8; multihop, codeqa ≥ 0.75 (PROVISIONAL); tools/json = 1.0; промпт не обрезан (V3) | R1.6; K1 |
| B22 | кэш префиксов и agent loop | `run_matrix.py --states cold,warm` + `--scenarios agent` | сервер | warm TTFT p50 ≤ 10% cold; agent loop до ≥ 100K без отказов (V5) | B3; V5 |
| B23 | замер против границы roofline | `tools/roofline.py` (предсказание до замера) + `tools/bench/vs_roofline.py --json` | каталоги прогона | файл обязателен; отношение > 1 → NOT RUN (входы границы неверны) (V9) | R1.5 |
| B24 | качество кванта: KLD/top-1 к Q8_0 | `tools/quality/logprob_compare.py --ref <Q8_0> --cand <4-бит>` | оба GGUF, корпус 50–100K токенов | формат с KLD в пределах 1.5× лучшего (`EXPERIMENTS.md:82`) | R2.1; Q1 |
| B25 | KV f16 против q8_0/fp8 на 128K/262K | `probes.py` при разных `-ctk/-ctv`; обёртки для PPL нет — tool missing | сервер | Δ needle ≤ 2 п.п., ΔPPL ≤ 1% (`EXPERIMENTS.md:66`) | R2.3; K2 |
| B26 | агентный набор заказчика | `tools/eval/` (+ через `tools/gateway/`) | 30–100 задач заказчика (ждут материалов) | доля решённых | R2.2 |
| B27 | сохранение/восстановление 262K-состояния с NVMe | `tools/kv/slot_persist.py`; восстановление — только через `tools/kv/session_store.py` (N8) | сервер с `--slot-save-path` на NVMe | ≥ 3 повтора, продолжение совпадает, restore + тёплый префилл ≤ 25% холодного (V7) | R5.2; K3 |
| B28 | спекуляция MTP / n-gram | `run_matrix.py` с/без; `roofline.py --spec-k` | движок с MTP | выигрыш ≥ 1.1× или выключено | R4.4; S1, S2 |
| B29 | GPU-кэш экспертов `--moe-cache-mib` и политики D1 | шаблон `llamacpp-server-moecache.json`; патча D1 нет (дизайн, NOT RUN) | GLM UD-Q4_K_XL | для политик «только место чтения» — бит-экзактность логитов; для счёта промахов на CPU — top-1 ≥ 0.99 (`ROADMAP.md:46`) | R3.3 |
| B30 | промах на CPU против копии по PCIe | tool missing (микробенчмарка нет; только режимы движков) | — | мс на промах | R3.4; M4 |
| B31 | трассы маршрутизации реальной модели и политики кэша | `tools/g0/expert_trace.cpp` (`g0-expert-trace`), `tools/g0/cachesim.py` | реальный GGUF, корпус заказчика | кривая попаданий held-out | R3.1, R3.2; M1, M2 |
| B32 | TP/EP в готовых движках | шаблонов нет — tool missing | R4.1 | выигрыш к PP при batch=1 на 262K | R4.2 |
| B33 | P2P-патч драйвера | `bench_gpu.sh` после патча | согласие заказчика | матрица P2P + R4.1 | R4.3 (ОЖИДАЕТ СОГЛАСИЯ) |
| B34 | эксплуатационные отказы F1–F8 | ручной сценарий `DEPLOY.md` §6 (`DEPLOY.md:144-151`); на mock проверены только F1/F2 шлюза | сервер | матрица pass/fail с логами (V10, ручной) | R5.3 |
| B35 | шаги runbook | `DEPLOY.md` (черновик) | сервер | каждый шаг с литеральным выводом | R5.4 |
| B36 | verification pack | `scripts/verify_pack.sh --runs <dir> --out docs/evidence/verification-pack-<date>`; `--check` | артефакты B1–B35 | общий PASS только если все обязательные критерии PASS | R6.2 |
| B37 | DFlash2 на цели: принятие | движка нет — tool missing (оракул `ref/dflash2.py` есть) | драфтер canada-quant | принятие ≥ 50%, < 40% → отказ | S3 (заморожено) |

## 3. Пробелы: что можно проверять на CPU уже сейчас, но не проверяется

Порядок — по риску для развёртывания на 4×4080 (первый кандидат — Qwen3.8-Flash-Next, граф `qwen4exp`; GLM-5.3-Flash —
`glm5-next`/`glm5next` через llama.cpp/ik_llama.cpp; веса — 4-битные k-quant GGUF; KV f16 и flash attention по умолчанию в
шаблонах R1). Оценка риска — PROVISIONAL (суждение parity-qa по составу проверок выше); факты внутри пунктов помечены отдельно.

| # | пробел (операция / архитектура / путь) | почему это риск | что сделать на CPU (без GPU) |
|---|---|---|---|
| 1 | **Паритет оракулов на пине движка R1.** Все сверки с дампами (G1 GLM, MiMo, N2, N6) в CI идут на llama.cpp `ec7630a` (`ci.yml:13`), а R1 и сервер N7 — на `79e2e74` (`tools/engines/pins.json`). Между пинами изменились `src/models/glm5-next.cpp` (312 строк diff), `src/models/qwen4exp.cpp` (140), `src/llama-graph.cpp` (321), `src/llama-memory-hybrid-idx.cpp` (184), `ggml/src/ggml-cpu/ops.cpp` (37), `src/models/mimo2.cpp` (29); `delta-net-base.cpp` — 0 (CONFIRMED: `diff` двух деревьев, 4.6). **Проба этой инвентаризации (4.7):** те же скрипты на `79e2e74` — G1 GLM 8/8, MiMo 5/5, N2 17/17 OK; N6 — 52/54: оба падения только в сравнении строк-заполнителей `indexer_pool_k_new` (на `79e2e74` в них другое содержимое); если это одно сравнение не утверждать, k-pool кэш, оценки, выбор QSA (0 повторов ничьих) и логиты совпадают с оракулом (2/2 OK) (CONFIRMED, 4.7) | доказательства корректности относятся не к тому бинарнику, который будет обслуживать заказчика; тест N6 привязан к детали реализации пина `ec7630a` и при обновлении пина падает не по существу, а существенное расхождение на другом пине никто бы не увидел | шаг CI с теми же `run_parity*.sh` на коммите из `tools/engines/pins.json` (сборка `dump-tensors` ≈ 2.5 мин на 4 ядрах, 4.7); сравнение заполнителей сделать зависящим от пина или убрать — по комментарию теста их слоты не читаются (`tests/test_qwen4exp_decode_parity.py:365-367`@a2c1815); безвредность на `79e2e74` — PROVISIONAL: всё, что могло бы их читать, совпало |
| 2 | **Квантованные веса.** Все крошечные модели — f32 (`docs/evidence/2026-10-10-qwen4exp-tiny-parity.md:165`); ни одна проверка не сравнивает матричное умножение llama.cpp с Q4_K/Q5_K/Q6_K/Q8_0 (состав UD-Q4_K_XL), IQ4_XS или MXFP4 (MiMo, DeepSeek) с эталоном на **деквантованных** весах; CPU-ядра k-quant на AVX2 (путь экспертов в RAM для GLM: `-ot`/`--n-cpu-moe`) не проверены вовсе | реальные файлы — именно эти форматы; ошибка раскладки блока или масштаба не видна на f32 | `llama-quantize` крошечных `glm5-next`/`qwen4exp` в Q4_K_M и Q8_0 (+ MXFP4 для `mimo2`), дамп `ffn_moe_*`/`attn_*`, эталон — `gguf.quants.dequantize` из gguf-py пина; допуск с учётом квантования активаций (q8_K на CPU) — задать и обосновать; сборка с `GGML_NATIVE=OFF` уже даёт AVX2-путь без AVX-512 (CONFIRMED: `GGML_AVX2=ON`, `GGML_AVX512=OFF` в CMakeCache сборки, 4.6); деквантование всех этих форматов в gguf-py пина есть (CONFIRMED: `gguf-py/gguf/quants.py:68` `dequantize`, классы `Q8_0` :379, `Q4_K` :476, `Q5_K` :526, `Q6_K` :553, `MXFP4` :657, `IQ4_XS` :1352 @ec7630a, 4.6) |
| 3 | **Точность по умолчанию: KV f16 + flash attention.** Паритет оракулов — только при `-ctk f32 -ctv f32 -fa off` (`run_parity.sh:16`, `run_parity_qwen4exp.sh:17`); шаблоны R1 запускают `-fa on` и `-ctk/-ctv ${KV_TYPE}` = `f16` (`scripts/r1/configs/llamacpp-server.json:21,36-37`@a2c1815; на голове PR #1 `259081e` `-fa on` — строка 37). N7: у крошечной glm5-next при настройках по умолчанию тёплый и холодный ответы расходятся до 0.027 в log-prob, а отклонение от f32-ответа — 0.416 уже на первом шаге и другой токен к шагу 3 (CONFIRMED, `docs/evidence/2026-10-10-n7-server-hybrid-cpu.md`, находка 7) | у GLM вероятная причина — переключение выбора DSA k-pool на почти-ничьих (PROVISIONAL, не трассировано); на реальных весах — UNKNOWN | вариант `run_parity.sh`/`run_parity_qwen4exp.sh` с `-ctk f16 -ctv f16 -fa on`; оракул с округлением q/latent/вероятностей до f16 (G1 находка 4 уже воспроизвела это до 6.4e-8 при FA off, `docs/evidence/2026-10-02-g1-parity.md:38-40`, но как тест не закреплено); счёт «сколько токенов меняют выбор DSA/QSA» |
| 4 | **GLM (`glm5-next`) кроме одного префилла.** Для GLM нет аналога N6: нет сверки декода по одному токену, нескольких ubatch, отката (`n_rs_seq`), переноса состояния KDA и истории свёрток, k-pool кэша индексера DSA между ubatch, латентного кэша MLA при декоде и MTP-блока `nextn` (генератор `glm5next_tiny.py --mtp` уже есть с N7). Нет и сквозной сверки логитов GLM: общий эксперт, `ffn_out`, `hc_head`, финальная норма и выход не проверяются (тесты `tests/test_glm_tiny_parity.py` — только по операциям) | GLM — второй кандидат и единственный с DSA; ошибка в переносе состояния видна только при декоде | перенести метод N6 (`DUMP_PER_GRAPH`, `DUMP_CALLS`, `DUMP_N_RS_SEQ`, `DUMP_MTP` в `tools/dump/dump_tensors.cpp` уже есть) на `glm5-next`: `ref/` для инкрементального пути KDA/DSA/MLA + `mtp_forward_chunk` GLM; сквозной прогон оракула от id токенов |
| 5 | **Диалект `glm5next` и ik_llama.cpp.** Реальные файлы Unsloth — архитектура `glm5next` (`AGENTS.md` «Weights on host epyc-ai»), шаблон R1 ik_llama.cpp рассчитан на них (`scripts/r1/configs/ik-llamacpp-server.json:2`); наш генератор пишет `glm5-next`, схемы метаданных различаются не только именем (CONFIRMED: `glm5next.expert_group_count`, `docs/evidence/2026-10-03-lighttransport-fork.md:92-105`). ik_llama.cpp `5194a9e` ни разу не запускался ни на одной модели — только сборка | вторая строка матрицы R1 для GLM не проверена даже на CPU | CPU-сборка ik_llama.cpp; писатель крошечного `glm5next`-GGUF (ключи — из заголовка реального файла или `include/strata/core/model.hpp` форка); те же дампы/оракулы и проверки N7 через его сервер |
| 6 | **Шаблоны чата, токенизатор, вызовы инструментов.** Крошечные модели — минимальный словарь, запросы — списки id (N7, «Method»). Как llama-server расставляет чекпойнты при перерисовке истории настоящим шаблоном GLM-5.3/Qwen3.8 — UNKNOWN (N7, находка 4); разбор tool calls (`--jinja`) на этих шаблонах не проверялся | продукт — агентные сессии с tool calls; B3 зависит от того, что шаблон делает со старыми ходами | забрать `tokenizer.json`/`chat_template` через `vendor-hf` (`third_party/hf/*/REQUEST.json`); GGUF только со словарём; прогон `/v1/chat/completions` и пробы B3 на CPU-сервере N7 |
| 7 | **Измерительные инструменты против настоящего сервера.** `tools/bench/` (`run_matrix.py`, `probes.py`, диалекты `/tokenize`), `tools/quality/logprob_compare.py`, `tools/gateway/`, `scripts/r1/run_r1.py` проверены только на mock/заглушках (`BENCH_PROTOCOL.md` §8: «Против реальных llama.cpp … NOT RUN»); на настоящем llama-server работали только `slot_persist.py` и `session_store.py` (N7/N8) | ошибка формата всплывёт в первый день на сервере и съест время на GPU; вердикты pack при этом безопасны (ошибки → NOT RUN) | CPU-сервер N7 + крошечная модель: `run_r1.py` с `llamacpp-server.json` (пути переопределить `--set`), `probes.py` на коротких длинах, `logprob_compare.py` между `-fa off` и `-fa on`, шлюз перед сервером; ожидаемый вердикт pack — NOT RUN |
| 8 | **Нефьюзные ветви графа.** CPU всегда берёт фьюзные операции (чанковая/авторегрессионная delta net, нефьюзный HC); эти ветви не исполнялись (`docs/evidence/2026-10-10-qwen4exp-tiny-parity.md:164`, «NOT EXERCISED») | бэкенд без фьюзной операции (возможно, часть CUDA-путей — UNKNOWN) пойдёт по непроверенной ветви | выяснить, можно ли выключить фьюзные операции на CPU (UNKNOWN); если да — повторить N2/N6 на нефьюзном пути |
| 9 | **Гиперпараметры реальных моделей.** Крошечные конфиги соблюдают только ограничения загрузчика; реальные (число hc, головы, секции IMRoPE, PLE, шаблон слоёв) — UNKNOWN (`docs/evidence/2026-10-10-qwen4exp-tiny-parity.md:167`). Аудит битности V1 на настоящих заголовках не запускался | ветви графа, включаемые только реальной геометрией, не проверены; «4-битный» файл с 3-битными тензорами обнаружится только после скачивания 100–200 GB | на раннере с доступом к HF (как `traces-real`) читать только заголовки GGUF (HTTP Range), гонять `tools/gguf_quant_audit.py` и строить крошечные модели с реальными соотношениями |
| 10 | **Несколько последовательностей и края отката (qwen4exp).** Не покрыты: общие ячейки (`cache_safe` = false), `seq_rm`, оставляющий «устаревший» полный пул, откат длиннее `n_rs_seq`, откат внутри многобатчевого вызова (`docs/evidence/2026-10-10-qwen4exp-decode-parity.md:436-442`); MTP с откатом в MTP-контексте — тоже | R1 с `-np 1` риск снижает; два слота проверены только на уровне сервера (N7) | расширить `DUMP_CALLS` на `seq_cp`/`seq_add` и длинный откат |
| 11 | **Ложнозелёные сигналы CI.** (а) шаги паритета проходят при пропуске тестов: только `ref-torch-cpu` падает на `skipped` (`ci.yml:57`); (б) `cuda-compile` зелёный по построению — каталога `kernels/` нет (`ci.yml:70-71`); (в) сборка `scripts/diag/*.cu` под sm_89 в CI пропускается: в `cpu-parity` нет nvcc (`tests/test_diag_kit.py:557`); (г) докстринг `tests/test_srvcheck.py:5`@ebecad4 ссылается на несуществующий `tests/test_srvcheck_llama.py` (CONFIRMED: файла нет ни в одной ветке, раздел 4.6) | зелёный CI может скрыть невыполненную проверку; программы R0 впервые соберутся на сервере | в шагах паритета — `grep skipped` как в `ref-torch-cpu`; убрать или наполнить `cuda-compile`; добавить `cuda-nvcc-13-0` в джоб с `test_diag_kit`; поправить докстринг |
| 12 | **MiMo (`mimo2`).** Проверены только внимание (SWA, sinks, GQA, частичный RoPE) и роутер на одном префилле; нет декода, сквозных логитов, MTP и MXFP4-экспертов | вторая модель шорт-листа, MXFP4 — отдельный формат | метод N2/N6 на `mimo2`; MXFP4 — вместе с п. 2 |
| 13 | **DeepSeek-V4.1-Flash.** Эталонная модель исполнялась на CPU вручную (2026-10-02), numpy-оракул `ref/ds41.py` не написан, теста и шага CI нет | третья цель; в 4 битах не помещается (E17) — риск для текущего развёртывания низкий | `ref/ds41.py` по префилл-семантике (находка A) и тест против `tools/ds41/run_reference.py` в джобе с torch |
| 14 | **Отчёт об ошибке по слоям.** Тесты GLM и MiMo только утверждают допуск; максимум abs/rel по слоям не печатается (контракт parity-oracle) — в отличие от N2/N6 | при регрессии не видно, насколько близко к порогу | печать таблицы как в `tests/test_qwen4exp_tiny_parity.py` |

**Статус пробела 1 (после инвентаризации, тот же день):** закрыт на CPU. Проверка заполнителей `indexer_pool_k_new` в N6 теперь учитывает пин: на `79e2e74` каждая заполняющая запись берёт свою ячейку, не являющуюся представителем пула. Workflow `parity-r1-pin.yml` гоняет G1/MiMo/N2/N6 на пине `pins.json`; локально на `79e2e74` N6 — 54/54. Подробности и литеральный вывод — `docs/evidence/2026-10-10-parity-r1-pin.md` (CONFIRMED).

**Статус пробела 4 (N9, 2026-10-10):** закрыт на CPU, ветка `claude/n9-glm-decode-parity`. Оракул всей модели `ref/glm5next.py` (из `ref/mhc.py`, `ref/kda.py`, `ref/dsa.py`, `ref/router.py`) с инкрементальным путём (`forward_chunk`/`cache_truncate`: история свёрток и состояние KDA, латентный кэш MLA, строки и пулы индексера) и MTP-блоком; сквозная сверка префилла G1 (`GlmTinyWholeModel`); 5 прогонов на настоящем графе `glm5-next` — `ub16`, `ub10`, `ub18`, декод 48 + 16×1, откат (`n_rs_seq` 3) — сверяются граф за графом с полным и инкрементальным оракулом: на `ec7630a` 45/45, худшая 4.431e-05; на `79e2e74` 45/45 + MTP 6/6, худшая 3.605e-05 (CONFIRMED, `docs/evidence/2026-10-10-glm5next-decode-parity.md`). Расхождений llama.cpp с оракулом нет. На `ec7630a` у DSA есть второй путь для ubatch ≤ 16 токенов (сбор выбранных латентов), на `79e2e74` его нет; MTP-граф GLM есть только на `79e2e74` (на `ec7630a` — исключение) (CONFIRMED, там же, находки 2 и 5). Шаги CI добавлены в `ci.yml` (`cpu-parity`) и `parity-r1-pin.yml`; в GitHub Actions — NOT RUN (ветка не публиковалась этой ролью).


## 4. Литеральный вывод команд этой инвентаризации

Пути песочницы сокращены: `$S` = `/tmp/claude-0/-home-user-408032gb/83634621-ad4f-5747-933c-0d86b3c4b300/scratchpad`.
Сборки — только в `$S/corr/` (CPU, `GGML_NATIVE=OFF`); llama.cpp `ec7630a` — `$S/llama-ec7630a`, `79e2e74` — `$S/llama.cpp`
(неглубокие клоны, только чтение). Скрипты-обёртки этой инвентаризации — `$S/corr/run_*.sh` (в репозиторий не входят).

### 4.1 Ветки
```
$ git -C /home/user/wt-research rev-parse --abbrev-ref HEAD; git -C /home/user/wt-research log --oneline -n 5; git -C /home/user/wt-research status --short
claude/research-landscape
a2c1815 [perf-bench] evidence tier cost: PCIe prefetch pays only when the CPU expert path is compute-bound (follow-up to R11 C24)
6c20aa1 [lead-architect] research README: index of R5-R12, COVERAGE and MATRIX; corrections to R5-R10 from R11 and the tier cost model
16bf0a5 [lead-architect] research: coverage audit of the 45 items of the customer's list (deliverable A) and R12 gap-fill
c95c670 [model-analyst] research: merged engine technology matrix (deliverable B) and tools/research/merge_matrix.py
7e534c9 [lead-architect] research R11: independent verification of 33 decision-relevant claims of R5-R10 (20 confirmed, 13 corrected)
(пусто)
$ git -C /home/user/408032gb rev-parse --abbrev-ref HEAD; git -C /home/user/408032gb log --oneline -n 5; git -C /home/user/408032gb status --short
claude/n7-server-hybrid-cpu
090ca71 [parity-qa] N7 after sync with PR #1: --cache-reuse dropped in bf1a72b; N6 finding-12 risk resolved for the server path on CPU
9a48ed5 [parity-qa] merge claude/relaxed-dijkstra-1d7i5o (PR #1 head 259081e) into N7: ROADMAP conflict resolved
259081e [infra-ops] evidence: backup bundles were made in a shallow clone; fixed, new bundle restores in an empty directory
125c9c3 [infra-ops] backup bundles: refuse shallow clones, verify by restoring into an empty directory
ca1d2a7 [infra-ops] offline kit audit closed: literal outputs, exact sizes, cp312 rationale; verify names left-out engines
(пусто)
$ git -C /home/user/wt-n8 rev-parse --abbrev-ref HEAD; git -C /home/user/wt-n8 log --oneline -n 5; git -C /home/user/wt-n8 status --short
claude/n8-session-store
43c2171 [parity-qa] evidence N8: server-cpu green on ebecad4 (runs 38081791575, 38081793624); round 4 pinned to 25c61f3
750fe3d [parity-qa] merge claude/n7-server-hybrid-cpu (090ca71, synced with PR #1) into N8: PROJECT_STATE conflict resolved
090ca71 [parity-qa] N7 after sync with PR #1: --cache-reuse dropped in bf1a72b; N6 finding-12 risk resolved for the server path on CPU
9a48ed5 [parity-qa] merge claude/relaxed-dijkstra-1d7i5o (PR #1 head 259081e) into N7: ROADMAP conflict resolved
ebecad4 [parity-qa] evidence N8 after audit: 24 by sha256 + 2 by size; no checksum but structural checks exist; unsaved timings relabelled
(пусто)
$ git -C /home/user/wt-research merge-base HEAD origin/claude/n8-session-store
b6f4fe3fae4bd83479862a4be26e49386118c88e
$ git -C /home/user/wt-research merge-base HEAD origin/claude/n7-server-hybrid-cpu
b6f4fe3fae4bd83479862a4be26e49386118c88e
$ git -C /home/user/wt-research merge-base --is-ancestor 259081e HEAD && echo yes || echo no
no
$ git -C /home/user/wt-n8 merge-base --is-ancestor 259081e HEAD && echo yes || echo no
yes
```

### 4.2 CI (GitHub API, только чтение; список прогонов снят 2026-10-10T20:16Z)
```
$ gh run list -R samnolak/408032gb --limit 40 --json databaseId,workflowName,headBranch,headSha,status,conclusion,event
38082629418 server-cpu claude/n8-session-store 43c2171 pull_request completed success
38082629376 ci claude/n8-session-store 43c2171 pull_request completed success
38082626725 ci claude/n8-session-store 43c2171 push completed success
38082585007 server-cpu claude/n8-session-store 750fe3d pull_request completed cancelled
38082584975 ci claude/n8-session-store 750fe3d pull_request completed success
38082581829 ci claude/n8-session-store 750fe3d push completed success
38082295327 server-cpu claude/n7-server-hybrid-cpu 090ca71 pull_request completed success
38082295320 ci claude/n7-server-hybrid-cpu 090ca71 pull_request completed success
38082292627 ci claude/n7-server-hybrid-cpu 090ca71 push completed success
38081957679 ci claude/research-landscape a2c1815 pull_request completed success
38081953672 ci claude/research-landscape a2c1815 push completed success
38081816686 ci claude/research-landscape 16bf0a5 pull_request completed success
38081812884 ci claude/research-landscape 16bf0a5 push completed success
38081793624 server-cpu claude/n8-session-store ebecad4 pull_request completed success
38081793592 ci claude/n8-session-store ebecad4 pull_request completed success
38081791581 ci claude/n8-session-store ebecad4 push completed success
38081791575 server-cpu claude/n8-session-store ebecad4 push completed success
38081168135 ci claude/research-landscape ec4008e pull_request completed success
38081164662 ci claude/research-landscape ec4008e push completed success
38080702361 ci claude/research-landscape 4d7c852 pull_request completed success
38080698171 ci claude/research-landscape 4d7c852 push completed success
38080628252 server-cpu claude/n8-session-store 33cf4e8 pull_request completed success
38080628233 ci claude/n8-session-store 33cf4e8 pull_request completed success
38080609883 server-cpu claude/n8-session-store 33cf4e8 push completed success
38080609846 ci claude/n8-session-store 33cf4e8 push completed success
38080354799 ci claude/relaxed-dijkstra-1d7i5o 259081e pull_request completed success
38080354794 engines-sm89 claude/relaxed-dijkstra-1d7i5o 259081e pull_request completed success
38080352780 ci claude/relaxed-dijkstra-1d7i5o 259081e push completed success
38079738202 ci claude/research-landscape 7a0bf02 pull_request completed success
38079738013 traces-real claude/research-landscape 7a0bf02 workflow_dispatch in_progress 
38079734013 ci claude/research-landscape 7a0bf02 push completed success
38076816839 ci claude/research-landscape 8981351 pull_request completed success
38076798213 traces-real claude/research-landscape 8981351 push completed failure
38076798166 ci claude/research-landscape 8981351 push completed success
38076515852 ci claude/relaxed-dijkstra-1d7i5o ca1d2a7 pull_request completed success
38076515851 engines-sm89 claude/relaxed-dijkstra-1d7i5o ca1d2a7 pull_request completed success
38076512811 ci claude/relaxed-dijkstra-1d7i5o ca1d2a7 push completed success
38076450758 ci claude/n7-server-hybrid-cpu 986519a push completed success
38076450727 server-cpu claude/n7-server-hybrid-cpu 986519a push completed success
38075939074 engines-sm89 claude/relaxed-dijkstra-1d7i5o bf1a72b pull_request completed cancelled
$ gh run view 38081953672 --json jobs   (ci, claude/research-landscape, a2c1815; шаги кроме служебных)
38081953672 ci claude/research-landscape a2c1815 success
  job 114300530685 ref-torch-cpu success
     deps (CPU torch) -> success
     DFlash2 oracle against the vendored vLLM functions -> success
  job 114300530830 lt-strata-sm89 success
     CUDA 13 toolkit + cuBLAS (no GPU on the runner) -> success
     lighttransport/Strata at the pinned commit + our patches (patches/lt-strata, ADR-009) -> success
     configure (sm_89 = RTX 4080 Super) -> success
     build the GLM decoder, the model inspector and the CPU tests -> success
     decoder binary contains sm_89 code -> success
     CPU-only tests of the fork (GGUF reader, model description) -> success
     probe - does base B open our tiny GLM GGUF (not the 4096/2048 geometry)? Exploratory, never fails the job -> success
  job 114300530843 strata-glm-sm89 (89) success
     CUDA 13 toolkit + cuBLAS (no GPU on the runner) -> success
     strata-glm at the pinned commit + our patches (patches/strata-glm, ADR-007) -> success
     Linux disk tier (patch 0003) functional test, CPU -> success
     configure (sm_89 = RTX 4080 Super) -> success
     build strata-glm -> success
     binary contains sm_89 code -> success
     new flags are validated before any CUDA call (no GPU needed) -> success
  job 114300530849 strata-glm-sm89 (86) success
     CUDA 13 toolkit + cuBLAS (no GPU on the runner) -> success
     strata-glm at the pinned commit + our patches (patches/strata-glm, ADR-007) -> success
     Linux disk tier (patch 0003) functional test, CPU -> success
     configure (sm_89 = RTX 4080 Super) -> success
     build strata-glm -> success
     binary contains sm_89 code -> success
     new flags are validated before any CUDA call (no GPU needed) -> success
  job 114300530859 cuda-compile success
     CUDA toolkit (compile only, no GPU on the runner) -> success
     compile kernels for sm_89 -> success
  job 114300530885 cpu-parity success
     llama.cpp at the pinned commit -> success
     deps -> success
     unit tests -> success
     ggml delta-net parity (CPU) -> success
     G1 tiny-model parity against the real glm5-next graph (CPU) -> success
     MiMo G1 tiny-model parity against the real mimo2 graph (CPU) -> success
     N2 qwen4exp (candidate 1) tiny-model parity against the real qwen4exp graph (CPU) -> success
     N6 qwen4exp several ubatches, token-by-token decode, rollback and MTP against the real qwen4exp graph (CPU) -> success
     G0 per-token routing tracer + cache simulator on the tiny glm5-next model (CPU smoke) -> success
     G0 trace + probe on a tiny qwen3moe (last layer keeps only output rows; predict_eval self-checks) (CPU smoke) -> success
     roofline, cache-simulator and GGUF-audit unit tests (type table checked against the pinned gguf-py) -> success
$ gh run view 38081791575 --json jobs   (server-cpu, claude/n8-session-store, ebecad4)
38081791575 server-cpu claude/n8-session-store ebecad4 success
  job 114300045445 n7 success
     deps -> success
     harness tests against a stand-in server (fault injection) -> success
     llama.cpp at the pinned commit (tools/engines/pins.json) -> success
     N7 llama-server checks on tiny qwen4exp and glm5-next (CPU) -> success
     N8 corrupted slot files - raw llama-server vs tools/kv/session_store.py (CPU) -> success
     keep results and server logs -> success
$ gh run view 38080354794 --json jobs; gh api …/check-runs/<job>/annotations   (engines-sm89, claude/relaxed-dijkstra-1d7i5o, 259081e)
38080354794 engines-sm89 claude/relaxed-dijkstra-1d7i5o 259081e success
  job 114295815094 build (ik_llama.cpp) success
  job 114295815212 build (exllamav3) success
  job 114295815251 build (strata) success
  job 114295815284 build (llama.cpp) success
ik_llama.cpp 5194a9e1451b sm_89 ok Build cuda_13.0.r13.0/compiler.36424714_0
exllamav3 151539c77abc sm_89 ok Build cuda_13.0.r13.0/compiler.36424714_0
strata fb58e0dbc839 sm_89 ok Build cuda_13.0.r13.0/compiler.36424714_0
llama.cpp 79e2e74eb110 sm_89 ok Build cuda_13.0.r13.0/compiler.36424714_0
$ gh api repos/samnolak/408032gb/check-runs/<job>/annotations   (notice, выдержка)
114300530685 ref-torch-cpu (a2c1815):   dflash2 oracle vs vLLM: Ran 7 tests in 0.048s OK  torch 2.14.1+cpu
114300530843 strata-glm-sm89 (89) (a2c1815):
  ok: --pack p --tokens t.txt --layer-split 14,25,35 --gpus 0,1 -> 2
  ok: --pack p --tokens t.txt --layer-split 25,14 -> 2
  ok: --pack p --tokens t.txt --layer-split 50 -> 2
  ok: --pack p --tokens t.txt --bogus -> 2
  cubin archs:      26 sm_89
  strata-glm build OK, 69 CUDA objects
114300530830 lt-strata-sm89 (a2c1815):
  strata-model-inspect, tiny GGUF arch=glm5-next (4157600 bytes): exit 1
  arch=glm5next output head: model: missing or invalid integer glm5next.expert_group_count|
  strata-model-inspect, tiny GGUF arch=glm5next (4157568 bytes): exit 1
  model_test: 1/1 Test #2: model_test .......................   Passed    0.00 sec 100% tests passed, 0 tests failed out of 1
  gguf_reader_test: 1/1 Test #1: gguf_reader_test .................   Passed    0.00 sec 100% tests passed, 0 tests failed out of 1
114296831693 lt-strata-sm89 (4d7c852):  cubin archs:      18 sm_89 | lt-strata build OK, 62 CUDA objects
114296831708 strata-glm-sm89 (86) (4d7c852):  cubin archs:      26 sm_86
$ gh run view 38079738013 --json status,jobs   (traces-real, claude/research-landscape, 7a0bf02)
in_progress
  7 smoke - the same tools on a tiny qwen3moe before any download (scripts/traces/smoke_tiny.sh) completed success
  8 traces, probes, predictors and cache policies per model in_progress
$ gh run view <id> --log
failed to get run log: Get "https://results-receiver.actions.githubusercontent.com/rest/runs/…/logs?…": Forbidden   (URL сокращён)
```


### 4.3 Полный набор `unittest` в трёх рабочих деревьях

Первый прогон (без `-v`, `TMPDIR=$S/corr/tmp`); на research в нём участвовал тогда ещё не закоммиченный
`tests/test_merge_matrix.py` (закоммичен позже в `c95c670`), n7 и n8 потом получили слияние PR #1 — поэтому ниже есть второй
прогон на HEAD при написании документа.
```
$ for w in wt-research 408032gb wt-n8; do cd /home/user/$w; python3 -m unittest discover -s tests; done
== wt-research 7e534c9 claude/research-landscape
rc=0 seconds=118
----------------------------------------------------------------------
Ran 522 tests in 118.042s

OK (skipped=86)
== 408032gb 986519a claude/n7-server-hybrid-cpu
rc=0 seconds=162
----------------------------------------------------------------------
Ran 447 tests in 161.925s

OK (skipped=86)
== wt-n8 ebecad4 claude/n8-session-store
rc=0 seconds=170
----------------------------------------------------------------------
Ran 465 tests in 170.121s

OK (skipped=86)
```
Второй прогон (`-v`, причины пропусков посчитаны `grep -oE "\.\.\. skipped '.*'" | sort | uniq -c`):
```
$ for w in wt-research 408032gb wt-n8; do cd /home/user/$w; python3 -m unittest discover -s tests -v; done   # + grep -oE "\.\.\. skipped '.*'" | sort | uniq -c
== wt-research a2c1815 claude/research-landscape untracked=0
rc=0 seconds=113
Ran 522 tests in 112.590s

OK (skipped=86)
     40 'set QWEN4EXP_DECODE_DUMP, QWEN4EXP_TINY_GGUF, LLAMA_CPP_GGUF_PY: run tools/tinygen/run_parity_qwen4exp_decode.sh'
     17 'set QWEN4EXP_TINY_DUMP, QWEN4EXP_TINY_GGUF, LLAMA_CPP_GGUF_PY: run tools/tinygen/run_parity_qwen4exp.sh'
      8 'run tools/tinygen/run_parity.sh'
      6 'set QWEN4EXP_DECODE_DUMP, QWEN4EXP_MTP_GGUF, LLAMA_CPP_GGUF_PY: run tools/tinygen/run_parity_qwen4exp_decode.sh'
      5 'run tools/tinygen/run_parity.sh mimo'
      3 'set GGML_GDN_HARNESS (see tests/ggml_parity/build.sh)'
      2 'torch is not installed: the vendored vLLM functions cannot be executed'
      2 'set LLAMA_CPP_GGUF_PY=<llama.cpp>/gguf-py'
      1 'shellcheck not available (env SHELLCHECK or PATH)'
      1 'set LLAMA_CPP_FULL_CLONE=<llama.cpp clone with history> to check ancestry'
      1 'nvcc not available (set NVCC=/path/to/nvcc)'
== 408032gb 090ca71 claude/n7-server-hybrid-cpu untracked=0
rc=0 seconds=169
Ran 461 tests in 169.004s

OK (skipped=86)
     40 'set QWEN4EXP_DECODE_DUMP, QWEN4EXP_TINY_GGUF, LLAMA_CPP_GGUF_PY: run tools/tinygen/run_parity_qwen4exp_decode.sh'
     17 'set QWEN4EXP_TINY_DUMP, QWEN4EXP_TINY_GGUF, LLAMA_CPP_GGUF_PY: run tools/tinygen/run_parity_qwen4exp.sh'
      8 'run tools/tinygen/run_parity.sh'
      6 'set QWEN4EXP_DECODE_DUMP, QWEN4EXP_MTP_GGUF, LLAMA_CPP_GGUF_PY: run tools/tinygen/run_parity_qwen4exp_decode.sh'
      5 'run tools/tinygen/run_parity.sh mimo'
      3 'set GGML_GDN_HARNESS (see tests/ggml_parity/build.sh)'
      2 'torch is not installed: the vendored vLLM functions cannot be executed'
      2 'set LLAMA_CPP_GGUF_PY=<llama.cpp>/gguf-py'
      1 'shellcheck not available (env SHELLCHECK or PATH)'
      1 'set LLAMA_CPP_FULL_CLONE=<llama.cpp clone with history> to check ancestry'
      1 'nvcc not available (set NVCC=/path/to/nvcc)'
== wt-n8 43c2171 claude/n8-session-store untracked=0
rc=0 seconds=183
Ran 479 tests in 182.042s

OK (skipped=86)
     40 'set QWEN4EXP_DECODE_DUMP, QWEN4EXP_TINY_GGUF, LLAMA_CPP_GGUF_PY: run tools/tinygen/run_parity_qwen4exp_decode.sh'
     17 'set QWEN4EXP_TINY_DUMP, QWEN4EXP_TINY_GGUF, LLAMA_CPP_GGUF_PY: run tools/tinygen/run_parity_qwen4exp.sh'
      8 'run tools/tinygen/run_parity.sh'
      6 'set QWEN4EXP_DECODE_DUMP, QWEN4EXP_MTP_GGUF, LLAMA_CPP_GGUF_PY: run tools/tinygen/run_parity_qwen4exp_decode.sh'
      5 'run tools/tinygen/run_parity.sh mimo'
      3 'set GGML_GDN_HARNESS (see tests/ggml_parity/build.sh)'
      2 'torch is not installed: the vendored vLLM functions cannot be executed'
      2 'set LLAMA_CPP_GGUF_PY=<llama.cpp>/gguf-py'
      1 'shellcheck not available (env SHELLCHECK or PATH)'
      1 'set LLAMA_CPP_FULL_CLONE=<llama.cpp clone with history> to check ancestry'
      1 'nvcc not available (set NVCC=/path/to/nvcc)'
```

### 4.4 Шаги `cpu-parity` локально (research, llama.cpp `ec7630a`)

Выдержка из вывода обёртки: строки `== шаг`, `Ran`/`OK` и ключевые строки скриптов дословно; пустые строки, разделители `---` и
строки `artifacts:` опущены; комментарии после `#` добавлены при составлении. Полные логи — `$S/corr/par/*.log`.
```
$ bash $S/corr/run_cpu_parity.sh     # те же команды, что ci.yml:19-41; build dir $S/corr/build-tools; gdn_harness слинкован с её libggml*.so
repo a2c1815; llama.cpp ec7630a640789c393694fb194f1bbbf0369fc62d
== ggml_gdn rc=0 s=1            # GGML_GDN_HARNESS=… LLAMA_CPP_GGUF_PY=… python3 -m unittest tests.test_kda_vs_ggml tests.test_fitplan -v
Ran 18 tests in 0.322s
OK
== glm_parity rc=0 s=2          # tools/tinygen/run_parity.sh $S/llama-ec7630a $S/corr/build-tools
Ran 8 tests in 0.267s
OK
== mimo_parity rc=0 s=1         # tools/tinygen/run_parity_mimo.sh …
Ran 5 tests in 0.076s
OK
== qwen4exp_parity rc=0 s=2     # tools/tinygen/run_parity_qwen4exp.sh …
Ran 17 tests in 0.387s
OK
== qwen4exp_decode rc=0 s=8     # tools/tinygen/run_parity_qwen4exp_decode.sh …
ub16: 4 graphs, 1 calls, n_rs_seq 0
ub10: 7 graphs, 1 calls, n_rs_seq 0
decode: 17 graphs, 17 calls, n_rs_seq 0
rollback: 16 graphs, 17 calls, n_rs_seq 3
mtp: 11 graphs, 9 calls, n_rs_seq 0
Ran 54 tests in 3.718s
OK
== trace_smoke rc=0 s=2         # tools/g0/run_trace_smoke.sh …
trace checks OK: 5 files, prompt part of generation traces == prompt-only traces
== smoke_tiny rc=0 s=1          # scripts/traces/smoke_tiny.sh … $S/corr/smoke-tiny
{"router_logits_max_rel_err": 0.00027022845461033285, "selection_rule_match": 1.0, "gating": 1, "groups": [0, 0], "probe_equals_trace_rows": 1.0}
-b 64 vs -b 1: 2334/2340 (token, layer) rows select the same experts
smoke ok
== tools_gguf rc=0 s=3          # LLAMA_CPP_GGUF_PY=… python3 -m unittest tests.test_roofline tests.test_cachesim tests.test_gguf_quant_audit tests.test_offline_weights -v
Ran 52 tests in 2.170s
OK
$ cd $S/corr/par && grep -c skipped *.log    # ни один тест паритета не пропущен
ggml_gdn.log:0
glm_parity.log:0
mimo_parity.log:0
qwen4exp_decode.log:0
qwen4exp_parity.log:0
smoke_tiny.log:0
tools_gguf.log:0
trace_smoke.log:0
```
Худшие ошибки из таблиц, которые печатают сами тесты N2/N6 (`grep`/`awk` по логам; повторяющиеся пробелы сжаты):
```
e2e logits                                          -1  1.490e-06  7.439e-07  (55, 87)                       # N2
max rel (runs): rollback final_output (gated norm) [full]   6    12  6.621e-06  5.422e-06                       # N6
max rel (MTP): hc_inject (attention mixer) [chunk]          8    11  1.786e-06  1.577e-06                       # N6 MTP
```
Тест GLM таблицу не печатает; ошибки сняты вспомогательным скриптом, который подменяет только `assertClose` (записывает max abs
и нормированную rel) и запускает те же 8 тестов на тех же дампах:
```
$ GLM_TINY_DUMP=… GLM_TINY_GGUF=… LLAMA_CPP_GGUF_PY=… python3 -I $S/corr/glm_errs.py /home/user/wt-research
Ran 8 tests in 0.299s
OK
check                          n     tol    max abs    max rel worst layer
kda gate                       9   1e-05  6.837e-07  1.413e-07 2
kda scan                       9   1e-04  6.859e-08  2.580e-07 4
kda q conv                     9   1e-05  8.808e-07  2.065e-07 6
kda k conv                     9   1e-05  7.479e-07  1.722e-07 2
kda v conv                     9   1e-05  8.920e-07  1.841e-07 4
hc_post                       12   1e-05  9.313e-07  4.657e-07 7
hc_comb                       12   1e-05  5.322e-07  5.323e-07 10
hc mixer input                12   1e-05  2.783e-06  1.404e-07 10
hc_attn_post                  12   1e-05  2.346e-06  2.877e-07 7
swiglu clamp                  11   1e-05  2.121e-08  1.363e-07 4
indexer key                    3   1e-04  5.160e-07  1.800e-07 -
pooled keys                    3   1e-04  4.953e-07  2.892e-07 -
indexer query                  3   1e-04  3.254e-07  8.880e-08 -
indexer scores                 3   1e-04  5.029e-07  2.552e-07 -
q_resid                        3   1e-05  5.381e-07  1.607e-07 -
q absorbed                     3   1e-05  7.298e-07  1.830e-07 -
latent                         3   1e-05  4.090e-07  1.360e-07 -
kqv_out                        3   1e-04  9.021e-07  2.657e-07 -
attn_out                       3   1e-04  5.284e-07  1.881e-07 -
```
(DSA-строки — слои 3, 7, 11; номер слоя в сообщении теста не передаётся. Роутер сравнивается `assertAlmostEqual` и в таблицу не
попадает.)

### 4.5 Шаги `server-cpu` локально (n8, llama.cpp `79e2e74`)
```
$ bash $S/corr/run_server_cpu.sh     # tools/srvcheck/run_n7.sh $S/llama.cpp $S/corr/build-n7 $S/corr/n7-out; затем slot_corruption_probe.py для обеих моделей
repo ebecad4; llama.cpp 79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6
== run_n7.sh rc=0 s=253
llama.cpp 79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6 (pins.json)
version: 0.6.0-dev (build 1, commit 79e2e74)
== qwen4exp gate (-c 1024 -np 1 -t 2 -fa off -ctk f32 -ctv f32 --no-warmup)
  "all_pass": true
== glm5next gate (-c 1024 -np 1 -t 2 -fa off -ctk f32 -ctv f32 --no-warmup)
  "all_pass": true
== qwen4exp characterization (-c 1024 -np 1 -t 2 --no-warmup)
  "all_pass": true
== glm5next characterization (-c 1024 -np 1 -t 2 --no-warmup)
  "all_pass": false
artifacts: $S/corr/n7-out
$ python3 tools/srvcheck/n7_report.py lines $S/corr/n7-out/qwen4exp-gate.json $S/corr/n7-out/glm5next-gate.json $S/corr/n7-out/qwen4exp-defaults.json $S/corr/n7-out/glm5next-defaults.json | grep -E "^## |^verdict|^prefix_extend|^isolation|^mtp  "   (выдержка: блоки glm5next без строки mtp; у qwen4exp оба verdict — all_pass true)
## glm5next.gguf: server_args -c 1024 -np 1 -t 2 -fa off -ctk f32 -ctv f32 --no-warmup
verdict {"determinism": true, "prefix_extend_correct": true, "prefix_diverge_correct": true, "slot_persist_same_process": true, "slot_persist_after_restart": true, "cache_reuse_correct": true, "isolation_two_slots": true, "server_exit_clean": true, "mtp_equals_plain_decode": true, "all_pass": true}
prefix_extend    n_next 256 reusable 232 warm prompt_n 25 cache_n 231 reused 231 | max|d| 7.152557373046875e-06 same_tokens True
isolation        server max_busy_slots 2 client overlap 0.158 s same_tokens [True, True] max|d| [4.458427429199219e-05, 1.5735626220703125e-05]
## glm5next.gguf: server_args -c 1024 -np 1 -t 2 --no-warmup
verdict {"determinism": true, "prefix_extend_correct": false, "prefix_diverge_correct": true, "slot_persist_same_process": true, "slot_persist_after_restart": true, "cache_reuse_correct": true, "isolation_two_slots": false, "server_exit_clean": true, "mtp_equals_plain_decode": true, "all_pass": false}
prefix_extend    n_next 256 reusable 232 warm prompt_n 25 cache_n 231 reused 231 | max|d| 0.026694536209106445 same_tokens True
isolation        server max_busy_slots 2 client overlap 0.178 s same_tokens [True, True] max|d| [0.001149892807006836, 0.0018963813781738281]
== n8 probe qwen4exp rc=0 s=1
intact           raw: accepted prompt_n   4 same_tokens True  max|d| 0         nulls 0   | store: restored                  prompt_n   4 same True
exp@0.2          raw: accepted prompt_n   4 same_tokens False max|d| inf       nulls 25  | store: refused  sha256 mismatch  prompt_n 200 same True
exp@0.3          raw: accepted prompt_n   4 same_tokens False max|d| 0.639     nulls 0   | store: refused  sha256 mismatch  prompt_n 200 same True
exp@0.4          raw: accepted prompt_n   4 same_tokens False max|d| 0.736     nulls 0   | store: refused  sha256 mismatch  prompt_n 200 same True
truncated-1000   raw: accepted prompt_n 200 same_tokens True  max|d| 0         nulls 0   | store: refused  size 255724 != 256724 prompt_n 200 same True
== n8 probe glm5next rc=0 s=3
exp@0.1          raw: accepted prompt_n   4 same_tokens False max|d| 1.18      nulls 0   | store: refused  sha256 mismatch  prompt_n 200 same True
exp@0.6          raw: accepted prompt_n   4 same_tokens False max|d| 0.529     nulls 0   | store: refused  sha256 mismatch  prompt_n 200 same True
truncated-1000   raw: accepted prompt_n 200 same_tokens True  max|d| 0         nulls 0   | store: refused  size 286020 != 287020 prompt_n 200 same True
```
(выдержка из 2 × 14 строк; все 26 повреждённых файлов: сырой сервер — `accepted`, хранилище — `refused`, после отказа
`same True`; построчно совпадает с таблицей `docs/evidence/2026-10-10-n8-session-store.md` раздел 2.) Рабочее дерево n8 во время
прогона сдвинулось на `750fe3d` (слияние PR #1); `tools/srvcheck/`, `tools/kv/`, `tools/tinygen/` этим слиянием не менялись
(`git diff --stat ebecad4 750fe3d` — 18 файлов, ни одного из этих каталогов).

### 4.6 Окружение, пины, прочее
```
$ ls kernels
ls: cannot access 'kernels': No such file or directory
$ which nvidia-smi nvcc; ls /dev/nvidia*
which rc=1
ls: cannot access '/dev/nvidia*': No such file or directory
$ python3 -c "import torch"
ModuleNotFoundError: No module named 'torch'
$ grep -m1 "model name" /proc/cpuinfo; nproc
model name	: Intel(R) Xeon(R) Processor @ 2.80GHz
4
$ grep -E "^GGML_(AVX2|AVX512|NATIVE):" <build-tools>/CMakeCache.txt
GGML_AVX2:BOOL=ON
GGML_AVX512:BOOL=OFF
GGML_NATIVE:BOOL=OFF
$ for f in <graph files>; do diff <ec7630a>/$f <79e2e74>/$f | grep -c "^[<>]"; done
src/models/glm5-next.cpp 312
src/models/qwen4exp.cpp 140
src/models/mimo2.cpp 29
src/models/delta-net-base.cpp 0
src/llama-graph.cpp 321
src/llama-memory-hybrid-idx.cpp 184
src/llama-memory-recurrent.cpp 4
ggml/src/ggml-cpu/ops.cpp 37
$ git -C <ec7630a> rev-parse HEAD; git -C <79e2e74> rev-parse HEAD
ec7630a640789c393694fb194f1bbbf0369fc62d
79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6
$ git log --all --oneline -- tests/test_srvcheck_llama.py | wc -l   (wt-research, wt-n8)
0
0
$ grep -nE "^class (Q4_K|Q5_K|Q6_K|Q8_0|IQ4_XS|MXFP4)\b|^def dequantize" <ec7630a>/gguf-py/gguf/quants.py
68:def dequantize(data: np.ndarray, qtype: GGMLQuantizationT
379:class Q8_0(__Quant, qtype=GGMLQuantizationType.Q8_0):
476:class Q4_K(__Quant, qtype=GGMLQuantizationType.Q4_K):
526:class Q5_K(__Quant, qtype=GGMLQuantizationType.Q5_K):
553:class Q6_K(__Quant, qtype=GGMLQuantizationType.Q6_K):
657:class MXFP4(__Quant, qtype=GGMLQuantizationType.MXFP4):
1352:class IQ4_XS(__Quant, qtype=GGMLQuantizationType.IQ4_XS
```

### 4.7 Проба пробела 1: те же скрипты паритета на llama.cpp `79e2e74` (не существующая проверка, разовый прогон)
```
$ bash $S/corr/run_pin_probe.sh      # cmake -S tools/g0 -B $S/corr/build-tools-79 -DLLAMA_CPP_DIR=$S/llama.cpp …; затем run_parity*.sh с ним
repo a2c1815; llama.cpp 79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6
== build dump-tensors @79e2e74 rc=0 s=157
[100%] Built target dump-tensors
== glm_parity rc=0 s=1
Ran 8 tests in 0.229s
OK
== mimo_parity rc=0 s=1
Ran 5 tests in 0.076s
OK
== qwen4exp_parity rc=0 s=2
Ran 17 tests in 0.327s
OK
== qwen4exp_decode rc=1 s=7
FAIL: test_qsa_pool_cache_and_selection (tests.test_qwen4exp_decode_parity.Decode.test_qsa_pool_cache_and_selection)
AssertionError: 1.3864799363845095 not less than 0.0001 : decode indexer_pool_k_new (padding rows) L3 graph 1: rel err 1.39e+00 (abs 3.06e+00)
FAIL: test_qsa_pool_cache_and_selection (tests.test_qwen4exp_decode_parity.Rollback.test_qsa_pool_cache_and_selection)
AssertionError: 1.3627376980950934 not less than 0.0001 : rollback indexer_pool_k_new (padding rows) L3 graph 2: rel err 1.36e+00 (abs 2.87e+00)
Ran 54 tests in 3.231s
FAILED (failures=2)
$ grep -E "^(decode|rollback) +logits.bin \[full\]" $S/corr/pin79/qwen4exp_decode.log
decode   logits.bin [full]                                             -1    10  1.387e-06  1.125e-06
rollback logits.bin [full]                                             -1     0  1.506e-06  8.570e-07
$ QWEN4EXP_DECODE_DUMP=… python3 -I $S/corr/qsa_nopad.py /home/user/wt-research   # тот же тест, только сравнение «padding rows» не утверждается
Ran 2 tests in 0.616s
OK
padding-row comparisons not asserted: 25 max rel 2.053132109357617
decode   qsa selection: tie replays = 0                                 3    -1  0.000e+00  0.000e+00
rollback qsa selection: tie replays = 0                                 3    -1  0.000e+00  0.000e+00
$ python3 - <последний $S/corr/tmp/tmp.*/n6>/decode   # max|row| по строкам дампа indexer_pool_k_new-3 в graph 0…3 (79e2e74)
g000 indexer_pool_k_new-3__0 ne [16, 13, 1, 1] max|row|: [2.39, 2.06, 2.31, 2.33, 1.96, 1.91, 2.18, 1.82, 2.64, 2.44, 2.4, 2.72, 1.84]
g001 indexer_pool_k_new-3__0 ne [16, 1, 1, 1] max|row|: [1.84]
g002 indexer_pool_k_new-3__0 ne [16, 1, 1, 1] max|row|: [1.84]
g003 indexer_pool_k_new-3__0 ne [16, 1, 1, 1] max|row|: [1.84]
```
(Полный лог — `$S/corr/pin79/qwen4exp_decode.log`; дампы — последний `$S/corr/tmp/tmp.*/n6`.)
