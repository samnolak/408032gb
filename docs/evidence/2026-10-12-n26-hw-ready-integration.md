# N26 — интеграционная подготовка к железу (2026-10-12)

Роль: lead-architect (сессия `session_01EyB9XrhA3NTRjjQx1A9qMh`), исполнители N26a (engine-integrator) и N26b (perf-bench).
Ветка `claude/n26-hw-ready` от `claude/n25-integration` (`80df84f`). Цель — **READY FOR HARDWARE VALIDATION**, не эксплуатация.
GPU и сервер в этом этапе не использовались: всё ниже — CPU, песочница и CI GitHub. Решения владельца этапа записаны в
`docs/DECISIONS.md` ADR-012.3.

## 1. Граф Draft PR #1–#16 (не линейная цепочка)

CONFIRMED (GitHub `list_pull_requests`, 2026-10-12, база → голова):

```
main eb59224
└─ #1  claude/relaxed-dijkstra-1d7i5o 37415fe   (ведёт другая сессия; #2 и #3 созданы на её 259081e)
   ├─ #2  claude/n7-server-hybrid-cpu 4f7be84
   │   ├─ #4  claude/n8-session-store 09eb438
   │   └─ #7  claude/n11-r1-tools-cpu-server b74efde
   └─ #3  claude/research-landscape c8f3547
       └─ #5  n9 af4db13 → #6 n10 0d4d582 → #8 n12 5730cc4 → #9 n13 6dd6b28 → #10 n14 b2096c1
           ├─ #12 n15 67bd5ba → #13 n18 35ba5ba
           └─ #11 n16 0b0eace → #14 n17 3ce2293 → #15 n19 e71b241 → #16 n25-integration 80df84f
                                                                  (N20–N24 влиты в N25 слиянием)
```

Что не входило в N25 (`git log --no-merges origin/claude/n25-integration..<ветка>`, 2026-10-12):
- research-landscape: 16 коммитов (обновления `RESEARCH-PHASE-2026-10-10.md`, трассы real-model);
- n14: 1 коммит (`b2096c1`, запись CI N14);
- n15 + n18: 45 коммитов (заголовки GGUF; ветка N15/N18 вне цепочки N16–N19; исправлено по аудиту, M1);
- n7, n8, n11: вся ветка N7 (+N8, +N11), включая `4f7be84`;
- PR #1: 11 коммитов после `259081e`, из них 6 — `cherry-pick -x` исправлений N11.

## 2. Слияния и конфликты

Порядок: research-landscape, n14, n18 (с n15), n8 (с n7), n11, PR #1 `37415fe`; затем N26b, N26a. Предварительная проверка
`git merge-tree --write-tree --name-only HEAD <ветка>`: конфликты только у n8 и n11.

| слияние | коммит | конфликты | как разрешено |
|---|---|---|---|
| research-landscape | `a0bcaa6` | нет | — |
| n14 | `2265694` | нет | — |
| n18 (с n15) | `ee4a772` | нет | — |
| n8 (с n7) | `e3e37d1` | `tools/tinygen/glm5next_tiny.py`, `docs/evidence/mutation-scripts/README.md` | генератор — сторона интеграции: в ней уже есть блок `--mtp` N7 (`n_all`, `dsa_all`, `nextn.*`) и `--quant-dims` N10, `--dialect ik` N12; README — объединение таблиц |
| n11 | `0e24585` | `server-cpu.yml`, `ROADMAP.md`, `glm5next_tiny.py`, `qwen4exp_tiny.py` | объединение: шаги/триггеры/артефакты N8 и N11; строки N8 и N11; флаги `--quant-dims`, `--dialect ik`, `--text-vocab`, `--ctx` вместе |
| PR #1 `37415fe` | `c1ffb80` | нет | — |
| N26b | `f622f3c` | нет | — |
| N26a | `4950fd0` | нет | — |

**Генераторы tiny-моделей после ручного слияния** — побайтно те же файлы, что на исходных ветках (CONFIRMED; gguf-py
`ec7630a`; первые 12 знаков sha256; `q` = qwen4exp, `g` = glm5next; пустой флаг = по умолчанию; `tv` = `--text-vocab`):

```
0fe744212110 q.wt-n11..gguf          0fe744212110 q.wt-n25..gguf          0fe744212110 q.wt-n26..gguf
23bf70ffd45b q.wt-n11.--mtp.gguf     23bf70ffd45b q.wt-n25.--mtp.gguf     23bf70ffd45b q.wt-n26.--mtp.gguf
26c83267fe1a g.wt-n11..gguf          26c83267fe1a g.wt-n25..gguf          26c83267fe1a g.wt-n26..gguf
544d1f627916 g.wt-n11.--text-vocab   544d1f627916 g.wt-n26.--text-vocab
5a3e071e0cc9 q.wt-n25.--quant-dims   5a3e071e0cc9 q.wt-n26.--quant-dims
6050ee362d63 g.wt-n25.--dialect_ik   6050ee362d63 g.wt-n26.--dialect_ik
955f7ba3ea01 q.wt-n11.tv             955f7ba3ea01 q.wt-n26.tv
bc3f35128ba1 g.wt-n25.--quant-dims   bc3f35128ba1 g.wt-n26.--quant-dims
f7833b6ce213 g.wt-n11.--mtp          f7833b6ce213 g.wt-n25.--mtp          f7833b6ce213 g.wt-n26.--mtp
```
Отсутствующие строки — флага нет на той ветке (например, `--text-vocab` на n25).

**Дублирующиеся изменения.** PR #1 несёт шесть `cherry-pick -x` исправлений N11:
- `026b0e9` → `99d3359`
- `2d59d65` → `5dde94e`
- `92e3886` → `f775dbf`
- `0f23aa6` → `0b9eb6e`
- `ffec637` → `5edeb45`
- `6dff91a` → `0ea9e66`

Содержимое одинаковое, слияние прошло без конфликтов и без двойного применения. Проверено (CONFIRMED):
- `git diff` интеграции с N11 и с PR #1 по `tools/kv/slot_persist.py` и `tools/bench/run_matrix.py` пуст;
- `tools/quality/logprob_compare.py` = PR #1 + поля почти-ничьих N24;
- `scripts/r1/run_r1.py` = PR #1 + проверка правил N26a.

Коммиты N7 в N8 и N11 общие и вошли один раз. Отдельные PR ничего не переносилось cherry-pick'ом: ветки влиты слиянием
целиком, история сохранена (`3090` и `main` — предки, CONFIRMED ниже).

**Все ветки — предки интеграции** (CONFIRMED, на `4950fd0`):

```
in  origin/3090 986da38
in  origin/claude/n10-quant-fa-parity 0d4d582
in  origin/claude/n11-r1-tools-cpu-server b74efde
in  origin/claude/n12-ik-glm5next-cpu 5730cc4
in  origin/claude/n13-ci-no-silent-skips 6dd6b28
in  origin/claude/n14-chat-templates b2096c1
in  origin/claude/n15-gguf-headers 67bd5ba
in  origin/claude/n16-mimo-decode-parity 0b0eace
in  origin/claude/n17-mimo-mtp-heads-fa 3ce2293
in  origin/claude/n18-gguf-headers-mtp-mimo 35ba5ba
in  origin/claude/n19-qwen4exp-multiseq e71b241
in  origin/claude/n20-recurrent-rollback-guard b7d6af1
in  origin/claude/n21-mimo-mtp-heads-fix 93a48d6
in  origin/claude/n22-ci-false-green 56202c8
in  origin/claude/n23-glm-template-tokenizer 7541c4a
in  origin/claude/n24-owner-decisions 76e9d56
in  origin/claude/n25-integration 80df84f
in  origin/claude/n7-server-hybrid-cpu 4f7be84
in  origin/claude/n8-session-store 09eb438
in  origin/claude/n9-glm-decode-parity af4db13
in  origin/claude/relaxed-dijkstra-1d7i5o 37415fe
in  origin/claude/research-landscape c8f3547
in  origin/main eb59224
```

Оговорка: PR #1 ведёт другая сессия; коммиты в нём после `37415fe` в интеграцию не входят (на момент слияния их не было).

## 3. Содержимое патчей и сборка R1

- В `patches/llama.cpp/` лежат три патча:
  - N20 — `79e2e74/0001-recurrent-rollback-guard.patch` и `ec7630a/0001-recurrent-rollback-guard.patch`;
  - N21 — `0001-mtp-heads-slot-replay.patch`;
  - N24 — `n24-swiglu-clamp-after-silu.patch`, только для эксперимента.
- В сборку R1 не входит ни один: `scripts/engines/build_engine.py` патчей не применяет (PHASE-CLOSE (б), строки 4 и 7);
  правило записано в `AGENTS.md` «Upstream pins». Внедрены только конфигурационные защиты (раздел 4).
- **N20 + N21 на одной копии — теперь в CI:** job `n20-n21-combined` в `parity-r1-pin.yml`. Он делает так:
  - копия пина через `apply.sh` (N20), затем `git apply` патча N21;
  - драйвер N20 в новом режиме `QWEN4EXP_RS_GUARD_MODE=combined`: копия должна быть на пине и нести оба патча, иначе exit 1;
  - драйвер N21 на той же сборке;
  - ожидаемые числа — шагов `n20-qwen4exp-rs-guard` и `n21-mimo2-mtp-fix-patched`.

  Локально, на копии `79e2e74` из N25 (CONFIRMED):

```
n20 rc=0
::notice::n20-qwen4exp-rs-guard @ llama.cpp 79e2e74: Ran 110, skipped 0 as expected; the driver's checkout <S>/n25/llama-r1-both is at 79e2e74
real	2m25.747s
n21 rc=0
::notice::n21-mimo2-mtp-fix-patched @ llama.cpp 79e2e74: Ran 57, skipped 0 as expected; the driver's checkout <S>/n25/llama-r1-both is at 79e2e74
real	0m21.754s
```

  Отказы режима (CONFIRMED):

```
combined mode: <S>/n20/llama-r1-guard does not carry .../patches/llama.cpp/0001-mtp-heads-slot-replay.patch
rc=1
combined mode: <S>/llama.cpp does not carry .../patches/llama.cpp/79e2e74/0001-recurrent-rollback-guard.patch
rc=1
```

## 4. Правила безопасности R1 (N26a) и NVMe (N26b)

Подробно — `docs/evidence/2026-10-12-n26a-r1-safety-rules.md` и `docs/evidence/2026-10-12-n26b-nvme-3x1tb.md`. Кратко:
- **N26a, проверка до первого шага.** `scripts/r1/safety_rules.py` проверяет правила R-NP, R-GLM-TEMPLATE (sha256 шаблона
  `5d47ac8e48510b5c6543db4000ef145201823bf46bae9cf51ac6a9553a16ef5a`), R-MTP-FILES, R-MIMO-MTP, R-WEIGHTS и R-SYNTH. Конфиг с
  нарушением `run_r1.py` отвергает до первого шага (exit 2).
- **N26a, ворота весов.** Движок, кроме mock, стартует только после пройденного `weights_verify` или при явной причине
  `weights_unverified_reason`.
- **N26a, шлюз.** Шлюз убирает `"reasoning_content": null`. Проба на CPU: напрямую HTTP 500, через шлюз 200.
- **N26a, итоги проверок.** Мутации: 17/17 во втором круге. Тесты: 90.
- **N26b.** Заказчик уточнил: 3 × 1 ТБ = 3 ТБ. Исправлены живые документы, оба hardware JSON и `tier_cost.py` (`n_drives` = 3,
  в торнадо-диаграмме — сравнение 2 и 3 дисков), а также `arch_ceilings.py` (`nvme_gbs` = 21.0). Пересчитанные потолки
  DeepSeek-V4.1 (PROVISIONAL, модель):
  - конвейер по слоям, 8 каналов DRAM: 12.1 → 10.2 tok/s;
  - тензорный параллелизм (TP): 13.5 → 11.1 tok/s.

  У Qwen NVMe не на пути декода. Ёмкость: обязательные веса ≈ 1.35 ТБ из 3 ТБ (PROVISIONAL).
- **Решения владельца.** `AGENTS.md`, ADR-012.3, `ROADMAP.md` R4.3 (ОТКЛОНЕНО), `BIOS_PLAN.md` (правило 0, шаги 2 и 6),
  `CORRECTNESS.md` B33.

## 5. Тесты

Полный набор на объединённом дереве до N26a/N26b (после слияния PR #1, `c1ffb80`; CONFIRMED):

```
$ TMPDIR=<S>/n26/cwd LLAMA_CPP_GGUF_PY=<S>/llama-ec7630a/gguf-py python3 -m unittest discover -s tests
Ran 1360 tests in 252.012s

OK (skipped=609)
```

```
$ python3 -m unittest tests.test_check_unittest
Ran 33 tests in 14.470s

OK
```

Финальный прогон на голове — раздел 7. Пропуски (609) — тесты паритета и сборки, которым нужны дампы llama.cpp, CUDA или
`shellcheck` в окружении; в CI их гоняют шаги-обёртки `tools/ci/check_unittest.py`, которые падают, если число прогнанных и
пропущенных тестов не совпало с `tools/ci/parity_expectations.json` (N13). Обязательные CI-шаги и их фактическое исполнение —
раздел 6.

## 6. CI: обязательные шаги и пропуски

Заполняется по CI интеграционного Draft PR (раздел 8).

## 7. Финальный прогон

Заполняется перед публикацией.

## 8. Публикация

Заполняется после push.

## 9. Резервная копия

Заполняется после публикации.

## 10. Что требует железа (NOT RUN)

Без изменений к `PHASE-CLOSE-2026-10-11.md` (д):
- исполнение CUDA-кода и паритет на CUDA-бэкенде;
- все замеры скорости, включая 262K;
- калибровка V6 и KLD;
- SwiGLU на настоящих весах GLM;
- A/B принятия MTP у MiMo;
- патчи N20 и N21 на GPU;
- P2P, all-reduce, 32 GB на каждой карте;
- каналы DRAM, 3 NVMe (модели, скорость, топология), BIOS;
- загрузка и проверка весов;
- проверка файла на epyc-ai.
