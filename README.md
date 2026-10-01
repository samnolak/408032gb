# 408032gb

Порт приёмов [Strata](https://github.com/Niko1221/Strata) (ярусный MoE-инференс:
кэш экспертов в VRAM, остальные эксперты на CPU из RAM, n-граммные таблицы на SSD,
спекуляция через родной MTP) на сервер **4× RTX 4080 Super 32 GB**.

Целевые модели, по порядку (обоснование: [docs/DECISIONS.md](docs/DECISIONS.md)):

| # | модель | параметры | статус |
|---|---|---|---|
| 1 | GLM-5.3-Flash | 320B / 18B активных | G0 готовится |
| 2 | MiMo-V2.6-Flash | 309B / 15B активных | ждёт |
| 3 | DeepSeek-V4.1-Flash | 552B + 196B Engram | ждёт |

## Роли

Заказчик — Ribrad. Исполнитель — команда агентов, см. [AGENTS.md](AGENTS.md):
lead-architect (Sol), model-analyst (Terra), kernel-engineer (Sol), engine-integrator (Sol),
parity-qa (Terra), perf-bench (Terra), infra-ops (Terra), evidence-auditor (Luna).

## Где что лежит

| путь | что |
|---|---|
| `docs/PLAN.md` | гейты G0–G7, критерии приёмки, текущий статус |
| `docs/DECISIONS.md` | архитектурные решения (ADR) |
| `docs/strata-map.md` | карта кода Strata: что переиспользуем, что пишем заново |
| `docs/models/` | паспорта моделей с источниками |
| `docs/evidence/` | журналы проверок (литеральный вывод команд) |
| `docs/prompts/` | готовые промпты для агентов на VM |
| `models/`, `hardware/` | машиночитаемая геометрия моделей и железа |
| `tools/fitplan.py` | планировщик раскладки по VRAM/RAM/SSD |
| `tools/g0/` | Gate 0: гистограммы использования экспертов |
| `ref/` | эталоны на numpy для parity-тестов |

## Проверка

```
python3 -m unittest discover -s tests -v
python3 tools/fitplan.py models/glm-5.3-flash.json --quant IQ3_XXS --ctx 131072
```

## Метки доказательности

Каждая цифра — CONFIRMED (с источником), PROVISIONAL (оценка, указан способ) или UNKNOWN.
