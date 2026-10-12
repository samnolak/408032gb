# tasks-synthetic — SYNTHETIC-набор для стенда `tools/eval`

**SYNTHETIC. Это не доказательство качества какой-либо модели.** Задачи придуманы для проверки стенда
(`tests/test_eval.py`) и как шаблон для заказчика: каждая показывает одну возможность формата. Продукты, компании и
документы вымышлены. Все id содержат `SYNTHETIC`, у всех задач `"synthetic": true`; прогон на таком наборе помечается в
`summary.md` строкой «SYNTHETIC TASK SET».

Проверка набора без сервера: `python3 tools/eval/run_eval.py --tasks tools/eval/tasks-synthetic --validate-only`
(нетронутые фикстуры должны падать, эталонные ответы — проходить).

| id | категория | что показывает |
|---|---|---|
| `SYNTHETIC-code_fix-001` | code_fix | ошибка на единицу; эталон — unified diff; `{{fixture_tree}}`, `{{file:...}}` |
| `SYNTHETIC-code_fix-002` | code_fix | эталон — блок `FILE:` (файл целиком); `allowed_paths` |
| `SYNTHETIC-code_fix-003` | code_fix | пакет и `tests/`; команда `unittest discover`; в эталонном diff неверный номер строки (поиск контекста) |
| `SYNTHETIC-code_write-001` | code_write | реализовать функцию; тесты скрыты и копируются только при проверке (`check.overlay`) |
| `SYNTHETIC-code_write-002` | code_write | реализовать функцию по видимым тестам; эталон — diff |
| `SYNTHETIC-tool_use-001` | tool_use | заготовленные ответы, строгий порядок вызовов, запрещённый инструмент, итоговый ответ |
| `SYNTHETIC-tool_use-002` | tool_use | встроенный калькулятор стенда; проверяется число в итоговом ответе |
| `SYNTHETIC-tool_use-003` | tool_use | встроенная таблица (`lookup`), два вызова в любом порядке, запрет `place_order` |
| `SYNTHETIC-long_doc_qa-001` | long_doc_qa | действующее значение против устаревшего в начале документа; `number` |
| `SYNTHETIC-long_doc_qa-002` | long_doc_qa | имя заголовка с двумя похожими отвлекающими; `regex` |
| `SYNTHETIC-long_doc_qa-003` | long_doc_qa | одинаковый порог у двух разных процессов; `normalized` |
| `SYNTHETIC-structured_output-001` | structured_output | извлечение счёта; `response_format`; строгий разбор; проверки значений |
| `SYNTHETIC-structured_output-002` | structured_output | массив с enum-метками; мягкий разбор (```` ```json ````) |
| `SYNTHETIC-reasoning-001` | reasoning | вычисление времени; `normalized` по строке `ANSWER:` |
| `SYNTHETIC-reasoning-002` | reasoning | счёт с включением-исключением; `number` |

Каталоги: `fixtures/` — маленькие репозитории; `hidden/` — скрытые тесты (overlay); `docs/` — документы;
`reference/` — эталонные ответы (модели не показываются; нужны для `--validate-only` и для мок-тестов).
