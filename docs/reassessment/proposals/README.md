# Предложения, ждущие решения заказчика

Здесь лежат изменения, которые ведущий агент переоценки подготовил, но **не применил**, потому что они меняют
правила работы самих агентов (hooks, AGENTS.md, CLAUDE.md). Автоматический классификатор Claude Code 2026-10-09
отклонил такую правку как self-modification; правило «агент не расширяет себе права сам» сохранено.

## 0001 — пуш из облачной сессии только в ветки `claude/*`

**Статус:** ПРЕДЛОЖЕНО, НЕ ПРИМЕНЕНО. Хук `scripts/hooks/guard-bash.sh` в репозитории не изменён.

**Проблема.** Хук запрещает любой `git push` (правило для изолированной VM, которая синхронизируется бандлами). Переоценка идёт
в облачном контейнере Claude Code, который удаляется после сессии: без пуша работа сохраняется только бандлом, переданным
файлом в чат. Заказчик разрешил работать «в отдельной ветке/ветках» и запретил пуш в `main` без разрешения.

**Найденная дыра старого хука (CONFIRMED, литеральный вывод 2026-10-09):** глобальная опция git перед `push` обходит проверку.
```
old hook: 'git push -u origin claude/relaxed-dijkstra-1d7i5o' -> exit 2
old hook: 'git push origin main' -> exit 2
old hook: 'git -C /tmp push origin main' -> exit 0
```

**Что предлагается** (`0001-guard-bash-push-policy.patch`, тест `0001-test_hooks.py`):
1. разрешить только `git push [-u|--set-upstream] origin [HEAD:]claude/<имя>`;
2. запретить всё остальное: `main`, другие ветки и remote, `--force`/`-f`/`--force-with-lease`/`+refspec`, `--mirror`, `--delete`,
   `--tags`, опции git перед `push` (`-C`, `-c`) — это закрывает и найденную дыру;
3. блок переписывания истории не меняется; VM по-прежнему синхронизируется только бандлами.

Проверка предложенного хука в песочнице (до отката, 2026-10-09):
```
$ python3 -m unittest tests.test_hooks -v
test_allowed (tests.test_hooks.GuardBash.test_allowed) ... ok
test_blocked (tests.test_hooks.GuardBash.test_blocked) ... ok
Ran 2 tests in 1.366s
OK
```

**Как применить (заказчик):**
```
git apply docs/reassessment/proposals/0001-guard-bash-push-policy.patch
git mv docs/reassessment/proposals/0001-test_hooks.py tests/test_hooks.py
python3 -m unittest tests.test_hooks -v
```
и добавить в AGENTS.md («Git and sync») строку: «Cloud agent sessions may push only a plain fast-forward of a `claude/*` branch to
`origin`; never `main`, never force/delete/mirror». Если пуши агентам не нужны — применить только закрытие дыры `git -C`.
