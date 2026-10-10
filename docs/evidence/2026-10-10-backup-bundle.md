# Резервные git-бандлы: неглубокий клон, причина и исправление (2026-10-10)

Автор: lead-architect (сессия ведущего), роль кода — infra-ops. Ярлыки — по `AGENTS.md`.

## Что нашёл владелец
Независимый `git clone` из переданного бандла `ca1d2a7` падает: нет родительского объекта
`f11c8e50c2de2f32fd0c76fca27921d6825b08c2` у коммита `bef4d440`, хотя `git bundle verify` проходил. Резервная копия не
восстанавливалась.

## Воспроизведение (CONFIRMED, песочница; `<S>` — каталог песочницы)
Старый бандл в совершенно пустом каталоге, обычный git:
```
$ E=$(mktemp -d /tmp/empty-restore-XXXX); ls -A $E | wc -l
0
$ cd $E && git clone -q <S>/408032gb-claude-relaxed-dijkstra-ca1d2a7.bundle repo
error: Could not read f11c8e50c2de2f32fd0c76fca27921d6825b08c2
fatal: Failed to traverse parents of commit bef4d440902052d620c3ac847db2b450342ddfbb
fatal: remote did not send all necessary objects
git clone rc=128
```

## Причина (CONFIRMED)
Рабочий клон сессии был неглубоким (shallow): контейнер клонирует репозиторий с ограниченной историей.
```
$ git rev-parse --is-shallow-repository
true
$ wc -l < .git/shallow; head -3 .git/shallow
1
bef4d440902052d620c3ac847db2b450342ddfbb
$ git cat-file -t f11c8e50c2de2f32fd0c76fca27921d6825b08c2
fatal: git cat-file: could not get object info
$ git rev-list --count HEAD
185
```
`git bundle create` в таком клоне упаковывает историю только до границы `bef4d440`; заголовок бандла не перечисляет
недостающих предков, и `git bundle verify` внутри того же клона считает границу корнем («The bundle records a complete
history»). Восстановление в пустом каталоге требует родителя `f11c8e50…`, которого нет. Под это попадают все четыре бандла,
переданные 2026-10-10: `1be78af`, `96a476a`, `2dee43a`, `ca1d2a7` — **ни один из них не считать резервной копией**.

## Исправление
1. Клон дополнен до полной истории (CONFIRMED):
```
$ git fetch --unshallow -q origin; echo "rc=$?"
rc=0
$ git rev-parse --is-shallow-repository
false
$ test -f .git/shallow || echo "no .git/shallow"
no .git/shallow
$ git cat-file -t f11c8e50c2de2f32fd0c76fca27921d6825b08c2
commit
$ git rev-list --count HEAD; git rev-list --count main
191
68
$ git fsck --full     (без строк dangling)
fsck rc=0
```
2. `scripts/offline/repo_backup.py` (`125c9c3`): `create` отказывается работать в неглубоком клоне и сразу проверяет результат;
   `check` восстанавливает бандл в новом пустом каталоге: `git clone --mirror`, `git fsck --full`, коммит каждой ссылки равен
   заголовку бандла (и `--expect`), вся история каждой ссылки читается, затем обычный `git clone --branch` с checkout.
   Старый бандл этой проверкой ловится:
```
$ python3 scripts/offline/repo_backup.py check <S>/408032gb-claude-relaxed-dijkstra-ca1d2a7.bundle
bundle … (1357943 bytes, sha256 4c23d0275d7d9a65c170fabeff7783da04966b6ee70f8042d0e886425d804cde)
ok   bundle header                                  2 refs
FAIL git clone --mirror                             error: Could not read f11c8e50c2de2f32fd0c76fca27921d6825b08c2
fatal: Failed to traverse parents of commit bef4d440902052d620c3ac847db2b450342ddfbb
fatal: remote did not send all necessary objects
restore check: 1 check(s) failed
check rc=1
```
3. `tests/test_repo_backup.py` воспроизводит ошибку на маленьком репозитории: в неглубоком клоне `git bundle verify`
   проходит («complete history»), а `check` того же бандла падает с «did not send all necessary objects»; после
   `git fetch --unshallow` `create` работает; `create` в неглубоком клоне отказывает (код 2); полный клон восстанавливается
   со всей историей; `--expect` с чужим коммитом — ошибка.
```
$ python3 -m unittest tests.test_repo_backup -v      (хвост)
Ran 4 tests in 2.445s

OK
```
4. Правило в `.claude/skills/bundle-sync/SKILL.md`: не делать бандл из неглубокого клона; бандл считается проверенным
   только после восстановления в пустом каталоге (`repo_backup.py create` / `check`).

## Новый бандл и проверка восстановления (CONFIRMED)
Бандл на `125c9c3` (ветки `main` и `claude/relaxed-dijkstra-1d7i5o`) инструментом:
```
$ python3 scripts/offline/repo_backup.py create --out <S>/408032gb-claude-relaxed-dijkstra-125c9c3.bundle
bundle … (1368247 bytes, sha256 90d9afefb063a38a9ec7be921d9fab39baf258cdc16ce862c579fc8f11076c32) refs: main=eb59224c835581740a715ddcaa9abe101f90aa2f claude/relaxed-dijkstra-1d7i5o=125c9c397b0119aacf8600cac7aac2ba390effed
ok   bundle header                                  2 refs
ok   expected refs/heads/main                       eb59224c835581740a715ddcaa9abe101f90aa2f (want eb59224c835581740a715ddcaa9abe101f90aa2f)
ok   expected refs/heads/claude/relaxed-dijkstra-1d7i5o 125c9c397b0119aacf8600cac7aac2ba390effed (want 125c9c397b0119aacf8600cac7aac2ba390effed)
ok   git clone --mirror                             ok
ok   git fsck --full                                ok
ok   ref refs/heads/claude/relaxed-dijkstra-1d7i5o  125c9c397b01, 192 commits
ok   ref refs/heads/main                            eb59224c8355, 68 commits
ok   git clone --branch claude/relaxed-dijkstra-1d7i5o HEAD 125c9c397b01, checkout ok
restore check: ok
create rc=0
```
Отдельно, обычным git в совершенно пустом каталоге вне репозитория, с прогоном тестов из восстановленного дерева:
```
dir /tmp/empty-restore-nciz entries: 0
git clone rc=0
125c9c397b0119aacf8600cac7aac2ba390effed          (HEAD)
false                                             (is-shallow-repository)
192                                               (rev-list --count HEAD)
commit                                            (f11c8e50… теперь есть)
bef4d44 [parity-qa] ref: KDA/GDN gated-delta-rule oracle transcribed from ggml ops.cpp + glm5-next.cpp @ec7630a; 7 property tests
fsck rc=0
0                                                 (git status --short | wc -l)
Ran 14 tests in 11.081s                           (tests.test_repo_backup tests.test_offline_engines)

OK
```
(Пояснения в скобках добавлены здесь; команды — `git rev-parse HEAD`, `git rev-parse --is-shallow-repository`,
`git rev-list --count HEAD`, `git cat-file -t f11c8e50…`, `git log --oneline -1 bef4d440`, `git fsck --full --no-dangling`,
`git status --short | wc -l`, `python3 -m unittest …`.)

Окончательный бандл для владельца делается тем же инструментом на голове ветки после этого коммита; вывод его проверки
восстановлением — в комментарии к PR #1 и в отчёте сессии.

## Не проверено
- Восстановление на ПК заказчика (Windows, git для Windows) и на сервере — NOT RUN.
