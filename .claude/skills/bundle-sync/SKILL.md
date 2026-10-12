---
name: bundle-sync
description: "How to move git history between the internet-connected Windows PC and the isolated inference VM with git bundles. Use when syncing code in either direction."
---

# Bundle sync

The VM has no internet. History moves as files.

PC -> VM:
```
git bundle create 408032gb-<date>.bundle <base>..main      # incremental
git bundle create 408032gb-full.bundle --all               # first time
# copy the file to the VM, then on the VM:
git bundle verify 408032gb-<date>.bundle
git fetch 408032gb-<date>.bundle main:incoming && git merge --ff-only incoming
```

VM -> PC: same commands in the other direction; the PC pushes to GitHub.

Rules: always `git bundle verify` before fetching; fast-forward only; never push from the VM;
never push to a third-party remote. Record the bundle file name and `git log --oneline -n 3`
in the sync note.

Never bundle from a shallow clone (`git rev-parse --is-shallow-repository` must print `false`; if not,
`git fetch --unshallow origin` first). In a shallow clone `git bundle verify` treats the shallow boundary as a
root and reports "The bundle records a complete history", yet a fresh clone of that bundle fails with
"did not send all necessary objects" (this happened on 2026-10-10: docs/evidence/2026-10-10-backup-bundle.md).
A backup bundle is called verified only after a restore into an empty directory:
```
python3 scripts/offline/repo_backup.py create --out 408032gb-<branch>-<sha>.bundle   # refuses a shallow clone
python3 scripts/offline/repo_backup.py check 408032gb-<branch>-<sha>.bundle          # clone, fsck, refs, checkout
```

Cloud agent sessions (not the VM) publish their `claude/*` branches to origin directly (AGENTS.md "Git and sync");
a bundle is still made as an independent backup copy of the work.
