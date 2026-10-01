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
