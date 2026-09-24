# Model 3 source snapshot

Manifested deployment files are copied byte-for-byte from the historical `submissions/submission_3/` bundle. Current retained experiment/bundle drivers live under `research/`; they are useful provenance but are not claimed as historically exact because this workspace has no Git history. Run deployment commands from this directory so its flat Crunch imports resolve.

The snapshot contains every recoverable deployment source file, the frozen non-model feature contracts when applicable, retained research drivers, documentation, and the explicitly identified reconstructed detector helper. It deliberately excludes `train/`, fitted models, feature matrices,
prediction checkpoints, caches, and run logs.

`SOURCE_MANIFEST.json` records the SHA-256 digest and byte size of every included
file. Rebuild all snapshots from the repository root with:

```bash
python3 models/sync_model_sources.py sync
python3 models/sync_model_sources.py check
```

Do not edit this generated directory directly. Change the canonical source and
run the sync command again.
