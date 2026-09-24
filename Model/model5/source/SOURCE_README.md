# Model 5 source snapshot

This snapshot preserves the repository layout. Run commands from this directory; shared modules are at its root and experiment packages are below `models/`.

The snapshot contains model and upstream package source, shared pipelines, notebooks, documentation, detector modules, and regression-test source. It deliberately excludes `train/`, fitted models, feature matrices,
prediction checkpoints, caches, and run logs.

`SOURCE_MANIFEST.json` records the SHA-256 digest and byte size of every included
file. Rebuild all snapshots from the repository root with:

```bash
python3 models/sync_model_sources.py sync
python3 models/sync_model_sources.py check
```

Do not edit this generated directory directly. Change the canonical source and
run the sync command again.
