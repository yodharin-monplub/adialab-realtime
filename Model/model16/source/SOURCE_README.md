# Compact experiment source snapshot

This mini-repository contains Models 11–16, the shared implementation,
the exact scoring module and the minimal upstream loader dependencies.
Training data, credentials, feature caches and fitted weights are excluded.
Experiments require the upstream artifact paths documented by each model.
Run commands from this directory so imports resolve.

Refresh from the main workspace: `python3 models/snapshot_compact_models.py sync`.
Check: `python3 models/snapshot_compact_models.py check`.
