# Core detector lineage

This folder contains all 28 implemented core-lineage versions from `d1`
through `d30` (missing numbers were never retained as source files).

The six production core banks are:

`d4`, `d15`, `d22`, `d26`, `d27`, and `d30`.

The other files are foundations, superseded versions, or rejected ablations:

`d1`, `d2`, `d3`, `d5`, `d6`, `d7`, `d8`, `d9`, `d10`, `d11`, `d12`, `d13`,
`d16`, `d17`, `d19`, `d20`, `d21`, `d23`, `d24`, `d25`, `d28`, and `d29`.

Keeping the full lineage here preserves the experimental record while the
explicit `CORE_BANKS` contract in `feature_pipeline.py` prevents rejected
features from entering an existing submission accidentally. The separate
`ALL_CORE_DETECTOR_BANKS` Model 4 contract includes all 28 for explicit
screening; inherited d3 prefixes are stored once rather than duplicated.

`channels18.py` is a shared helper rather than a numbered detector. It restores
the normalized d3 innovation-channel contract required by the archived d19 and
d20 experiments.
