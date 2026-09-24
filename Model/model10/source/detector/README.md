# Detector source organization

Every retained detector implementation is classified into exactly one lineage:

- `core/`: 28 implementations from `d1` through `d30`; six are in the current
  production core contract.
- `candidate/`: 21 implementations from `d31` onward; twelve are in the current
  production candidate contract and six (`d61`-`d66`) belong to Model 8.

The folders include rejected and superseded experiments as requested. Folder
membership records research lineage; actual model inclusion is controlled only
by `CORE_BANKS`, `CANDIDATE_BANKS`, `SPECIALIST_BANKS`, and `NOVEL_BANKS` in
`feature_pipeline.py` for existing submissions. Models 4-6 retain their frozen
43-bank registry through `MODEL4_CORE_DETECTOR_BANKS` and
`MODEL4_CANDIDATE_DETECTOR_BANKS`. Repository-wide `ALL_CORE_DETECTOR_BANKS`
and `ALL_CANDIDATE_DETECTOR_BANKS` now register all 49 sources, including the
six train-only Model 8 experiments.

## Audit every detector

Run the source, causality, streaming-parity, isolation, and numeric-redundancy
audit with:

```bash
python3 detector/audit_detectors.py --cache-scan full
```

The human report is written to `detector/AUDIT.md`; the complete structured
result is written to `models/artifacts/detector_audit.json`. The audit reports
redundant cached columns but deliberately does not remove them, so accepted
model feature contracts stay compatible.
