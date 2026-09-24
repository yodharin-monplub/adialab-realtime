# Submit this directory

This bundle uses Crunch's default script layout and contains **no pretrained
model weights**:

- `main.py`: exposes train-from-scratch `train()` and streaming `infer()`;
- `ensemble_training.py`: fits both selected LightGBM feature families;
- `resources/`: intentionally empty until Crunch calls `train()`;
- `requirements.txt`: runtime and training dependencies;
- complete feature, detector, metric, and training support code.

`train()` consumes only the labeled tuples supplied by the competition. It
does not read a local test directory, protected archive, cached feature file,
OOF prediction, or pretrained weight. It writes this runtime structure:

```text
resources/
├── manifest.json
├── core/
│   ├── final.txt
│   └── manifest.json
└── candidate/
    ├── final.txt
    └── manifest.json
```

For the web uploader, select this entire directory as the submission root.
For the CLI, copy its contents into the project created by the competition's
setup command, then run `crunch test` and `crunch push`.
