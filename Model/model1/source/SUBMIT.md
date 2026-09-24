# Submit this directory

This is the default Crunch script layout:

- `main.py`: entry point exposing `train()` and streaming `infer()`;
- `resources/`: the validated pretrained core/candidate logit ensemble;
- `requirements.txt`: runtime dependencies;
- detector and feature modules imported by the entry point.

For the web uploader, select this entire directory as the submission root.

For the CLI, first use the one-time setup command shown on the competition's
**Submit via CLI** page. Copy this directory's contents into that initialized
project, then run:

```bash
crunch test
crunch push --message "pretrained equal-logit TSAUC ensemble"
```

`train()` validates and preserves the uploaded ensemble when Crunch forces a
first training call. No train data, protected test archive, feature cache, OOF
prediction, or rejected experimental model is included here.
