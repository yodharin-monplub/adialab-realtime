# Model84: LightGBM capacity diversity

| Arm | Confirmation TSAUC | Full OOF TSAUC | Gain | Wins | Lower 95% | Pass |
|---|---:|---:|---:|---:|---:|---|
| Model67 | 0.640196 | 0.637609 | baseline | baseline | baseline | baseline |
| expressive weight 0.500 | 0.640556 | 0.637888 | +0.000360 | 2/3 | -0.000439 | False |

Fold gains: [0.0008581645631565715, -0.0005451703844628897, -0.00016222505061558312, 0.0008092896509152014, 0.0004330730665471627]. Split-half gains: [-0.00027461963186270477, 0.0010660815390918232].

No packaging, upload, or cloud run performed.

Report repair: the frozen runner trained only the pilot-selected arm on confirmation folds but initially attempted to load all arms while reporting. report_fix.py restricts loading to the already-frozen selected arm; no fit, prediction, blend, or score was changed.
