# Model59: combined original-plus-whitened LightGBM voter

| Arm | Confirmation TSAUC | Full OOF TSAUC | Gain | Wins | Lower 95% | Pass |
|---|---:|---:|---:|---:|---:|---|
| model39 | 0.638793 | 0.634268 | baseline | baseline | baseline | baseline |
| LightGBM voter weight 0.40 | 0.640123 | 0.636217 | +0.001330 | 2/3 | -0.000122 | False |

Pilot gains: [0.0004290000347644263, 0.0008418961644757195, 0.0015976955970133933, 0.0028780692532184293].
Confirmation split-half gains: [0.001850351650766191, 0.0008904174469736015].
Exploratory held-series OOF; inherited features and ensemble are development-informed.

No packaging, upload, or cloud run performed.
