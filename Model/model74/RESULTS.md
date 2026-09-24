# Model74: leave-one-fold-out age-band selector

| Arm | Full OOF TSAUC | Gain | Fold wins | Lower 95% | Pass |
|---|---:|---:|---:|---:|---|
| Model67 | 0.637609 | baseline | baseline | baseline | baseline |
| Cross-fitted selector | 0.637405 | -0.000204 | 2/5 | -0.000509 | False |

Fold gains: [-0.0008492412620633161, -0.00018236261638027162, -0.00013276590750699313, 9.118253370576923e-06, 0.00013609318302287843].
Split-half gains: [-0.000257593555936908, -0.00031669358640740164].

Switches away from Model67:
- held fold 0, ages 1-5: m62 (other-fold gain +0.002131)
- held fold 0, ages 6-10: m63 (other-fold gain +0.002956)
- held fold 0, ages 11-20: m39 (other-fold gain +0.003504)
- held fold 0, ages 21-40: m39 (other-fold gain +0.004504)
- held fold 0, ages 41-80: m39 (other-fold gain +0.002150)
- held fold 1, ages 1-5: m63 (other-fold gain +0.000875)
- held fold 1, ages 6-10: m63 (other-fold gain +0.002952)
- held fold 1, ages 11-20: m39 (other-fold gain +0.001870)
- held fold 1, ages 21-40: m62 (other-fold gain +0.000568)
- held fold 1, ages 41-80: m63 (other-fold gain +0.000749)
- held fold 2, ages 1-5: m62 (other-fold gain +0.000654)
- held fold 2, ages 6-10: m39 (other-fold gain +0.004520)
- held fold 2, ages 11-20: m39 (other-fold gain +0.001852)
- held fold 2, ages 21-40: m39 (other-fold gain +0.002341)
- held fold 3, ages 1-5: m62 (other-fold gain +0.002316)
- held fold 3, ages 6-10: m63 (other-fold gain +0.002911)
- held fold 3, ages 41-80: m64 (other-fold gain +0.000811)
- held fold 4, ages 1-5: m62 (other-fold gain +0.002206)
- held fold 4, ages 6-10: m63 (other-fold gain +0.002853)
- held fold 4, ages 21-40: m39 (other-fold gain +0.000773)
- held fold 4, ages 41-80: m64 (other-fold gain +0.000586)

Every held fold's selections were learned from the other four folds only. Band design and source pool remain development-informed.

No GPU rental, packaging, upload, or cloud run performed.
