# ADIA Lab Structural Break Challenge: Real-Time Edition

Our solution to the [ADIA Lab Structural Break Challenge: Real-Time Edition](https://hub.crunchdao.com/competitions/structural-break-real-time) on CrunchDAO (May to October 2026). We finished with **public TS-AUC 0.6367**. The private-set result is announced at the end of October 2026.

This repository is the full lab notebook. It holds **176 numbered experiments**, each with its code and a `readme.txt` saying what was tried, why, and what happened, including the many things that did not work.

---

## The task in three lines

- Each series has a long **history** (1,000–5,000 points, no break) and an **online** segment (10–1,000 points). The online points arrive **one at a time**.
- After every new point you output a score in [0, 1]: how confident you are that a structural break has **already** happened. Breaks can change the mean, variance, distribution shape or dependence. Half the series have no break.
- The metric is **TS-AUC**. At each online step t, an AUC is computed across all series alive at t, and the per-step AUCs are averaged with weight n_pos(t)·n_neg(t). Only the **ranking within a time step** matters, so calibration across steps is irrelevant.

## Result

| Submission | Public TS-AUC | What changed |
|---|---|---|
| Sub28/29 | 0.6243 | tree "chain": detector bank → stack → rolling max |
| Sub30 | 0.6296 | + distilled student (oracle-teacher soft targets) |
| Sub31 | 0.6309 | + GRU student on whitening/detector inputs |
| Sub33 | 0.6331 | student with split-test features + re-cut series, bagged |
| Sub40 | 0.6340 | student trained on all 10,000 labelled series |
| Sub41 | 0.6363 | bigger GRUs (hidden 256) with oracle soft targets |
| **Sub42** | **0.6367** | GRUs trained on all 10,000 series (**final**) |

For scale: the top of the public leaderboard was 0.663, and prizes go to the top 10 (about 0.652). We finished in the top ~15%.

## The final model

Three components run step by step on each series and are blended on z-scores:

```
score = z(chain) + 1.5 · z(student) + 1.0 · z(nets)
```

| Component | What it is | Trained on |
|---|---|---|
| **chain** | 31 detector-bank LightGBMs + core voters + ExtraTrees/RandomForest → logistic stack over 34 probabilities → rolling max over 128 steps, plus 10% of a compact 154-feature model | development series, hard labels |
| **student** (the strong part) | LightGBM on 317 causal features + 27 split-test features, trained on **soft targets** 0.5·y + 0.5·oracle-teacher; 40% of the weight on synthetic series (splices + three pooled re-cut sets) | all 10,000 labelled series |
| **nets** | 5 GRUs (hidden 256, 2 layers), 102 inputs: 61 AR-whitening features + 40 detector columns + log(1+t), same soft targets | all 10,000 labelled series |

Everything is **causal and incremental**: each step updates a running state, so the whole 10,000-series test runs in about 4.8 cloud hours. The final model is documented end to end in [`Model/PIPELINE_SUBMISSION42.txt`](../Model/PIPELINE_SUBMISSION42.txt).

---

## What worked (biggest levers first)

1. **Oracle distillation** (Models 109–111, 157). A *teacher* model is allowed to see the whole online segment, including the future, and knows where the break is. The causal *student* is trained on 0.5·label + 0.5·teacher-probability instead of hard labels. This was the first idea to beat a long plateau: about +0.007 on the raw student.
2. **Spliced series** (Model 18 onward). Glue the pre-break part of series A to the real post-break tail of series B to create new breaks from real material. Parametric break injection and bootstrap replicas both *failed* (Models 134–135). Only real post-break tails helped.
3. **Split-test features** (Model 144). Estimate the most likely split point, then run two-sample tests (location, scale, distribution) on AR-whitened data before and after it: +0.0037 dev, 5/5 folds.
4. **Re-cut series** (Models 147, 163). "Move the start": hand the first s online points to the history, so the same real series gets its break at a different step. Three pooled re-cut sets with soft targets gave the largest student gain after distillation.
5. **Training on every labelled series** (Models 154, 170). Once the recipe was fixed and validated on the sealed holdout, retraining on the holdout series as well helped both the student and the nets (+0.0069 raw for the nets).
6. **A GRU for diversity** (Models 124, 169). It is weaker alone (~0.62) than the tree student, but it adds to the blend because it errs in different places.

## What did not work (measured and closed)

Each of these has its own `Model/modelN/readme.txt` with numbers.

- **More of the same data for the student:** more than three re-cut sets (163), extra splice sets (165), "stitch" rows (167), bagging (166), a mirrored re-cut set pooled in (174, +0.0004 at 1.7 s.e.).
- **New feature families for the student:** break evidence integrated over all break positions (149). It gave the strongest single feature ever measured (0.588) and still added nothing. Also moment/Fourier banks (133) and 500-feature banks (114).
- **Anything derived for the nets:** re-cut, spliced and mirrored sequences (172–173), longer training (172), more capacity (169), more seeds (169), and the 2025 challenge's real series (176, +0.0009 at 0.9 s.e. over paired seeds).
- **A fourth blend component:** ridge and ExtraTrees on the student's exact data (175). ExtraTrees even beat the student on one fold, and still added +0.00007 to the blend.
- **Others:** a ranker objective (146), a raw-series TCN (143), learned temporal stackers (116), time-varying blend weights and post-processing (14, 142), and 2025 data as tree rows (145).

---

## How we evaluated, and the mistakes the evaluation caught

- **Splits.** There are 8,333 development series in 5 series-level folds, plus **1,667 series sealed as an outer holdout**. That holdout was used only to confirm a winner, never to tune.
- **The dev folds became optimistic.** After 100+ models were selected on them, they overstated the cloud score by about 0.006. The sealed holdout matched the cloud score to within about 0.0014, so it was the go/no-go test.
- **Decision rule.** A change is accepted only if its gain exceeds **2 standard errors AND wins at least 4 of 5 folds**. Blend weights are chosen on folds 0–1 and confirmed on folds 2–4. The rule earned its keep: Sub34 gained +0.0013 on dev (5/5 folds) and *lost* 0.0009 on the holdout.
- **Leaks we caught.** A GroupNorm over the time axis leaked the future into a TCN (143). Spliced series tagged by their *plan* index slipped held-fold donors into training, inflating one fold by +0.017 (172; fixed in `split_splices.py`). Rule of thumb: any single-fold gain above ~0.01 is a leak until proven otherwise.
- **Seed noise in the nets.** Two GRUs trained on *identical* data differ by up to **0.0064 TS-AUC on a single fold** from initialisation alone (176), which is more than most effects we tried to measure. Net changes need paired seeds: the same initialisation with and without the change, at least twice.
- **Leaderboard overlap.** The 100 local test series turned out to be part of the public cloud test. We did not exploit this.

---

## Repository layout

```
AGENTS.md              instructions the author gave the coding agent (see "How this was built")
Data/                  competition data - NOT in git, download it (see below)
Model/
  modelN/readme.txt    one experiment each: idea, commands, results, verdict
  modelN/*.py          its code (trained artefacts, runs/ and data/, are not in git)
  PIPELINE_SUBMISSION42.txt   the final model, how to rebuild/validate/upload it
  FINAL_EVALUATION_RULES.txt  how the final ranking picks a submission
  SESSION_RESULTS_*.md, NEXT_RESULTS.md   older summaries
Other/                 tooling: requirements.txt, layout script, tests, helpers
```

Good entry points:
- [`Model/PIPELINE_SUBMISSION42.txt`](../Model/PIPELINE_SUBMISSION42.txt): the final model in one page.
- [`Model/model109/readme.txt`](../Model/model109/readme.txt) and [`model111`](../Model/model111/readme.txt): the distillation idea.
- [`Model/model144`](../Model/model144/readme.txt), [`model147`](../Model/model147/readme.txt): split tests and re-cuts.
- [`Model/model170`](../Model/model170/readme.txt): all-series training for the nets.
- [`Model/model172`](../Model/model172/readme.txt), [`model176`](../Model/model176/readme.txt): the leak and the seed-noise findings.

## Reproducing

**What is and isn't included.** The code for every experiment is here. The data, the trained models (LightGBM files and GRU weights, many GB) and the sealed 129 MB submission bundle are **not**. Reproducing Sub42 means retraining. The per-model readmes give the exact commands. The nets need a GPU; we rented an RTX 3090 on vast.ai, and the helpers are `Model/model169/vast.py` and `pod.py`.

1. **Data.** Join the competition on the CrunchDAO hub and put its files in `Data/ADIA Lab Structural Break Challenge Real-Time Edition/`. The 2025 challenge data is optional; the rules allow it as external data. It goes in `Data/ADIA Lab Structural Break Challenge/`.
2. **Environment.** We used Python 3.12, then:
   ```bash
   python -m venv Other/.venv
   pip install -r Other/requirements.txt
   ```
3. **Path links.** The code and the frozen, hash-verified protocol files use the project's original paths (`models/`, `train/`, `submissions/`, `.venv/`), so recreate them as links.
   - On Windows:
     ```bash
     powershell -File Other/restructure_layout.ps1 -LinksOnly
     ```
   - On Linux or macOS:
     ```bash
     ln -s Model models && ln -s "Data/ADIA Lab Structural Break Challenge Real-Time Edition" train && ln -s Other/.venv .venv && mkdir -p submissions
     ```
4. **Run from the repository root** with the paths the readmes use, for example:
   ```bash
   python models/model154/run_all.py AP1
   ```
   Some legacy runners import Unix-only modules; `models/model111/run_model111.py` installs small stubs for them on Windows.

Secrets (API keys, the bot token, mail credentials) live in an untracked `Other/.env` and never in the code.

## How this was built

The work was driven by the author, who set goals, suggested ideas (several of the best, including the split-test features and the re-cut family, came from their prompts) and approved every upload. **Claude Code** served as the coding and experimentation agent: it wrote the code, ran the experiments and kept the readmes. `AGENTS.md` holds the standing instructions it worked under, and commit messages carry its co-author line.
