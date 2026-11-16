---
name: ml-pipeline
description: MANDATORY whenever a task involves training, fine-tuning, tuning, or evaluating a machine-learning model on data (tabular, time series, text, images — any modality), e.g. "predict churn", "build a classifier", "forecast sales". Not for LLM prompt evals or calling an existing model's API. Enforces a strict 16-step pipeline that starts with inspecting the raw data, gates each phase behind the user's explicit permission, and produces marimo notebooks with matplotlib visuals so the user can see and understand every step. Never jump straight to model training.
---

# ML Pipeline Discipline

**Hard rule: no model is trained until every earlier step in the pipeline is done and the
user has explicitly approved the phase gates before it.** "Train a model on this data" is a
request to *start the pipeline at step 1*, not at step 10.

## The pipeline (strict order — never reorder, never skip silently)

```
BUSINESS QUESTION
  → 0. Business framing + data intake (before the data is opened)
RAW DATA
  → 1. Data inspection
  → 2. Exploratory data analysis (EDA)
  → 3. Define the prediction problem
  ──────────── GATE A: user approval ────────────
  → 4. Data cleaning
  → 5. Data engineering
  → 6. Train / validation / test split
  → 7. Feature engineering
  → 8. Preprocessing
  ──────────── GATE B: user approval ────────────
  → 9. Baseline model
  → 10. Model training
  → 11. Hyperparameter tuning
  → 12. Model evaluation
  ──────────── GATE C: user approval ────────────
  → 13. Error analysis
  → 14. Final test (test set touched ONCE)
  → 15. Deployment (only if user asks)
  → 16. Monitoring + retraining plan
  ──────────── GATE D: wrap-up report ───────────
```

Gate D is the closing report: the final test number against baseline and validation, what the
model can and cannot be trusted for (population, time range), and the monitoring plan.

## Phase gates — explicit permission, every time

At each gate, STOP and give the user, in plain non-jargon language:

1. **What was done** — the steps completed, 1–2 sentences each.
2. **What was found** — key findings, with the visuals that show them.
3. **Decisions made and why** — e.g. "dropped 312 duplicate rows", "chose time-based split
   because the data has dates".
4. **What comes next** — the next phase's steps, in one short list. At Gate B this section
   includes the `Model rationale:` block (format under Progress tracking).
5. **The question** — ask for explicit permission to continue. Wait for a clear yes.
   Silence, ambiguity, or "hmm" is not a yes. If the user redirects, incorporate it.

When the work is for a client, the client is asked before the gate is recorded: send the one-page
gate summary from `references/use-cases.md` (same five points plus the client's written decision),
and treat "continue with changes" as not yet a yes. The gate line in PIPELINE.md is still recorded
by the user running the pipeline.

If the user says "skip ahead" or "just train it": explain in 2–3 sentences which steps are
missing and the concrete risk (usually leakage or garbage-in), then ask once for explicit
override confirmation. If they confirm, proceed and record the override in PIPELINE.md. An override
line must name what it overrides (`Gate B`, `Gate C`, `red flag <name>`, `final test`) and carry
`reason: <why>`; the hook refuses one without a reason, and does not ask for the skipped gate's
figures or rationale — skipping them is what the override records.

## Progress tracking (survives across sessions)

On first use in a project, create `ml_pipeline/PIPELINE.md` — a checklist of the 16 steps
with status (`todo / in progress / done / approved-gate / overridden`), one line of results
per finished step, and dated gate approvals. Update it after every step. On any new session,
read it first and resume from the first unfinished step — never restart, never skip ahead
of it.

Record each gate approval on its own line in exactly this shape — the enforcement hook
parses it:

```
- Gate A: approved 2026-09-16
- Gate B: approved 2026-09-17
- Override: Gate B - user approved skipping to training 2026-09-17 - reason: <why>
```

A line that says a gate is `todo`, `pending`, or `not yet approved` never counts as approval.

After recording a gate approval in a git repository, checkpoint it:
`git add ml_pipeline/ && git commit -m "ml-pipeline: Gate X approved" && git tag -f gate-X` (plus
any code files you changed; never raw data, model binaries, or secrets). Every approved phase is then
a reproducible point to return to. Outside a git repository, skip this and never `git init` unasked.

Before recording `Gate B: approved`, PIPELINE.md must contain a model rationale in exactly this shape
(the hook checks the five bullets exist; the user judges their content):

```
Model rationale:
- traits: binary, 1,428 rows (small), minority 11.4%, has_datetime, no groups
- baseline: majority-class dummy + logistic regression (class_weight=balanced)
- candidates: gradient boosting with small trees; regularized logistic
- ruled out: neural nets (small data); k-NN (mixed scales, weak on tabular); random split (temporal)
- metric: PR-AUC primary, recall at fixed precision secondary
```

## Tools: marimo notebooks + matplotlib visuals

- **The workbench is a marimo notebook**, not loose scripts. Keep notebooks in
  `ml_pipeline/`: `01_eda.py` (steps 1–3), `02_prep.py` (steps 4–8), `03_model.py`
  (steps 9–12), `04_eval.py` (steps 13–16). In Claude Code, drive them live with the
  `marimo-pair` skill so the user watches the work happen. If it is unavailable, or in other
  harnesses, write the notebook files and tell the user to open them with `marimo edit <file>`.
- **Every step that looks at data produces matplotlib figures** (seaborn on top is fine).
  Save every figure with `guard.fig(step, name, figure, explanation)`: it writes
  `ml_pipeline/figures/NN_<name>.png` and the `- Figure NN_<name>.png: …` line the hook checks.
  A PNG saved by hand has no such line and blocks the gate.
- **Explain every figure in 1–2 plain sentences**: what it shows and why it matters for
  the next decision. A figure without an explanation is not done.

## What each step must produce

0. **Business framing and data intake** — before the data is opened, write a `Business framing:`
   block in PIPELINE.md: the decision the model supports, the use case (pick a card from
   `references/use-cases.md` — churn, customer value, segmentation — and note how this case differs),
   the rule or estimate used today (it becomes the business baseline at step 9), and what result is
   good enough. List the data requested, the period each table covers, and its join key. The hook
   does not check this block; step 3 cannot be written well without it.
1. **Data inspection** — copy the plugin's guard library to `ml_pipeline/guard.py` (its path is given at
   session start; in Codex/Kimi it is `skills/ml-pipeline/lib/mlpipeline_guard.py` next to this file),
   load the raw data read-only, and run `guard.profile(df, target=..., time_col=..., group_col=...)`.
   If the raw data has no target yet because step 5 builds it (churn, customer value), profile with
   `task="target_pending"`; for segmentation use `task="clustering"`. A pending profile reports no
   class balance and no target leakage, so do not choose the metric from it. Parse every date-like column first (`parse_dates=` / `pd.to_datetime`): the guard only recognises
   real datetime dtypes, and a date left as a string slips past the temporal-split check.
   Read `ml_pipeline/data_profile.json` before anything else: its `leakage_suspects` and `traits`
   decide the split, the metric, and the model family. Report rows × columns, column kinds, missing
   values, duplicates, and every leakage suspect. For images or text, build a manifest table first
   (path, label, size or length) and profile that. No modification yet. Record in PIPELINE.md the
   raw file's SHA-256 and row count, one `SEED` reused by every split/model/CV, and the Python and
   library versions. If a later session finds a different hash, tell the user: the profile and
   every gate after it are redone.
2. **EDA** — `guard.eda_figures(df, target=..., time_col=...)` writes the required figures
   (`01_missingness`, `02_target_balance`, `02_distributions`, `02_correlations`, `02_temporal_coverage`
   when a date column exists) with explanations computed from the data; add any others with
   `guard.fig(step, name, figure, explanation)`. Gate A needs two `02_*` figures; data with no
   numeric columns (text, images) may get only one, so add e.g. a length or size distribution.
   If matplotlib is missing, install it — text descriptions are not figures and the gate will not
   accept them. Output: the figures, your interpretation of each, and a short list of hypotheses
   and problems spotted.
3. **Define the prediction problem** — write a short contract: target (exact definition,
   units), prediction unit and population, prediction time/horizon, information actually
   available at prediction time, objective, evaluation metric, constraints. **The user must
   approve this contract at Gate A** — it controls everything after. Start from the matching card
   in `references/use-cases.md` (index date, observation and prediction windows, typical leakage,
   business baseline), then consult `references/model-selection.md` with the profile's traits when
   choosing the metric and the candidate model families.
4. **Data cleaning** — missing values, duplicates, invalid/impossible values, inconsistent
   categories, unit/format issues. Report before/after counts for every rule, drawn as a `04_*`
   figure (Gate B needs it). Document every rule in PIPELINE.md. Prefer preserving data over
   deleting. Only row-by-row fixes run here (typos, units, impossible values); anything computed
   from the data (fill values, outlier caps, rare-category merges) is decided now but fit on train
   at step 8.
5. **Data engineering** — joins/integration, aggregation, time alignment to an index date,
   business rules, one-row-per-prediction-unit feature table, data-quality checks
   (row counts, uniqueness, ranges). When this step builds the target or the prediction-unit table,
   re-run `guard.profile(table, target=..., time_col=..., group_col=...)` on it before the split:
   the red flags, the imbalance check and the model choice read that profile, and `guard.split()`
   raises `ProfileOutdated` for a target the profile on disk has not seen. Report any new leakage
   suspects before going on.
6. **Split before any fitting** — with the guard copied at step 1:
   `train, val, test = guard.split(df, target=..., time_col=<col> if the data is temporal, group_col=<col> if the same entity appears in multiple rows)`.
   It drops exact duplicates, refuses a random split when a datetime column exists, keeps groups
   together, checks that no row lands in two splits, and freezes a fingerprint of the test set.
   The test set is touched exactly once, at step 14, through `guard.final_test()`. If the guard
   raises a `LeakageError` (`TemporalSplitRequired`, `OverlapLeakage`, `ProfileOutdated`,
   `TestSetTampered`, `TestSetAlreadyUsed`, `GateNotApproved`), stop and tell the user; never cast
   away a date column or delete `.guard_state.json` to get past it. When the data is both temporal
   and grouped (one customer, several index dates) the time split wins: count the entities that
   appear in more than one split and report the number at Gate B. Cross-validation then uses
   `TimeSeriesSplit` (the hook allows it on temporal data even when groups are set).
7. **Feature engineering** — design features on the training set's statistics only, then
   apply the same transformations to validation/test.
8. **Preprocessing** — scalers, encoders, imputers fit on train only, wrapped in a pipeline
   object so train and inference can never diverge.
9. **Baseline first** — a dummy predictor (majority class / mean) AND one simple model
   (logistic/linear regression or small tree), plus the business baseline from step 0 when there
   is one (the rule the client uses today). Record their metrics. Every later model is
   judged against this line; a complex model that can't beat it gets reported as such.
10. **Model training** — train candidate models on train, compare on validation. Log every
    run's config and score.
11. **Hyperparameter tuning** — on validation/cross-validation only. The test set is never
    part of tuning.
12. **Model evaluation** — the contract's metric plus supporting views: confusion matrix and
    ROC/PR curves for classification, residual plots for regression, always compared to the
    baseline. Report the CV spread (± std) or a bootstrap interval, not a single number. Figures
    required. Before Gate C, translate the metric into the business terms from step 0 (value kept
    minus cost of acting, at the volume the client can handle) and derive the decision threshold
    from it on validation; if the costs are unknown, say the threshold is provisional.
13. **Error analysis** — on the validation set, never test: worst predictions, performance by
    slice/subgroup, calibration, where the model fails and a hypothesis for why. At least one
    `13_*` figure (Gate D needs it).
14. **Final test** — `score = guard.final_test(pipe.predict, test, target=..., metric_fn=...)`.
    Pass the raw frozen `test` frame (never a transformed copy) and the whole fitted pipeline
    (preprocessing + model); for score-based metrics such as PR-AUC pass
    `lambda X: pipe.predict_proba(X)[:, 1]`.
    It refuses to run before the ledger records Gate C (or its override), verifies the frame is the
    frozen test set, refuses a second call, and logs the result to
    PIPELINE.md. Report the number honestly, even if it is worse than validation. No going back to
    tune on it — if the result forces changes, agree a new test strategy with the user and record
    `- Override: final test - user approved a second evaluation YYYY-MM-DD - reason: <why>`.
15. **Deployment** — only when the user asks. Save the full pipeline artifact
    (preprocessing + model together), verify a reloaded artifact reproduces predictions,
    document the inference input contract.
16. **Monitoring + retraining** — write down: what drift to watch (input and prediction
    distributions), what metric threshold triggers retraining, and how retraining reuses
    this same pipeline from step 1.

## Segmentation and other problems without a target

Clustering runs through the same 16 steps and the same gates. Profile with
`guard.profile(df, task="clustering")`, then leave `target` out of `guard.eda_figures`,
`guard.split` and `guard.final_test`: the split still holds out a frozen set (a later period when
the data has dates) and refuses to run without a clustering profile, and
`guard.final_test(pipe.predict, test, metric_fn=...)` calls `metric_fn(features, labels)` once to
check that the segments hold on unseen rows. The baseline is a rule the business
already uses (RFM, existing segments); the contract states what the segments are for and how many
the client can act on. The step-by-step mapping is in `references/use-cases.md`.

## Enforcement in Claude Code (hook)

Installed as a Claude Code plugin, `hooks/guard_training.py` runs before every Bash, Write,
Edit, and notebook tool call, scans the code for fitting and training calls, and **denies**:

- **model training** until PIPELINE.md has `Gate B: approved …` or an `Override: Gate B …`
  line. With no `ml_pipeline/PIPELINE.md` at all, training is denied with a pointer to step 1.
- **any fitting** — scalers, encoders, imputers included — until `Gate A: approved …`.
- **fitting on the test split** (`X_test`, `df_test`, `test_*`) — always, at every gate.
- **evaluating on the test split** — `.predict/.score` or a metric function whose arguments name
  `X_test`, `df_test`, `test_*` — until `Gate C: approved …` (or `Override: Gate C …`).
  `guard.final_test()` checks the same gate itself, however it is called.
- **recording a gate before its artifacts exist** — `Gate A` needs `data_profile.json`, one `01_*.png`
  and two `02_*.png`; `Gate B` needs a `04_*.png` and the `Model rationale:` block; `Gate C` a
  `12_*.png`; `Gate D` a `13_*.png` and the `Step 14 final test:` line. Every PNG must be larger than
  1 KB and have its `- Figure <file>:` line. The denial lists exactly what is missing.
- **red flags the profile can prove** — `neural-net-small-data`, `random-split-temporal`,
  `group-split`, `accuracy-imbalanced`, `resample-before-split` (see `references/model-selection.md`).
  Override one only after the user agreed: `- Override: red flag <name> - user approved <what> YYYY-MM-DD - reason: <why>`.
  An override unlocks only what it names: a red-flag override never opens Gate B or Gate C.

A PostToolUse hook also warns (never blocks) when a result looks too good to be honest
(accuracy/AUC/F1 ≥ 0.98) or when `train_test_split()` is called without `stratify=` or on data that
mentions dates. To prove a pipeline does not leak, run it on `examples/canary/canary.csv`: honest
test accuracy cannot exceed 0.80, and `mlpipeline_canary.verdict(score, n_test)` says whether a
number is above the ceiling.

A gate line quoted in prose — a skill doc, a shared document or ticket written through a
connector — records nothing and is not checked; only a write to PIPELINE.md itself is.

A denial is not an obstacle to route around: do the missing steps, get the user's explicit
approval, record the gate line, and retry. The hook fails open on its own errors, and
`ML_PIPELINE_ENFORCE=0` switches it off for projects that are not ML pipelines. Codex and Kimi
have no hook support, so there the same rules apply as instructions only.

## Non-negotiables

- Selection leakage is the one that matters most: the test set is never used to pick a model, a
  seed, a feature, or a threshold. Validation only.
- Every gate is recorded with its figures on disk and their explanations in PIPELINE.md. A figure
  that was described but not drawn does not exist.
- All fitting (cleaning statistics, features, preprocessing, models) uses training data only.
- Temporal data gets temporal splits; repeated entities get group splits.
- Baseline before any complex model; every result is reported relative to it.
- Failures and disappointing numbers are reported plainly — never hidden or reframed.
- Chat explanations stay beginner-friendly; the code stays production-grade.
