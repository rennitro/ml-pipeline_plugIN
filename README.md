# ml-pipeline

A Claude Code plugin that makes the agent build machine-learning models in a fixed, data-first
order: inspect the data, agree on the problem, prepare and split it, and only then train, with
your explicit approval at four gates. Hooks enforce the order; a small Python library keeps the
test set honest.

![The pipeline: 16 steps in 4 phases, each phase ending in an approval gate](assets/pipeline.svg)

Version 0.5.1 · Apache-2.0 · based on [jananthan30/ml-pipeline](https://github.com/jananthan30/ml-pipeline)

## Install

**Claude Code**

```
/plugin marketplace add rennitro/ml-pipeline_plugIN
/plugin install ml-pipeline@ml-pipeline
```

Restart Claude Code afterwards. The skill starts on its own for any task that trains, tunes or
evaluates a model; you can also call it with `/ml-pipeline`.

The hooks run as `python <script>`. If your system only has `python3` on the `PATH` (common on
macOS and Linux), the hooks cannot start and **enforcement is silently off** — see
[Troubleshooting](#troubleshooting).

**Codex CLI**

```
codex plugin marketplace add rennitro/ml-pipeline_plugIN
codex plugin add ml-pipeline@ml-pipeline
```

**Kimi Code CLI, or Codex without plugins**

```bash
git clone https://github.com/rennitro/ml-pipeline_plugIN && cd ml-pipeline_plugIN && ./install-other-tools.sh
```

This copies the skill, the guard library and the references into `~/.codex/skills/ml-pipeline/`
and `~/.kimi-code/skills/ml-pipeline/`, and appends an enforcement section to each tool's global
`AGENTS.md`. Codex and Kimi have no hooks, so there the rules are instructions only.

## Try it in five minutes

The repository ships a canary dataset: 2,000 rows where no honest model can exceed 0.80
accuracy. It is a safe first project.

```bash
git clone https://github.com/rennitro/ml-pipeline_plugIN
mkdir canary-demo
cp ml-pipeline_plugIN/examples/canary/canary.csv canary-demo/
cd canary-demo
claude
```

Then ask:

> Build a classifier that predicts `y` in canary.csv.

What happens:

1. The agent creates `ml_pipeline/`, copies the guard library there as `guard.py`, profiles the
   data and draws the first figures. It does not train anything.
2. It stops at **Gate A** with a plain-language summary: what it did, what it found (with the
   figures), the decisions it made, what comes next, and a question.
3. Answer "yes, Gate A approved" to continue. The agent records the approval in
   `ml_pipeline/PIPELINE.md` and moves on to cleaning and splitting.

Try "just train it" instead: the agent explains which steps are missing and what that risks,
and asks you to confirm an override. If it tries to call `.fit()` anyway, the hook denies the
call.

After a few gates the project looks like this:

```
canary-demo/
  canary.csv
  ml_pipeline/
    PIPELINE.md          progress, results per step, dated gate approvals
    guard.py             the guard library (profile, figures, split, final test)
    data_profile.json    column kinds, missingness, traits, leakage suspects
    figures/             01_missingness.png, 02_target_balance.png, ...
    01_eda.py            marimo notebooks, one per phase
    02_prep.py
```

## How it works

The pipeline has a business step 0 and 16 steps in four phases. Each phase ends in a gate that
only you can open.

| Phase | Steps | Gate | The gate opens |
|---|---|---|---|
| — | 0 Business framing and data intake | — | — |
| Understand | 1 Data inspection · 2 EDA · 3 Define the prediction problem | **A** | fitting of any kind (cleaning statistics, scalers, encoders) |
| Prepare | 4 Cleaning · 5 Data engineering · 6 Split · 7 Features · 8 Preprocessing | **B** | model training |
| Model | 9 Baseline · 10 Training · 11 Tuning · 12 Evaluation | **C** | the single evaluation on the test set |
| Prove & ship | 13 Error analysis · 14 Final test · 15 Deployment (on request) · 16 Monitoring | **D** | nothing: it records the wrap-up report |

Three parts work together:

- **The skill** (`skills/ml-pipeline/SKILL.md`) tells the agent what each step must produce and
  how to report at a gate. Two references back it: `model-selection.md` picks a model family
  from the data's traits; `use-cases.md` frames common business problems (churn, customer value,
  segmentation) and holds a one-page gate summary for clients.
- **The hooks** (`hooks/`) check every Bash, Write, Edit, notebook and MCP call before it runs
  and deny what the recorded gates do not allow yet.
- **The guard library** (`lib/mlpipeline_guard.py`) is copied into each project. It profiles the
  data, draws the required figures, splits without leakage, and lets the frozen test set be
  scored exactly once.

Why this design: a coding agent tends to reach `.fit()` before anyone has looked at the data or
agreed what is predicted, and instructions alone do not stop it. The hooks make the order a hard
constraint, while the gates keep the decisions with a person.

## Common tasks

### Approve a gate

Answer the agent's question with a clear yes. It writes one line per gate to
`ml_pipeline/PIPELINE.md`:

```
- Gate A: approved 2026-09-16
```

The hook refuses the line until that gate's artifacts exist (see
[What each gate needs](#what-each-gate-needs)). In a git repository the agent also commits and
tags the approved state (`gate-A`, `gate-B`, ...). Silence or "hmm" is never a yes.

### Skip ahead

Say so. The agent explains the risk, asks once for confirmation, and records an override with
your reason:

```
- Override: Gate B - user approved skipping to training 2026-09-17 - reason: deadline demo
```

An override needs `reason:`, does not need the skipped gate's figures, and unlocks only what it
names. A red-flag override never opens a gate.

### Work on a problem whose target is built later (churn, customer value)

The raw data has no target column yet, so step 1 profiles it as pending:

```python
guard.profile(df, task="target_pending")
```

Step 5 builds the one-row-per-customer table with its target; profile that table before the
split. `guard.split()` raises `ProfileOutdated` until you do.

```python
guard.profile(table, target="churned", time_col="index_date", group_col="customer_id")
train, val, test = guard.split(table, target="churned", time_col="index_date")
```

### Segment customers (no target)

```python
guard.profile(df, task="clustering")
train, val, test = guard.split(df, time_col="order_date")
```

The gates stay the same. At step 14, `guard.final_test(pipe.predict, test, metric_fn=...)` calls
`metric_fn(features, labels)` once on the held-out rows. The features it receives are the raw
frame, so compute the metric on the preprocessed features yourself, for example
`metric_fn=lambda X, labels: silhouette_score(pipe[:-1].transform(X), labels)`.

### Check a pipeline for leakage with the canary

Run the pipeline on `examples/canary/canary.csv`, then test the reported accuracy:

```bash
python -c "import sys; sys.path.insert(0, 'lib'); import mlpipeline_canary as c; print(c.verdict(0.91, 400))"
```

```
(True, 'accuracy 0.910 exceeds the honest ceiling 0.80 + 3 SE (0.860) on n=400: the test set leaked')
```

To write a fresh copy into a project: `mlpipeline_canary.write_csv("ml_pipeline/canary.csv")`.

### Turn enforcement off

For a project that is not an ML pipeline, set `ML_PIPELINE_ENFORCE=0` in the environment
Claude Code starts in. The skill still applies; only the hooks stop.

## Reference

### What each gate needs

The hook refuses to record a gate until these exist in `ml_pipeline/`:

| Gate | Required |
|---|---|
| A | `data_profile.json`, one `figures/01_*.png`, two `figures/02_*.png` |
| B | one `figures/04_*.png`, and a `Model rationale:` block with the lines `traits`, `baseline`, `candidates`, `ruled out`, `metric` |
| C | one `figures/12_*.png` |
| D | one `figures/13_*.png`, and the `- Step 14 final test:` line |

Every PNG must be larger than 1 KB and have a `- Figure <file>: <explanation>` line in
`PIPELINE.md`; `guard.fig()` writes both.

### What the hook denies

| Denied | Until |
|---|---|
| Any fitting, preprocessing included | Gate A |
| Model training | Gate B, or `Override: Gate B` |
| Evaluating on the test split (`X_test`, `test_*`, `*_test`) | Gate C, or `Override: Gate C` |
| Fitting on the test split | never allowed |
| Training with no `ml_pipeline/PIPELINE.md` at all | step 1 has created it |

Red flags, denied when the data profile proves them (override with
`- Override: red flag <name> - user approved <what> YYYY-MM-DD - reason: <why>`):

| Red flag | Fires on | Do instead |
|---|---|---|
| `neural-net-small-data` | a neural net trained on fewer than 5,000 rows | regularized linear model, gradient boosting |
| `random-split-temporal` | `train_test_split` / `KFold` with a datetime column | `guard.split(time_col=...)`, `TimeSeriesSplit` |
| `group-split` | a non-group splitter when `group_col` is set (`TimeSeriesSplit` is allowed if the data is also temporal) | `guard.split(group_col=...)`, `GroupKFold` |
| `accuracy-imbalanced` | `scoring="accuracy"` with a minority class under 10% | `average_precision`, `f1`, `balanced_accuracy` |
| `resample-before-split` | `fit_resample` on anything but the training split | split first, then resample the training split |

A second hook only warns, never blocks: a metric of 0.98 or more, and `train_test_split()`
without `stratify=` or on data that mentions dates.

### Guard library

| Function | Step | What it does |
|---|---|---|
| `profile(df, *, target=None, task=None, time_col=None, group_col=None)` | 1, again at 5 | Writes `data_profile.json`. Without a target, `task` must be `"clustering"` or `"target_pending"`. |
| `eda_figures(df, *, target=None, time_col=None)` | 2 | Draws the required `01_*` and `02_*` figures with explanations computed from the data. |
| `fig(step, name, figure, explanation)` | any | Saves `figures/NN_<name>.png` and its `- Figure` line. |
| `split(df, *, target=None, time_col=None, group_col=None, test_size=0.2, val_size=0.2, seed=0)` | 6 | Drops duplicates, splits by time or group when asked, checks overlap, freezes the test set. |
| `final_test(predict_fn, test, *, target=None, metric_fn)` | 14 | Scores the frozen test set once and logs `- Step 14 final test:`. |

Every function also takes `state_dir="ml_pipeline"`; `final_test` takes
`pipeline_md="ml_pipeline/PIPELINE.md"`.

Profile traits that drive the red flags and the model playbook: `small_data` (under 5,000
rows), `imbalanced` (minority class under 10%), `has_datetime`, `has_groups`,
`high_card_categoricals` (over 50 levels). A column is a leakage suspect when its correlation
with the target exceeds 0.95, it predicts the target with over 98% purity, its date is later
than the time column, or its name suggests post-outcome data.

### Guard errors

All derive from `LeakageError`. Stop and tell the user; never delete `.guard_state.json` or
drop a date column to get past one.

| Error | Raised when | Fix |
|---|---|---|
| `TemporalSplitRequired` | the frame has a datetime column and no `time_col` | pass `time_col=...` |
| `OverlapLeakage` | an identical row lands in two splits | deduplicate before splitting |
| `ProfileOutdated` | the target differs from the one in the saved profile, or a split without a target has no clustering profile | re-run `profile()` on this table |
| `GateNotApproved` | `final_test()` runs before Gate C is recorded | finish steps 9–12 and get approval |
| `TestSetTampered` | the frame given to `final_test()` is not the frozen test set | pass the `test` frame from `split()` unchanged |
| `TestSetAlreadyUsed` | a second `final_test()` | agree a new test strategy and record `- Override: final test - ... - reason: ...` |

### Environment variables

| Variable | Effect |
|---|---|
| `ML_PIPELINE_ENFORCE=0` | switches the hooks off (`false`, `off`, `no` also work) |
| `ML_PIPELINE_FILE=<path>` | uses this file instead of searching for `ml_pipeline/PIPELINE.md` in the working directory and up to four parents |

## Troubleshooting

**Nothing is ever denied.** The hooks probably cannot start. They run as `python`; on a system
with only `python3`, change the three commands in `hooks/hooks.json` to `python3` and restart
Claude Code. Also check that `ML_PIPELINE_ENFORCE` is not set to `0`.

**A gate line is refused.** The denial lists exactly what is missing: a figure, its `- Figure`
line, or a `Model rationale:` line. Create it, then record the gate again.

**Code in an unrelated project is denied.** The plugin is installed for your user, so the hook
runs in every project and denies training code when there is no `ml_pipeline/PIPELINE.md`. Set
`ML_PIPELINE_ENFORCE=0` for that project.

**The hook blocks a file that only quotes example code.** It reads code samples in `.py` files
and commands literally. Documentation files (`.md`, `.txt`, `.json`, ...) are not checked.

## Development

```bash
python -m pip install -r tests/requirements-dev.txt
python -m unittest discover -s tests
```

The hook tests use the standard library only; the guard tests need pandas, numpy and matplotlib.
`evals/` holds behavioural evaluation cases (gate stop, understand first) for an agent run.

```
.claude-plugin/       plugin and marketplace manifests (Claude Code)
.codex-plugin/        plugin manifest (Codex)
skills/ml-pipeline/   SKILL.md and references/
hooks/                PreToolUse guard, PostToolUse warnings, SessionStart paths
lib/                  guard library and canary generator
examples/canary/      canary dataset with a known 0.80 ceiling
evals/  tests/
```

After changing the plugin, raise `version` in both `plugin.json` files, commit, then run
`claude plugin update ml-pipeline@ml-pipeline` and restart Claude Code. Without a new version
number the update does not pick up the change.

## License

Apache-2.0. Based on [jananthan30/ml-pipeline](https://github.com/jananthan30/ml-pipeline) by
Jananthan Paramsothy.
