# Model selection playbook

Read at step 3 (when writing the prediction contract) and again before the Gate B report. Key on
`ml_pipeline/data_profile.json → traits`. Always start with a baseline (a dummy predictor and one
simple model), judged on validation with the contract's metric. "Avoid" means: not without a
written reason in the `Model rationale:` block.

For the business framing of common problems (churn, customer value, segmentation) read
`use-cases.md` next to this file; it says what the contract, the windows, and the business
baseline usually are. This file then picks the model family.

## Tabular, small (`small_data`: fewer than 5,000 rows)
- Baseline: majority-class / mean dummy, then logistic or linear regression with L2.
- Strong default: gradient boosting with shallow trees (depth 2–4, strong regularization, early
  stopping) or the regularized linear model itself.
- Worth trying: random forest (few knobs); GAM / EBM when the model must be read by a person.
- Avoid: neural nets (overfit and hide it — the hook denies this); large hyperparameter searches
  (use repeated k-fold on train and report the spread); anything you cannot explain to the user.

## Tabular, medium (5,000 – 500,000 rows)
- Baseline as above. Strong default: LightGBM / XGBoost / CatBoost with early stopping on validation.
- Worth trying: a linear model with feature crosses; an MLP only if boosting plateaus and rows > 50,000.
- Avoid: k-NN and RBF-SVM above ~50,000 rows (quadratic time); unregularized trees.

## Tabular, large (more than 500,000 rows)
- Strong default: histogram gradient boosting (`HistGradientBoosting*`, LightGBM) with subsampling;
  linear models on hashed or sparse features when the table is very wide.
- Avoid: anything O(n²); exact k-NN; kernel SVMs.

## Imbalanced target (`imbalanced`: minority class under 10%)
- Metric: average precision (PR-AUC) primary; recall at a fixed precision or F1 at a chosen
  threshold; never accuracy for model selection (the hook denies `scoring="accuracy"`). Always pass
  `scoring=` explicitly: estimators like `LogisticRegressionCV` default to accuracy, which the hook
  cannot see.
- First try `class_weight="balanced"` / `scale_pos_weight`; if you resample, `fit_resample` on the
  training split only (the hook denies anything else).
- The majority-class dummy baseline shows exactly what accuracy hides.
- Pick the decision threshold with `TunedThresholdClassifierCV` on train (internal CV) or on
  validation, never test. Calibrate on data the model was not fit on:
  `CalibratedClassifierCV(FrozenEstimator(model))` (`cv="prefit"` is deprecated); sigmoid below
  ~1,000 samples, isotonic above.

## High-cardinality categoricals (`high_card_categoricals`: more than 50 levels)
- Prefer CatBoost, or target / ordinal encoding fit on train inside a pipeline object. For
  `TargetEncoder`, use `fit_transform(X_train, y_train)` (it cross-fits); `fit` then `transform` on the
  same rows leaks the target.
- Avoid one-hot encoding (width explosion, and leakage if fit on all rows).

## Temporal data (`has_datetime`)
- Split: chronological — `guard.split(time_col=...)`; validation is the period after train, test the
  latest period; `TimeSeriesSplit` for CV, with `gap=` at least the longest lag/window or the forecast
  horizon. Random splits and k-fold are denied by the hook.
- Features: lags and rolling statistics computed only from the past; calendar features.
- Baseline: last value / seasonal naive for forecasting; logistic on lag features for temporal
  classification. Strong default: gradient boosting on lag features; ETS / ARIMA for univariate series.
- Avoid: tree models extrapolating a trend (detrend first); LSTMs before boosting has been tried.

## Repeated entities (`has_groups`: patients, users, devices, sites)
- Split by group — `guard.split(group_col=...)`, `GroupKFold` for CV. Row-level splits are denied.
- Avoid: features that identify the entity; the model memorizes it instead of learning.
- Both temporal and grouped: `guard.split(time_col=...)` wins and ignores `group_col`. Tell the user,
  then check how many groups appear in more than one split and report it. CV is then
  `TimeSeriesSplit(gap=...)`, which the `group-split` red flag allows on temporal data; any other
  non-group splitter stays denied.

## Text
- Baseline: TF-IDF (1–2 grams) + logistic regression or linear SVM.
- Strong default: a fine-tuned small transformer only when the baseline is clearly insufficient and
  there are more than ~5,000 labelled examples. Avoid training embeddings from scratch on small corpora.

## Images
- Baseline: features from a pretrained CNN + logistic regression. Strong default: fine-tune a
  pretrained backbone with augmentation applied after the split.
- Avoid: training from scratch below ~100,000 images; augmenting before splitting; random splits when
  images share a source (patient, device, session → group split).

Text or images with fewer than 5,000 items: fine-tuning a pretrained model trips
`neural-net-small-data`. Try the frozen-features baseline first; fine-tune only after the user agrees,
recording the red-flag override with reason "pretrained backbone".

## Skewed regression target (customer value, spend, counts)
- Baseline: mean / median dummy, then the historical average per entity, then a regularized linear model.
- Strong default: gradient boosting on a log-transformed target or with a Tweedie / Poisson objective;
  a two-part model (will it be non-zero, then how much) when most targets are zero.
- Metric: MAE (or median absolute error) plus rank quality; RMSE alone is ruled by a few large values.
- Avoid: reporting error on the log scale only (the client reads money); dropping the zeros.

## Clustering / segmentation (`task: clustering` — profiled with `task="clustering"`)

A `task: target_pending` profile (target built at step 5) has no traits about the target: re-profile
the step-5 table with `target=...` before choosing anything from this playbook.
- Baseline: a rule the business already understands — RFM scoring or the existing segments.
- Strong default: k-means or a Gaussian mixture on scaled features; k-prototypes or Gower distance
  with hierarchical clustering when categoricals matter.
- Choose the number of segments on validation with silhouette, segment sizes, and stability across
  resamples and periods — and cap it at what the client can act on.
- Metric: silhouette plus stability on the held-out period; the final check is
  `guard.final_test(pipe.predict, test, metric_fn=silhouette_score, features_fn=pipe[:-1].transform)`,
  once — the metric sees the preprocessed features, as the clustering did.
- Avoid: clustering unscaled features (the widest column decides everything); identifiers and
  near-duplicate features; DBSCAN on high-dimensional tables; judging segments by silhouette alone
  when nobody can name or use them. The step-by-step mapping is in `use-cases.md`.

## Interpretability required
- Logistic / linear with monotone constraints, GAM / EBM, shallow trees; gradient boosting with
  monotone constraints plus SHAP for explanation. Avoid black-box ensembles where a clinician or
  regulator must read the model.

## Wrong applications the hook denies (red flags) and their override names

| red flag | what fires it | do instead |
|---|---|---|
| `neural-net-small-data` | MLP / keras / torch training with fewer than 5,000 rows | regularized linear, gradient boosting |
| `random-split-temporal` | `train_test_split` / `KFold` with a datetime column | `guard.split(time_col=)`, `TimeSeriesSplit` |
| `group-split` | any non-Group split with `group_col` set | `guard.split(group_col=)`, `GroupKFold` |
| `accuracy-imbalanced` | `scoring="accuracy"` with minority under 10% | `average_precision`, `f1`, `balanced_accuracy` |
| `resample-before-split` | `fit_resample` on anything but the train split | split first, then resample train |

Override one with `- Override: red flag <name> - user approved <what> YYYY-MM-DD - reason: <why>` in
`PIPELINE.md`, only after the user explicitly agreed. Advisory smells (warned, not denied): k-NN or
kernel SVM above 50,000 rows; one-hot on high-cardinality columns; `accuracy_score` reported on an
imbalanced target.
