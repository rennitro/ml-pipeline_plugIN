# Business use-case cards

Read at step 0 (business framing) and again at step 3 (prediction contract). A card is a
pre-filled contract for a common business problem: the 16 steps stay the same, the card says what
they usually contain. Adapt every card to the client; never copy one unchanged. The hook does not
enforce anything in this file — it is guidance for the agent and the analyst.

`model-selection.md` picks the model family from the data's traits; this file picks the problem
framing from the business question. Use both.

## Step 0 — business framing and data intake

Do this before the data is opened, and record the answers under `Business framing:` in
`ml_pipeline/PIPELINE.md`. They are the raw material for the step 3 contract.

- **Decision** — what the client will do differently because of the model (e.g. who gets a
  retention offer). No decision, no model.
- **Use case** — which card below fits, and how the client's situation differs from it.
- **Current practice** — the rule or estimate the client uses today. It becomes the *business
  baseline* at step 9, next to the dummy and the simple model.
- **Success measure** — what result is good enough, in money or counts where possible.

Data to request, with the period covered and the join key for each table:

| Data | Typical content | Needed for |
|---|---|---|
| Customer master | id, sign-up date, demographics or firmographics | prediction unit and population |
| Transactions | date, amount, product, channel | behavioural features, customer value |
| Contracts / subscriptions | start, end, plan, cancellation | the definition of churn |
| Interactions | support tickets, campaigns, web activity | extra features — most leakage suspects live here |
| Data dictionary | meaning, unit, and *when each column is filled* | deciding what is known at prediction time |

Keep the delivered files unchanged; step 1 records their hash.

## The cards

| | Churn | Customer lifetime value (CLV) | Customer segmentation |
|---|---|---|---|
| Business question | Who will leave, and who is worth keeping? | What is a customer worth over the next period? | Which groups of customers should be treated differently? |
| Task | binary classification | regression, or a probabilistic model (e.g. BG/NBD + Gamma-Gamma) | clustering — no target |
| Agree with the client | what counts as churn (cancellation, or N days inactive) and how far ahead to predict | horizon (6 / 12 / 24 months); revenue or margin | what the segments are for (campaign, pricing, service) and how many they can act on |
| Typical leakage | fields filled at or after cancellation: closing date, cancellation ticket | purchases from the target period inside the features | none in the classic sense; the risk is segments that do not hold over time |
| Split | chronological, with an observation window and a prediction window | chronological | hold out a later period to check stability |
| Business baseline | "no purchase in N days" rule | historical average spend | RFM scoring, or the client's existing segments |
| Metric | PR-AUC; lift in the top decile | MAE; rank accuracy (do the top-value customers come out on top) | silhouette, stability across periods, and whether the business can name and use each segment |
| Deliverable | ranked customer list with a risk score | value estimate per customer, and value bands | segment descriptions and each customer's segment |

Further cards in the same shape: purchase propensity and cross-sell (binary classification, same
windows as churn), demand forecasting (see *Temporal data* in `model-selection.md`).

CLV targets are heavily right-skewed with many zeros. Report MAE and rank quality rather than
RMSE alone, consider a log or two-part (will they buy / how much) model, and always show the
historical-average baseline.

## Time windows (churn, CLV, propensity)

Everything hangs on an **index date**: features come only from the *observation window* before it,
the target only from the *prediction window* after it. Set the windows at step 5 (time alignment);
the chronological split at step 6 builds on them, and `TimeSeriesSplit(gap=...)` must be at least
the prediction window. Most "too good" results on these problems are windows that overlap.

The raw data has no target yet, so step 1 profiles it with `guard.profile(df, task="target_pending")`.
Once step 5 has built the one-row-per-customer table with its target, profile that table with
`target=...`: only then do the class balance, the leakage suspects and the red flags describe the
real problem, and `guard.split()` refuses the target until it has been profiled.

### One customer, several index dates

The data is then both temporal and grouped. `guard.split(time_col=...)` wins and ignores
`group_col`, so the same customer can land in train and test. At step 6 count how many customers
appear in more than one split and report it at Gate B. If the overlap is large, keep one index
date per customer.

## Business value at Gate C

Before asking for Gate C, translate the validation metric into the client's terms and derive the
decision threshold from it — on validation, never test. For churn, for example:

    value = (customers contacted who would have left) × (offer success rate) × (value of a kept customer)
            − (customers contacted) × (cost of one contact)

Ask the client for the cost of a contact, the offer's expected success rate, and how many customers
they can actually contact; without those the threshold is a guess, and the Gate C report says so.
Compare against the business baseline at the same number of contacts.

## Segmentation: how the steps change

There is no target: profile with `guard.profile(df, task="clustering")`, then leave `target` out of
`guard.eda_figures`, `guard.split` and `guard.final_test` (the split refuses to run without a
clustering profile). The gates, their figures, and the hook stay the same; a clustering fit is
locked until Gate B like any other model, and the final check until Gate C.

| Step | Supervised | Segmentation |
|---|---|---|
| 1 Profile | target, class balance, leakage suspects | `task: clustering`; no target fields. Decide which columns describe behaviour and which only identify the customer |
| 3 Contract | target, horizon, metric | purpose of the segments, the unit segmented, the features allowed, the number of segments the client can act on, and the stability check |
| 6 Split | train / validation / test | `guard.split(df, time_col=...)` — the held-out set is a later period (or unseen customers) used only to check that the segments hold |
| 8 Preprocessing | scalers, encoders fit on train | the same, and it matters more: distance-based clustering is ruled by feature scale |
| 9 Baseline | dummy + simple model | RFM scoring or the client's existing segments |
| 10–11 Training, tuning | compare candidates on validation | compare algorithms and the number of segments on validation: silhouette, segment sizes, stability across resamples |
| 12 Evaluation | contract metric vs baseline | silhouette plus a segment-profile figure (`12_*`), compared with the baseline segments |
| 13 Error analysis | worst predictions, slices | segments that are tiny, unstable, or that the business cannot tell apart (`13_*`) |
| 14 Final test | one score on the frozen test set | `guard.final_test(pipe.predict, test, metric_fn=silhouette_score, features_fn=pipe[:-1].transform)` once — `metric_fn` receives the preprocessed features and the labels. Report how far it falls below validation |

The `Model rationale:` block keeps its five bullets: `traits` (clustering, rows, scale of the
features), `baseline` (RFM), `candidates`, `ruled out`, `metric` (silhouette + stability).
A good silhouette on segments nobody can act on is a failed project; say so at Gate C.

## Client gate summary

When the work is for a client, send this one-page summary before recording each gate, in the
client's language and without jargon. The analyst still records the gate in `PIPELINE.md`; keep
the client's written answer with the project.

```markdown
# [Project] – Gate [A / B / C / D] summary

**Date:** [yyyy-mm-dd] · **Prepared by:** [analyst] · **Answer needed by:** [yyyy-mm-dd]

## 1. What was done
[The completed steps, 1–2 sentences each.]

## 2. What we found
[The 3–5 key findings, each with a figure and 1–2 sentences on what it shows and why it matters.]

## 3. Decisions we made and why
| Decision | Reason | Effect |
| --- | --- | --- |
| [e.g. removed 312 duplicate rows] | [why] | [rows or customers affected] |

## 4. Our questions for you
1. [A specific question that can be answered yes/no or with a number.]

## 5. What comes next
[The next phase's steps in a short list, with the date of the next check-in.]

## 6. Your decision
- [ ] Continue
- [ ] Continue with changes: [what to change]
- [ ] Do not continue yet: [what is missing]
```

Questions that usually belong in section 4:

| Gate | Typical questions |
|---|---|
| A | Is the target definition and the horizon right? Are the flagged columns really known at prediction time? Is this the right population? |
| B | Are the cleaning rules acceptable? Are the business rules and the joins right? Should any customer group be excluded? |
| C | Is the performance enough compared with current practice? How many customers can you actually contact or handle? What does one contact cost? |
| D | Do you accept the result with the stated limits? Do you want it deployed? Who owns monitoring on your side? |

"Continue with changes" is not a yes: incorporate the changes, then ask again.
