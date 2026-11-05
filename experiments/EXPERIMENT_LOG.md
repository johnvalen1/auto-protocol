# Experiment log: LLM augmentation for CT neuro protocol classification

Running record of every analysis done for the paper. Each experiment's plan, primary
endpoint and decision rule are written here **before** it is run; results are added
after. Nothing is removed — failed or negative experiments stay in.

- Data: `CT_Neuro_Gold_Standard_V7_with_augments_lower_classes.xlsx`, sheet `data`
  (rows with `exclude_record = TRUE` dropped; 8,796 real + 1,227 synthetic rows).
- Split: fixed `split` column (7,795 real train / 1,001 real test). Synthetic rows are
  train only. The test set is never used for tuning or selection.
- Code: branch `experiment/real-only-threshold-tuning`, folder `experiments/`.
- Compute: Anthropic cloud workspace (Python 3, scikit-learn 1.9.1). Dataset contains
  random record identifiers only, no direct identifiers (confirmed by J. Valen, 2026-11-01).

## Model (unchanged across all experiments)

As implemented in `autoprotocol.py` / `autoprotocol_with_augments.py`:
- Features: word TF-IDF (1–2-grams, 20k features, English stop words) + character
  TF-IDF (`char_wb` 3–5-grams, 30k features).
- Classifier: ensemble of 3 `ClassifierChain`s of logistic regressions (random label
  order, seeds 0–2), probabilities averaged.
- Output: per-protocol thresholds chosen to maximise per-protocol F1 over a grid
  0.05–0.95 (step 0.05), averaged across 5 × 4-fold multilabel-stratified CV folds;
  each case gets 1–3 advised protocols.
- Top-k accuracy in the code = any true protocol among the 3 highest-probability
  protocols (threshold-independent).

> **Paper note:** the ARRS deck (slides 8–10) describes "bag-of-words, one-vs-rest
> logistic regression". The code is the TF-IDF classifier-chain ensemble above. The
> paper's methods must describe the code. Slides 3 and 10 define top-k as "all true
> protocols among advised", which is not what the code computes; the abstract's
> wording ("true protocol present among advised") is closer.

---

## Exp 0 — Audit of submitted results (2026-10-31)

**Question:** do the numbers in the ARRS deck/abstract (SC1449) reproduce from the data
and the saved predictions (`p_value_computations_1.xlsx`)?

**Findings**
- Cohort numbers reproduce exactly (8,796 real; 7,795/1,001; 1,227 synthetic; median
  age 61, IQR 38–77; 50.6% female; 8.8% paediatric).
- Per-protocol sensitivities (deck slides 18–19) reproduce exactly.
- Top-k 89% → 92% does **not** reproduce: the repo metric gives 87.6% → 91.7%; the
  slide's definition ("all true protocols among advised") gives 65.4% → 67.2%.
- "+22% mean class-specific accuracy" reproduces as +18 points (25.4% → 43.6%).
- Per-protocol tests on slides 18–19 used Fisher's exact test, which assumes
  independent samples; both models are scored on the same cases, so McNemar (paired)
  is correct. Under McNemar: L Spine C+ p = 0.06 (not significant), Temporal Bones C+
  p = 0.016, Head C+ loss p = 0.031.
- `p_value_computations_2.xlsx` lists Neck NC augmented sensitivity as 0.816; correct
  value is 0.118 (2/17).
- Author name differs: abstract "Jason Chong", deck "Jaron Chong".

**Main observation:** high-volume protocols (Head NC, TIA, Neck C+) were stable;
the losses were in mid-frequency protocols (Head C+, C Spine NC, Head & Neck Angio,
L Spine NC, Head Venogram, T Spine NC), each with higher precision but lower
sensitivity — the model advised them less often.

## Exp 1 — Real-only threshold tuning (2026-11-01)

**Rationale (from Exp 0):** in `autoprotocol_with_augments.py` CV folds are drawn over
real + synthetic rows, so each minority protocol's threshold is tuned largely on
synthetic validation rows, which are easier than real referrals.

**Intervention:** CV folds drawn over real rows only; all synthetic rows added to every
training fold; thresholds tuned on real validation rows only. Model unchanged.

**Pre-specified primary comparison:** augmented + real-only tuning (repo's fold-averaged
threshold rule) vs baseline. Pooled out-of-fold threshold tuning run as an ablation.

**Results (test, n = 1,001)**
- Repo augmented model reproduces `p_value_computations_1.xlsx` exactly.
- Mid-frequency protocols recovered: Head & Neck Angio 38%→38%, L Spine NC 74%→74%,
  Head Venogram 89%→89%, T Spine NC 56%→56%, C Spine NC 61%→58% (repo augmented: 34%,
  71%, 83%, 50%, 55%). Orbits NC 47%→41% (repo augmented 24%).
- Mean sensitivity, 10 non-augmented protocols: 60.8% → 61.2% (repo augmented 59.3%).
- Mean sensitivity, 11 augmented protocols: 25.4% → 43.6% (unchanged vs repo augmented).
- Sinus NC 0%→71% (Holm-adjusted p = 0.011); Temporal Bones C+ 12%→41%.
- **Head C+ not recovered:** 54% → 46% (6 lost, 0 gained, McNemar p = 0.031,
  Holm p = 0.66). Confirmed in CV on 511 real training cases: 48.6% → 42.5%.
  Its threshold barely changed (0.17 → 0.18), so the cause is not thresholding.
- Pooled-tuning ablation gave the same pattern.

## Exp 2 — Counterbalancing infection cue terms (2026-11-01)

**Rationale (from Exp 1):** the residual Head C+ loss is in the probabilities, not
the threshold. Using **training data only** (Exp 1 out-of-fold predictions), 22 real
Head C+ training cases were lost after augmentation; the most over-represented terms
were mastoid/mastoiditis (36% of lost vs 1% of kept), infection and drainage.
Synthetic vs real labelling of these referrals differs: 78% of 112 synthetic "mastoid"
rows are Temporal Bones C+, while 75% of 79 real "mastoid" referrals were protocoled
Head NC / Head C+ / Neck C+. Among training rows matching `mastoid|infect|drainage`,
the share carrying a non-augmented protocol fell from 89.3% to 54.6% after augmentation.
(Six lost Head C+ *test* cases had been inspected earlier; they were not used to
choose the terms.)

**Intervention:** within each training fold only, real rows matching the cue terms and
carrying ≥1 non-augmented protocol repeated k times; synthetic rows kept.
k chosen by formula to restore the 89.3% share: k = 6.94 → 7 (primary, pre-declared);
k = 3 and 14 as CV-only robustness checks.

**Results**
- Test: Head C+ 46% → 50% (baseline 54%; 3 lost vs 6, McNemar p = 0.25).
  Head NC 82%, Neck C+ 86%, L Spine NC 77% (all ≥ baseline).
- CV (real training rows): Head C+ 42.5% → 44.4% (baseline 48.6%); mean
  augmented-protocol sensitivity 41.8% → 38.1%; non-augmented mean unchanged
  (56.9% → 56.8%). Losses in Temporal Bones NC/C+, Orbits NC, Facial Bones C+.
- k = 3 and 14 gave the same pattern; Head C+ CV never exceeded ~44%.

**Conclusion:** counterbalancing trades minority gains for a partial Head C+ recovery.
It confirms the mechanism (dose-response between real-convention weighting and
Head C+ vs Temporal Bones C+), but Exp 1 remains the preferred configuration.
The residual cost reflects a mismatch between GPT-4o's textbook protocol choices and
local protocoling practice; the fix belongs in data generation (local exemplars in
the prompt, targeted radiologist label review of overlapping indications).

**Data issue found:** "Head Angiogram or COW" (20 real train, 2 test) and
"Head angiogram or COW" are treated as separate protocols (22 classes, not 21).

---

## Exp 3 — Label fix, non-LLM controls, aggregate endpoint (pre-registered 2026-11-01 20:50 CT)

**Rationale:** (a) the duplicate label invalidates the "21 protocols" claim, so all
main conditions are rerun after merging it; (b) the paper must show that the gain
comes from LLM-generated text, not simply from more minority examples or from
reweighting; (c) with 17–18 test cases per minority protocol, per-protocol tests are
underpowered, so a single pre-specified aggregate endpoint is defined.

**Changes:** `LABEL_FIX=1` merges "Head Angiogram or COW" → "Head angiogram or COW"
(21 classes). Everything else identical to Exp 1.

**Conditions** (all: real-row CV folds, thresholds tuned on real rows only, same seeds)
1. `baseline` — real data only.
2. `baseline_cw` — real data only, `class_weight="balanced"` (control).
3. `ros` — real data only; within each training fold, real rows of each augmented
   protocol resampled with replacement, adding as many copies as that protocol has
   synthetic rows (identical per-protocol dose to GPT-4o augmentation) (control).
4. `aug_realval` — GPT-4o augmentation with real-only tuning (Exp 1 method).
5. `aug_original` — GPT-4o augmentation with the repo's original tuning (reference).
6. `aug_realval_cb` — Exp 2 counterbalancing, k by formula (reference).

**Primary endpoint:** pooled sensitivity over all (case, protocol) pairs whose true
protocol is one of the 11 augmented protocols, on the real test set. Comparisons:
`aug_realval` vs `baseline` (does augmentation help), `aug_realval` vs `ros` and
`aug_realval` vs `baseline_cw` (is it the LLM text). Test: McNemar on paired
(case, protocol) outcomes; 95% CI for the difference by case-level bootstrap
(2,000 resamples, seed 42). Holm correction across the three primary comparisons.

**Key secondary endpoints:** pooled sensitivity over the 10 non-augmented protocols
(non-inferiority check; same tests); per-protocol sensitivity/precision with McNemar
+ Holm (21 tests); macro/micro/weighted F1, subset accuracy, Hamming loss, macro AUC,
top-3 accuracy; all endpoints repeated in CV on real training rows (5 × 4 folds,
thresholds for each fold tuned on the other folds).

**Decision rules (fixed in advance)**
- "LLM augmentation adds value beyond non-LLM rebalancing" if `aug_realval` beats both
  `ros` and `baseline_cw` on the primary endpoint (Holm-adjusted p < 0.05) **and** the
  CV primary endpoint points the same way. If not, the paper reports that, and the
  claim is narrowed accordingly.
- No condition is added, dropped or re-tuned after test results are seen.

**Results (run 2026-11-01; 21 classes after label fix; results file `exp3_results.xlsx`)**

Primary endpoint — pooled sensitivity, 11 augmented protocols (188 test pairs):

| Comparison | GPT-4o aug | Comparator | Difference (95% CI) | Gained / lost | Holm p |
|---|---|---|---|---|---|
| vs baseline | 41.0% | 22.3% | +18.6 pts (+12.3 to +25.3) | 41 / 6 | < 0.001 |
| vs random oversampling | 41.0% | 29.8% | +11.2 pts (+4.2 to +18.0) | 33 / 12 | 0.005 |
| vs class weighting | 41.0% | 35.1% | +5.9 pts (−1.6 to +13.0) | 30 / 19 | 0.15 |

CV primary endpoint (real training rows): baseline 32.6%, oversampling 36.0%,
class weighting **46.4%**, GPT-4o aug 44.7%.

**Decision rule: not met.** GPT-4o augmentation beats random oversampling of real
minority rows on test and in CV, but does not beat class weighting (test difference
not significant; CV favours class weighting). Per the pre-registered rule, the claim
is narrowed: LLM augmentation is better than duplicating real minority examples, and
comparable to class weighting on minority sensitivity.

Secondary — pooled sensitivity, 10 non-augmented protocols (1,040 test pairs):
GPT-4o aug 75.1% vs baseline 74.1% (+1.0 pts, 95% CI −0.2 to +2.1, p = 0.46 Holm);
no comparison showed a significant difference. CV: 76.6% vs 76.0%.

Secondary — global test metrics (baseline / class weighting / oversampling / GPT-4o aug):
macro F1 0.387 / 0.442 / 0.430 / **0.512**; micro F1 0.630 / 0.635 / 0.649 / **0.669**;
subset accuracy 0.487 / 0.451 / 0.491 / **0.517**; macro AUC 0.905 / 0.898 / 0.905 /
**0.933**; Hamming loss 0.045 / 0.046 / 0.043 / **0.040**; top-3 0.853 / 0.863 / 0.867 /
**0.901**; mean protocols advised 1.35 / 1.43 / 1.38 / 1.34.

**Observations (descriptive, not pre-registered):**
- The two methods help different protocols. GPT-4o is the only method that learns
  protocols with almost no real examples: Sinus NC (3 real training cases) 0% → 71%
  test / 0% → 100% CV, where class weighting stays at 0%; L Spine C+ and C Spine C+
  also favour GPT-4o in test. Class weighting does better where real examples exist
  (T Spine NC, Orbits NC, Neck Angiogram, Head & Neck Angio).
- Class weighting raises minority sensitivity by advising more protocols (1.43 per
  case) at the cost of precision: lowest subset accuracy and macro AUC of all
  conditions, and CV Head NC 85.8% → 81.8%. GPT-4o improves the precision-aware
  metrics (F1, AUC, Hamming) as well as sensitivity.
- Head C+ (test, 70 cases): baseline 54.3%, GPT-4o real-only 48.6% (4 lost, p = 0.13),
  counterbalanced 55.7%; CV 47.2% / 42.4% / 44.9%. Residual Head C+ cost persists.
- Label fix changed headline numbers slightly: baseline top-3 85.3% (was 87.6%),
  GPT-4o 90.1% (was 91.7%).

**Implication / next step:** the methods appear complementary (synthetic text for
protocols with ~no real data; reweighting for protocols with some). A combined
condition (GPT-4o augmentation + class weighting) is the natural next experiment and
must be pre-registered as Exp 4 before running.

---

## Exp 4 — Combined augmentation + class weighting; final model selection (pre-registered 2026-11-02 12:25 CT)

**Rationale (from Exp 3):** GPT-4o augmentation and class weighting improved different
protocols — synthetic text helped protocols with almost no real data (Sinus NC, L Spine
C+, C Spine C+), class weighting helped those with some real data (T Spine NC, Orbits
NC, Neck Angiogram). Their combination is the remaining untested candidate. Exp 4 is
also the **final model-selection step**: no new data are available for temporal or
external validation, so the configuration the paper is built on is chosen by a rule
fixed here, using cross-validation only.

**New condition:** `aug_cw` — GPT-4o synthetic rows in every training fold, thresholds
tuned on real rows only (as Exp 1), logistic regressions with `class_weight="balanced"`.

**Candidates for selection** (all with real-only threshold tuning, 21 classes):
baseline; class weighting; random oversampling; GPT-4o aug; GPT-4o aug + class
weighting; GPT-4o aug + counterbalance (k = 7). Excluded a priori: GPT-4o aug with the
repo's original tuning, because its thresholds are tuned on synthetic rows (Exp 1).

**Selection metrics** — CV on real training rows, 5 repeats × 4 folds, each fold's
thresholds tuned on the other three folds, pooled over all folds and repeats:
- S_aug: pooled sensitivity over the 11 augmented protocols
- S_non: pooled sensitivity over the 10 non-augmented protocols
- P_micro: micro-averaged precision over all 21 protocols
- macro F1

**Selection rule**
1. Eligible if S_non ≥ baseline S_non − 1.0 point **and** P_micro ≥ baseline
   P_micro − 1.0 point. (The precision condition guards against buying sensitivity by
   advising more protocols, a pattern seen with class weighting in Exp 3.)
2. Among eligible candidates, select the highest S_aug.
3. If the runner-up's S_aug is within 1.0 point of the leader, select whichever of the
   two has the higher CV macro F1.
4. If no augmented candidate is eligible, the paper reports that and baseline is the
   reference model.

**Safeguards**
- Two-stage analysis. Stage 1 (`analyze_exp4_select.py`) reads only cross-validation
  outputs and never loads test predictions; the selection is committed to this log
  before Stage 2 (`analyze_exp4_test.py`) computes any test metric.
- Seed stability (after selection, before Stage 2): the selected configuration and
  baseline are rerun with classifier-chain seeds 10–12 and 20–22. Reported as a
  range; not used for selection.
- Test reporting (Stage 2): primary = selected vs baseline on pooled augmented-protocol
  sensitivity (McNemar on paired case–protocol outcomes; 2,000-resample case-level
  bootstrap CI). Selected vs every other candidate reported with Holm correction.
  Non-augmented pooled sensitivity as secondary. All candidates shown in full.
- No further conditions will be added for the paper's main analysis after this point
  without a new pre-registered entry.

**Disclosure for the paper:** no independent hold-out set exists. The test set was
examined during Exps 0–3, including six Head C+ test referrals inspected during the
Exp 1 diagnosis, and test results for five of the six candidates were seen before
this entry. Exp 4 adds one new condition, and selection uses cross-validation only.

**Stage 1 — selection (CV only; recorded 2026-11-02 before any Exp 4 test metric)**

| Candidate | S_aug | S_non | P_micro | Macro F1 | Advised/case | Eligible |
|---|---|---|---|---|---|---|
| baseline | 32.6% (±3.0) | 76.0% | 67.5% | 0.40 | 1.31 | yes |
| class weighting | 46.4% (±1.5) | 75.8% | **64.3%** | 0.42 | 1.38 | no (precision) |
| random oversampling | 36.0% (±2.3) | 76.6% | 67.0% | 0.41 | 1.33 | yes |
| **GPT-4o aug** | **44.7% (±1.9)** | 76.6% | 67.7% | **0.48** | 1.33 | **yes** |
| GPT-4o aug + class weighting | 52.8% (±1.8) | 75.4% | **64.9%** | 0.46 | 1.37 | no (precision) |
| GPT-4o aug + counterbalance | 42.0% (±2.3) | 76.5% | 67.3% | 0.47 | 1.33 | yes |

(± = SD across the 5 CV repeats.) Eligibility thresholds: S_non ≥ 75.03%,
P_micro ≥ 66.49%.

**Selected: GPT-4o aug (real-only threshold tuning)** — rule 2, highest S_aug among
eligible candidates (lead over the next eligible candidate, counterbalance, 2.6 pts).

Note: the combined condition has the highest minority sensitivity of all (52.8%) but
loses 2.6 points of micro precision versus baseline, as does class weighting (−3.2);
both buy sensitivity by advising more protocols. They are reported in the paper as
alternatives favouring sensitivity over precision.

**Seed check (CV only; recorded before Stage 2):** chain seeds 0–2, 10–12, 20–22.

| Config | S_aug range | S_non range | P_micro range | Macro F1 range |
|---|---|---|---|---|
| baseline | 32.6–36.6% | 75.9–76.4% | 67.5–67.9% | 0.400–0.402 |
| GPT-4o aug | 44.0–45.7% | 75.9–76.6% | 67.5–67.8% | 0.476–0.482 |

The selected configuration's advantage on minority sensitivity holds for every seed
set (smallest gap 7.4 points), with no change in non-augmented sensitivity or precision.
(Two script bugs in `analyze_exp4_test.py`, a file-path and a missing import, were
fixed before this ran; no results were produced by the failed attempts.)

**Stage 2 — test results (1,001 real test cases; results file `exp4_results.xlsx`)**

Primary — selected (GPT-4o aug) vs baseline, pooled sensitivity over the 11 augmented
protocols (188 case–protocol pairs): **22.3% → 41.0%, +18.6 points (95% CI +12.3 to
+25.3), 41 gained / 6 lost, McNemar p < 0.001.**

Selected vs every candidate (Holm-adjusted):

| Comparator | Augmented-protocol sensitivity: selected vs comparator | Holm p | Non-augmented sensitivity: selected vs comparator | Holm p |
|---|---|---|---|---|
| baseline | 41.0 vs 22.3% | < 0.001 | 75.1 vs 74.1% | 0.62 |
| class weighting | 41.0 vs 35.1% | 0.30 | 75.1 vs 74.7% | 1.0 |
| random oversampling | 41.0 vs 29.8% | 0.010 | 75.1 vs 75.9% | 0.77 |
| GPT-4o aug + class weighting | 41.0 vs **48.9%** | **0.032** | 75.1 vs 75.4% | 1.0 |
| GPT-4o aug + counterbalance | 41.0 vs 39.9% | 0.69 | 75.1 vs **76.4%** | **0.022** |

Test, all candidates (S_aug / S_non / micro precision / macro F1 / top-3):
baseline 22.3 / 74.1 / 60.1 / 0.387 / 85.3; class weighting 35.1 / 74.7 / 59.0 / 0.442 /
86.3; oversampling 29.8 / 75.9 / 61.4 / 0.430 / 86.7; **GPT-4o aug 41.0 / 75.1 / 64.2 /
0.512 / 90.1**; GPT-4o aug + CW 48.9 / 75.4 / 63.2 / 0.540 / 89.6; GPT-4o aug +
counterbalance 39.9 / 76.4 / 64.0 / 0.511 / 89.9.

Seed check, test (seeds 0–2 / 10–12 / 20–22): baseline S_aug 22.3 / 25.0 / 22.3%;
GPT-4o aug 41.0 / 44.7 / 43.1%. Macro F1 baseline 0.386–0.406, GPT-4o aug 0.512–0.534.

**Interpretation**
- The selected configuration's primary result is large, significant, and stable across
  seeds, with no detectable loss on the non-augmented protocols.
- **On the test set, GPT-4o aug + class weighting outperformed the selected
  configuration** on minority sensitivity (+7.9 points, Holm p = 0.032) and macro F1,
  and its test precision (63.2%) was above baseline. In CV it lost 2.6 points of
  precision versus baseline, which made it ineligible under the pre-registered rule.
  The selection stands: switching to it now would be choosing the model on the test
  set. The paper reports it as a secondary finding and a candidate for prospective
  evaluation, with both the CV and test numbers.
- Counterbalancing gave slightly higher non-augmented sensitivity on test (+1.3 points,
  Holm p = 0.022) with similar minority sensitivity; also reported as secondary.
- Per-protocol: Sinus NC 0 → 71%, Temporal Bones C+ 12 → 41%, Facial Bones C+ 35 → 53%,
  L Spine C+ 0 → 18%; Head C+ 54 → 49% remains the one notable non-augmented decrease;
  Neck Angiogram 0% in both.

**Conclusion for the paper:** main model = GPT-4o augmentation with real-only threshold
tuning, chosen by the pre-registered CV rule.

---

## Final analysis — four-arm study for the paper (pre-registered 2026-11-02 18:50 CT)

**Decision by the authors (2026-11-02):** the paper will use a simplified, pre-specified
four-arm design. Exps 1–4 are preliminary development on an earlier version of this
dataset (same test cases); the paper discloses that in one sentence. Code:
`final_four_arm.py`, `analyze_final.py`.

**Data decisions and reasons**
- Base data = V7 (its split, exclusions and the 1,227 GPT-4o synthetic indications used
  for the ARRS submission).
- Ambiguous studies: the V8 column `exclude_record_based_on_anatomical_ambiguity` flags
  344 real studies whose indication was judged too vague to protocol, by a radiologist
  reading the indication only, before any model errors were seen. 333 are in the V7
  cohort (288 train, 45 test). They are removed from training and from the primary test
  set; the reference label for these studies is unreliable.
- "Head Angiogram or COW" merged into "Head angiogram or COW" (same protocol, two
  spellings).
- Neck NC removed: 67 of its 69 real studies are flagged ambiguous; 2 remain, both in
  training, none in test, so it could not be evaluated and would be learned almost
  entirely from synthetic text. Its 103 synthetic rows are removed too.
- Neck Angiogram kept (authors' decision, to avoid dropping protocols).
- C Spine C+ (13 real studies left) and L Spine C+ (18) fall below the 20-case cohort
  rule after ambiguous removal; kept, because the rule was applied when the cohort was
  defined, before ambiguous removal.
- Fixed synthetic dose: every augmented protocol is brought to the same total of real +
  synthetic training cases, = the largest total all can reach from the V7 synthetic
  rows: **103** (limited by T Spine NC: 66 real + 37 synthetic). 103 is below the
  smallest non-augmented protocol (Head Venogram, 157), so the frequency order of common
  protocols is preserved. Synthetic rows subsampled without replacement, seed 42:
  701 of 1,227 used.
- Result: 20 protocols (10 augmented), 7,505 real training studies, 956 primary test
  studies. Secondary test set = primary + 28 ambiguous test studies = 984 (the other 17
  ambiguous test studies were Neck NC and are excluded with that protocol).

**Arms** (same model, 5 × 4-fold CV over real training rows, thresholds tuned on real
rows only, 1–3 advised protocols): baseline; class weighting; GPT-4o augmentation;
GPT-4o augmentation + class weighting.

**Primary comparison:** GPT-4o augmentation + class weighting vs baseline, pooled
sensitivity over the 10 augmented protocols, primary test set. Exact McNemar on
case–protocol pairs; 95% CI by 2,000-resample case bootstrap (seed 42).

**No-cost check:** pooled sensitivity over the 10 common protocols, same comparison;
criterion = lower 95% CI bound above −2.0 percentage points. The difference and CI are
reported in full so the definition can be revisited.

**Secondary:** GPT-4o vs baseline, class weighting vs baseline, GPT-4o + class
weighting vs class weighting (Holm across the three), for rare and common protocols;
per-protocol sensitivity (Holm across 20); precision, exact match, complete-match recall,
top-3, F1, AUC, Hamming loss, advised per case; cross-validation estimates; everything
repeated on the secondary test set (ambiguous included).

**Results (run 2026-11-02; `final_results.xlsx`)**

Primary test set (956 studies, ambiguous excluded):
- **Primary:** rare-protocol sensitivity 45/152 (29.6%) → 76/152 (50.0%); +20.4 points
  (95% CI 11.8 to 29.1); 41 gained / 10 lost; McNemar P < .001.
- **No-cost check:** common-protocol sensitivity 761/1,000 (76.1%) → 776/1,000 (77.6%);
  +1.5 points (95% CI −0.3 to 3.3); lower bound above −2.0 → **criterion met**.
- Secondary, rare protocols (Holm): GPT-4o vs baseline +15.1 (68 vs 45/152, P < .001);
  class weighting vs baseline +11.2 (62 vs 45, P = .002); GPT-4o + CW vs class weighting
  +9.2 (76 vs 62, 95% CI 0.7 to 17.6, P = .039) — GPT-4o adds beyond class weighting.
- Secondary, common protocols (Holm): GPT-4o +0.1 (P = 1.0); class weighting +2.1
  (P = .056); GPT-4o + CW vs CW −0.6 (P = .69).
- Precision 806/1,265 (63.7%) → 852/1,348 (63.2%); advised per case 1.32 → 1.41;
  exact match 498 → 477; all reference protocols advised 641 → 689; top-3 842 → 862;
  macro F1 0.437 → 0.544; weighted F1 0.658 → 0.698; macro AUC 0.922 → 0.932.
- Per protocol (Holm across 20): Sinus NC 0 → 13/17 (P = .005); TIA 150 → 160/184
  (P = .037); Temporal Bones C+ 2 → 11/17 (unadjusted .004, adjusted .070).
  Largest decrease: Head NC 394 → 383/476 (22 lost, 11 gained; unadjusted P = .080).
  Neck Angiogram 0/13 in both.

Secondary test set (984, ambiguous included): rare 47/168 → 85/168 (+22.6, 95% CI
14.4 to 31.0); common 771/1,024 → 787/1,024 (+1.6, 95% CI −0.2 to 3.4).

CV (real training rows): rare 41.1% → 55.4%; common 76.0% → 76.9%; precision
68.6% → 65.6%.

---

## Secondary operating-point analysis (added 2026-11-05, after the final results were seen)

**Rationale:** the authors asked whether advising about 2 protocols per case instead
of ~1.4 enlarges the gain from augmentation. This is a post-hoc secondary analysis;
the pre-specified final results above remain the main results. No model is retrained:
only the rule that turns probabilities into advised protocols changes. Code:
`analyze_operating_points.py`.

**Rules, applied identically to all four arms, both test sets:**
- (a) Top-2 (main version of this analysis): advise exactly the 2 most probable
  protocols. Nothing is tuned.
- (b) Scaled thresholds: multiply every protocol threshold by one factor, chosen on
  cross-validation (real training rows, out-of-fold probabilities) so the mean number
  advised is about 2.0; 1–3 protocols per case as before. Factor set per arm without
  looking at the test set.
- Curve: (b) repeated for CV-calibrated targets of 1.25 to 2.75 advised per case, plus
  fixed top-1, top-2, top-3.

**Reported:** rare and common pooled sensitivity with paired comparisons vs baseline
(McNemar, case bootstrap CI), precision, exact match, complete-match recall, F1, Hamming
loss, advised per case, and per-protocol sensitivity under top-2.

**Results (run 2026-11-05; `operating_points.xlsx`), primary test set (956)**
- CV-calibrated threshold factors for ~2.0 advised: baseline 0.39, class weighting
  0.50, GPT-4o 0.37, GPT-4o + CW 0.51. Test advised per case 2.05–2.07.
- Scaled thresholds, ~2 advised: rare sensitivity baseline 38/152 (25.0%), class
  weighting 77 (50.7%), GPT-4o 84 (55.3%), GPT-4o + CW 96 (63.2%). GPT-4o vs baseline
  +30.3 points (95% CI 22.8 to 38.6; 48 gained / 2 lost); common +0.1 (−1.0 to 1.1).
  GPT-4o + CW vs baseline +38.2 (29.8 to 46.7); common −1.9 (−3.6 to −0.3), lower bound
  below −2: would fail the no-cost criterion at this operating point.
- Precision falls for every arm (baseline 63.7% → 46.6%; GPT-4o 67.3% → 48.8%); exact
  match falls (baseline 498 → 285; GPT-4o 537 → 295); complete-match recall rises
  (baseline 641 → 743; GPT-4o 666 → 784). Macro F1 falls (baseline 0.437 → 0.381;
  GPT-4o 0.533 → 0.515) but the gap between them widens (0.096 → 0.134).
- Why the gain grows: baseline puts its extra suggestions almost entirely on common
  protocols (rare labels advised 105 → 118; common 1,160 → 1,846), so its rare
  sensitivity does not rise. GPT-4o puts many on rare protocols (122 → 191).
- Top-2 (fixed): rare baseline 21/152, GPT-4o 63, GPT-4o + CW 87; exact match collapses
  (75–93 of 956) because ~82% of cases have one reference protocol. Forcing exactly two
  also lowers baseline rare sensitivity below its current value, which exaggerates the
  gain; scaled thresholds are the fairer version of this analysis.
- Ambiguous-included test set: same pattern (GPT-4o vs baseline at ~2: +31.6 rare,
  +0.1 common).

---

## Open items for the paper (not experiments)

- [ ] Radiologist verification of synthetic rows: number reviewed, number accepted /
      edited / rejected, reviewer, criteria.
- [ ] GPT-4o prompt(s), generation date, temperature/settings, and how the per-protocol
      target (~150 training cases) was set.
- [ ] Methods text to match the code (see Model section) and a single top-k definition.
- [ ] Correct author spelling (Jason vs Jaron Chong).
- [ ] Planned, not yet run: dose-response (50/100/150 synthetic per protocol, by
      subsampling); multiple seeds; GPT-4o zero-shot protocoling comparator (needs
      API access outside this workspace); regeneration with local exemplars.
