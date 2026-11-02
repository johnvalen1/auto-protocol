"""Experiment 4, Stage 1: model selection from cross-validation ONLY (pre-registered).
Reads only out-of-fold predictions on real training rows. Never accesses test arrays
(test_prob, Y_te, test_text): the guarded loader below raises if any is requested.
usage: python analyze_exp4_select.py <runs_dir>"""
import sys, json
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

R = sys.argv[1]
src = open(__file__.replace("analyze_exp4_select.py", "analyze.py")).read()
exec(src[src.index("GRID ="):src.index("def topk")])  # tune, apply, AUG (identical rules)

class CVOnly:
    ALLOWED = {"oof", "Y_tr", "syn", "labels"}
    def __init__(self, path): self.z = np.load(path, allow_pickle=True)
    def __getitem__(self, k):
        if k not in self.ALLOWED: raise KeyError(f"Stage 1 may not read '{k}'")
        return self.z[k]

CAND = {"baseline": "baseline", "class weighting": "baseline_cw", "random oversampling": "ros",
        "GPT-4o aug": "aug_realval", "GPT-4o aug + class weighting": "aug_cw",
        "GPT-4o aug + counterbalance": "aug_realval_cb_k7"}
Z = {n: CVOnly(f"{R}/{f}.npz") for n, f in CAND.items()}
labels = list(Z["baseline"]["labels"]); A = np.array([l in AUG for l in labels])
for n in Z: assert list(Z[n]["labels"]) == labels

def folds(z, r):
    syn, Yt = z["syn"], z["Y_tr"]; rows = np.where(~syn)[0]
    for a, b in MultilabelStratifiedKFold(4, shuffle=True, random_state=r).split(rows, Yt[rows]):
        yield rows[b]

def cv_counts(z):
    """Per repeat: TP, FN, FP per label and labels advised, on real out-of-fold rows."""
    oof, Yt, syn = z["oof"], z["Y_tr"], z["syn"]; out = []
    for r in range(oof.shape[0]):
        fl = list(folds(z, r)); tp = np.zeros(len(labels)); fn = tp.copy(); fp = tp.copy(); adv = 0; n = 0
        for k, v in enumerate(fl):
            t_ = np.mean([tune(Yt[w], oof[r][w]) for i, w in enumerate(fl) if i != k], axis=0)
            yp = apply(oof[r][v], t_); y = Yt[v]
            tp += ((y == 1) & (yp == 1)).sum(0); fn += ((y == 1) & (yp == 0)).sum(0); fp += ((y == 0) & (yp == 1)).sum(0)
            adv += yp.sum(); n += len(v)
        out.append((tp, fn, fp, adv / n))
    return out

def metrics(tp, fn, fp, adv):
    f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0)
    return dict(S_aug=tp[A].sum() / (tp[A] + fn[A]).sum(), S_non=tp[~A].sum() / (tp[~A] + fn[~A]).sum(),
                P_micro=tp.sum() / (tp + fp).sum(), macro_F1=f1.mean(), advised_per_case=adv)

rows, sds = {}, {}
for n, z in Z.items():
    c = cv_counts(z)
    rows[n] = metrics(*[sum(x[i] for x in c) for i in range(3)], np.mean([x[3] for x in c]))
    per = pd.DataFrame([metrics(*x) for x in c]); sds[n] = per.std().to_dict()
T = pd.DataFrame(rows).T; SD = pd.DataFrame(sds).T

b = T.loc["baseline"]
T["eligible"] = (T.S_non >= b.S_non - 0.01) & (T.P_micro >= b.P_micro - 0.01)
elig = T[T.eligible].sort_values("S_aug", ascending=False)
aug_elig = elig.drop(index="baseline", errors="ignore")
if len(aug_elig) == 0:
    chosen, why = "baseline", "rule 4: no augmented candidate eligible"
else:
    lead = elig.index[0]; chosen, why = lead, "rule 2: highest S_aug among eligible"
    if len(elig) > 1 and elig.S_aug.iloc[0] - elig.S_aug.iloc[1] < 0.01:
        two = elig.iloc[:2]; chosen = two.macro_F1.idxmax()
        why = f"rule 3: top two within 1.0 pt on S_aug ({elig.index[0]} vs {elig.index[1]}); higher macro F1"

pd.set_option("display.width", 250)
print("=== EXP 4 STAGE 1: CV SELECTION METRICS (real training rows, 5x4 folds, pooled) ===")
print((T.drop(columns="eligible") * [100, 100, 100, 1, 1]).round(2).assign(eligible=T.eligible).to_string())
print("\n=== SD ACROSS THE 5 CV REPEATS ===")
print((SD * [100, 100, 100, 1, 1]).round(2).to_string())
print(f"\nbaseline thresholds for eligibility: S_non >= {100*(b.S_non-0.01):.2f}%, P_micro >= {100*(b.P_micro-0.01):.2f}%")
print(f"SELECTED: {chosen}  ({why})")
T.assign(sd_S_aug=SD.S_aug, sd_S_non=SD.S_non, sd_P_micro=SD.P_micro).to_csv(f"{R}/exp4_stage1_selection.csv")
json.dump(dict(selected=chosen, file=CAND[chosen], reason=why), open(f"{R}/exp4_selected.json", "w"))
