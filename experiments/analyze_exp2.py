"""Experiment 2 analysis: counterbalanced augmentation vs baseline and Experiment 1.
usage: python analyze_exp2.py <runs_dir> <primary_k>
Uses the same threshold rule, paired McNemar tests and CV check as analyze.py."""
import sys
import numpy as np, pandas as pd
from scipy.stats import binomtest
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

R, KP = sys.argv[1], int(sys.argv[2])
src = open(__file__.replace("analyze_exp2.py", "analyze.py")).read()
exec(src[src.index("GRID ="):src.index("Z = {")])  # tune, apply, topk, global_metrics, holm, AUG

def folds(z, real_only, r):
    syn, Y = z["syn"], z["Y_tr"]
    rows = np.where(~syn)[0] if real_only else np.arange(len(Y))
    for a, b in MultilabelStratifiedKFold(4, shuffle=True, random_state=r).split(rows, Y[rows]):
        yield rows[b]

def fold_avg_thresholds(z, real_only):
    return np.mean([tune(z["Y_tr"][v], z["oof"][r][v]) for r in range(z["oof"].shape[0])
                    for v in folds(z, real_only, r)], axis=0)

def cv_sens(z, real_only):  # thresholds for each fold tuned on the other folds; scored on real rows
    oof, Yt, syn = z["oof"], z["Y_tr"], z["syn"]; tp = np.zeros(Yt.shape[1]); pos = np.zeros(Yt.shape[1])
    for r in range(oof.shape[0]):
        fl = list(folds(z, real_only, r))
        for k, v in enumerate(fl):
            t_ = np.mean([tune(Yt[w], oof[r][w]) for i, w in enumerate(fl) if i != k], axis=0)
            v = v[~syn[v]]; yp = apply(oof[r][v], t_)
            tp += ((Yt[v] == 1) & (yp == 1)).sum(0); pos += (Yt[v] == 1).sum(0)
    return tp / pos

runs = {"baseline": ("baseline", False), "Exp1: real-only tuning": ("aug_realval", True),
        f"Exp2: counterbalanced k={KP} [PRIMARY]": (f"aug_realval_cb_k{KP}", True)}
robust = {f"Exp2 k={k}": (f"aug_realval_cb_k{k}", True) for k in (3, 14)}
Z = {n: np.load(f"{R}/{f}.npz", allow_pickle=True) for n, (f, _) in {**runs, **robust}.items()}
labels = list(Z["baseline"]["labels"]); Y = Z["baseline"]["Y_te"]
PRED = {n: apply(Z[n]["test_prob"], fold_avg_thresholds(Z[n], ro)) for n, (_, ro) in runs.items()}
pd.set_option("display.width", 250)

g = pd.DataFrame({n: global_metrics(Y, PRED[n], Z[n]["test_prob"]) for n in runs}).T
print("=== TEST GLOBAL METRICS ==="); print(g.round(3).to_string())

def sens(yp): return ((Y == 1) & (yp == 1)).sum(0) / (Y == 1).sum(0)
def prec(yp):
    tp = ((Y == 1) & (yp == 1)).sum(0); n = yp.sum(0); return np.where(n > 0, tp / np.maximum(n, 1), np.nan)
B = "baseline"; E1 = "Exp1: real-only tuning"; E2 = list(runs)[2]
rows = []
for j, L in enumerate(labels):
    pos = Y[:, j] == 1
    if pos.sum() < 5: continue
    lost = int((pos & (PRED[B][:, j] == 1) & (PRED[E2][:, j] == 0)).sum())
    gain = int((pos & (PRED[B][:, j] == 0) & (PRED[E2][:, j] == 1)).sum())
    rows.append(dict(protocol=L, augmented=L in AUG, n_test=int(pos.sum()), base=sens(PRED[B])[j], exp1=sens(PRED[E1])[j],
                     exp2=sens(PRED[E2])[j], prec_base=prec(PRED[B])[j], prec_exp2=prec(PRED[E2])[j],
                     lost=lost, gained=gain, mcnemar_p=binomtest(gain, gain + lost).pvalue if gain + lost else 1.0))
t = pd.DataFrame(rows); t["holm_p"] = holm(t.mcnemar_p); t = t.sort_values("n_test", ascending=False)
print("\n=== TEST PER-PROTOCOL SENSITIVITY (paired McNemar: baseline vs Exp2 primary) ==="); print(t.round(3).to_string(index=False))
for c in ["base", "exp1", "exp2"]:
    print(f"mean sensitivity {c}: augmented {t[t.augmented][c].mean():.3f} | non-augmented {t[~t.augmented][c].mean():.3f}")

cv = pd.DataFrame({n: cv_sens(Z[n], ro) for n, (_, ro) in {**runs, **robust}.items()}, index=labels)
cv.insert(0, "real_train_n", Z["baseline"]["Y_tr"].sum(0))
cv = cv[cv.real_train_n >= 5].sort_values("real_train_n", ascending=False)
print("\n=== CV SENSITIVITY, REAL TRAINING ROWS ==="); print(cv.round(3).to_string())
na = cv.index.map(lambda l: l not in AUG)
print("CV mean, non-augmented:", cv[na].iloc[:, 1:].mean().round(3).to_dict())
print("CV mean, augmented:   ", cv[~na].iloc[:, 1:].mean().round(3).to_dict())
t.to_csv(f"{R}/exp2_per_protocol_test.csv", index=False); cv.to_csv(f"{R}/exp2_cv_sensitivity.csv"); g.to_csv(f"{R}/exp2_test_global.csv")
