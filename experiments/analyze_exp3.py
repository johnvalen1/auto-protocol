"""Experiment 3 analysis, as pre-registered in EXPERIMENT_LOG.md.
usage: python analyze_exp3.py <runs_dir>"""
import sys
import numpy as np, pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score, roc_auc_score, hamming_loss, accuracy_score
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

R = sys.argv[1]
src = open(__file__.replace("analyze_exp3.py", "analyze.py")).read()
exec(src[src.index("GRID ="):src.index("Z = {")])  # tune, apply, topk, global_metrics, holm, AUG

COND = {  # name: (file, folds over real rows only)
    "baseline": ("baseline", True), "class weighting": ("baseline_cw", True),
    "oversampling (ROS)": ("ros", True), "GPT-4o aug, real-only tuning": ("aug_realval", True),
    "GPT-4o aug, repo tuning": ("aug_original", False), "GPT-4o aug + counterbalance": None}
import glob, os
cb = sorted(glob.glob(f"{R}/aug_realval_cb_k*.npz")); COND["GPT-4o aug + counterbalance"] = (os.path.basename(cb[0])[:-4], True)
Z = {n: np.load(f"{R}/{f}.npz", allow_pickle=True) for n, (f, _) in COND.items()}
labels = list(Z["baseline"]["labels"]); Y = Z["baseline"]["Y_te"]
for n in Z: assert list(Z[n]["labels"]) == labels and (Z[n]["Y_te"] == Y).all(), n
print("classes:", len(labels), "| label fix:", bool(Z["baseline"]["label_fix"]))
A = np.array([l in AUG for l in labels])

def folds(z, real_only, r):
    syn, Yt = z["syn"], z["Y_tr"]; rows = np.where(~syn)[0] if real_only else np.arange(len(Yt))
    for a, b in MultilabelStratifiedKFold(4, shuffle=True, random_state=r).split(rows, Yt[rows]):
        yield rows[b]

def thr(z, ro):
    return np.mean([tune(z["Y_tr"][v], z["oof"][r][v]) for r in range(z["oof"].shape[0]) for v in folds(z, ro, r)], axis=0)

PRED = {n: apply(Z[n]["test_prob"], thr(Z[n], ro)) for n, (_, ro) in COND.items()}
pd.set_option("display.width", 250)

# ---------- primary & key secondary endpoints (test) ----------
def pooled(yp, mask): return ((Y[:, mask] == 1) & (yp[:, mask] == 1)).sum() / (Y[:, mask] == 1).sum()

def compare(new, ref, mask, n_boot=2000):
    pos = Y[:, mask] == 1; a = (PRED[new][:, mask] == 1) & pos; b = (PRED[ref][:, mask] == 1) & pos
    gain = int((a & ~b).sum()); lost = int((~a & b).sum())
    p = binomtest(gain, gain + lost).pvalue if gain + lost else 1.0
    rng = np.random.default_rng(42); d = []
    for _ in range(n_boot):
        i = rng.integers(0, len(Y), len(Y)); P = pos[i].sum()
        d.append((a[i].sum() - b[i].sum()) / P)
    return dict(new=new, ref=ref, sens_new=pooled(PRED[new], mask), sens_ref=pooled(PRED[ref], mask),
                diff=pooled(PRED[new], mask) - pooled(PRED[ref], mask), ci_lo=np.percentile(d, 2.5),
                ci_hi=np.percentile(d, 97.5), gained=gain, lost=lost, mcnemar_p=p)

G = "GPT-4o aug, real-only tuning"
prim = pd.DataFrame([compare(G, r, A) for r in ["baseline", "oversampling (ROS)", "class weighting"]])
prim["holm_p"] = holm(prim.mcnemar_p)
print(f"\n=== PRIMARY: pooled sensitivity, 11 augmented protocols (test pairs n={int((Y[:, A] == 1).sum())}) ===")
print(prim.round(4).to_string(index=False))
sec = pd.DataFrame([compare(G, r, ~A) for r in ["baseline", "oversampling (ROS)", "class weighting"]])
sec["holm_p"] = holm(sec.mcnemar_p)
print(f"\n=== SECONDARY: pooled sensitivity, 10 non-augmented protocols (test pairs n={int((Y[:, ~A] == 1).sum())}) ===")
print(sec.round(4).to_string(index=False))

print("\n=== POOLED SENSITIVITY, ALL CONDITIONS (test) ===")
for n in COND: print(f"{n:32s} augmented {pooled(PRED[n], A):.3f} | non-augmented {pooled(PRED[n], ~A):.3f}")

g = pd.DataFrame({n: global_metrics(Y, PRED[n], Z[n]["test_prob"]) for n in COND}).T
print("\n=== TEST GLOBAL METRICS ==="); print(g.round(3).to_string())

# ---------- per-protocol (test) ----------
rows = []
for j, L in enumerate(labels):
    pos = Y[:, j] == 1; r_ = dict(protocol=L, augmented=L in AUG, n_test=int(pos.sum()))
    for n in COND: r_[n] = ((PRED[n][:, j] == 1) & pos).sum() / pos.sum()
    gain = int((pos & (PRED[G][:, j] == 1) & (PRED["baseline"][:, j] == 0)).sum())
    lost = int((pos & (PRED[G][:, j] == 0) & (PRED["baseline"][:, j] == 1)).sum())
    r_.update(gained=gain, lost=lost, mcnemar_p=binomtest(gain, gain + lost).pvalue if gain + lost else 1.0)
    rows.append(r_)
pp = pd.DataFrame(rows); pp["holm_p"] = holm(pp.mcnemar_p); pp = pp.sort_values("n_test", ascending=False)
print("\n=== PER-PROTOCOL TEST SENSITIVITY (McNemar/Holm: GPT-4o real-only vs baseline) ===")
print(pp.round(3).to_string(index=False))

# ---------- CV on real training rows ----------
def cv(z, ro):
    oof, Yt, syn = z["oof"], z["Y_tr"], z["syn"]; tp = np.zeros(len(labels)); ps = np.zeros(len(labels))
    for r in range(oof.shape[0]):
        fl = list(folds(z, ro, r))
        for k, v in enumerate(fl):
            t_ = np.mean([tune(Yt[w], oof[r][w]) for i, w in enumerate(fl) if i != k], axis=0)
            v = v[~syn[v]]; yp = apply(oof[r][v], t_)
            tp += ((Yt[v] == 1) & (yp == 1)).sum(0); ps += (Yt[v] == 1).sum(0)
    return tp, ps
CV = {n: cv(Z[n], ro) for n, (_, ro) in COND.items()}
print("\n=== CV POOLED SENSITIVITY, REAL TRAINING ROWS (5x4 folds) ===")
for n, (tp, ps) in CV.items(): print(f"{n:32s} augmented {tp[A].sum() / ps[A].sum():.3f} | non-augmented {tp[~A].sum() / ps[~A].sum():.3f}")
cvt = pd.DataFrame({n: tp / ps for n, (tp, ps) in CV.items()}, index=labels)
cvt.insert(0, "real_train_n", Z["baseline"]["Y_tr"].sum(0)); cvt = cvt.sort_values("real_train_n", ascending=False)
print("\n=== CV PER-PROTOCOL SENSITIVITY ==="); print(cvt.round(3).to_string())

with pd.ExcelWriter(f"{R}/exp3_results.xlsx") as w:
    prim.to_excel(w, sheet_name="primary_augmented", index=False); sec.to_excel(w, sheet_name="secondary_nonaugmented", index=False)
    g.to_excel(w, sheet_name="test_global"); pp.to_excel(w, sheet_name="test_per_protocol", index=False)
    cvt.to_excel(w, sheet_name="cv_per_protocol")
print("\nsaved", f"{R}/exp3_results.xlsx")
