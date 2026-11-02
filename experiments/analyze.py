"""Analyze threshold_fix.py outputs. usage: python analyze.py <runs_dir>"""
import sys, json
import numpy as np, pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score, roc_auc_score, hamming_loss, accuracy_score
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

R = sys.argv[1]
GRID = np.linspace(0.05, 0.95, 19)
AUG = ['T Spine NC', 'Facial Bones C+', 'Orbits C+', 'Neck NC', 'Orbits NC', 'Temporal Bones NC',
       'Neck Angiogram', 'Temporal Bones C+', 'C Spine C+', 'L Spine C+', 'Sinus NC']

def tune(y, p):  # identical rule to repo optimize_thresholds
    th = np.full(y.shape[1], 0.5)
    for i in range(y.shape[1]):
        best = 0
        for t in GRID:
            f = f1_score(y[:, i], (p[:, i] >= t).astype(int), zero_division=0)
            if f > best: best, th[i] = f, t
    return th

def apply(p, th, lo=1, hi=3):  # identical to repo apply_thresholds_with_limits
    out = np.zeros_like(p, dtype=int)
    for i in range(len(p)):
        L = np.where(p[i] >= th)[0]
        if len(L) < lo: L = np.argsort(p[i])[-lo:]
        if len(L) > hi: L = L[np.argsort(p[i][L])[-hi:]]
        out[i, L] = 1
    return out

def folds(z, scheme, r):
    syn = z["syn"]; Y = z["Y_tr"]
    rows = np.where(~syn)[0] if scheme == "aug_realval" else np.arange(len(Y))
    for a, b in MultilabelStratifiedKFold(4, shuffle=True, random_state=r).split(rows, Y[rows]):
        yield rows[b]

def thresholds(z, scheme, method):
    """fold_avg: repo method (tune per fold on that fold's validation rows, average).
       real_fold_avg: same but only real validation rows used in tuning.
       pooled_real: tune once on pooled out-of-fold predictions of real rows."""
    oof, Y, syn = z["oof"], z["Y_tr"], z["syn"]
    per = []
    for r in range(oof.shape[0]):
        if method == "pooled_real":
            per.append(tune(Y[~syn], oof[r][~syn])); continue
        for v in folds(z, scheme, r):
            if method == "real_fold_avg": v = v[~syn[v]]
            per.append(tune(Y[v], oof[r][v]))
    return np.mean(per, axis=0)

def topk(y, p, k=3):  # identical to repo: any true label in top-3 probabilities
    return np.mean([len(set(np.argsort(p[i])[-k:]) & set(np.where(y[i])[0])) > 0 for i in range(len(y))])

def global_metrics(y, yp, p):
    return dict(top3_any=topk(y, p), all_true_in_advised=np.mean(((y == 1) & (yp == 0)).sum(1) == 0),
                subset_acc=accuracy_score(y, yp), micro_f1=f1_score(y, yp, average="micro", zero_division=0),
                macro_f1=f1_score(y, yp, average="macro", zero_division=0),
                weighted_f1=f1_score(y, yp, average="weighted", zero_division=0),
                hamming=hamming_loss(y, yp), macro_auc=roc_auc_score(y, p, average="macro"),
                mean_advised=yp.sum(1).mean())

def holm(ps):
    ps = np.asarray(ps); o = np.argsort(ps); adj = np.empty_like(ps); m = len(ps); run = 0
    for rank, i in enumerate(o):
        run = max(run, min(1, (m - rank) * ps[i])); adj[i] = run
    return adj

Z = {s: np.load(f"{R}/{s}.npz", allow_pickle=True) for s in ["baseline", "aug_original", "aug_realval"]}
labels = list(Z["baseline"]["labels"]); assert labels == list(Z["aug_realval"]["labels"])
Y = Z["baseline"]["Y_te"]; assert (Y == Z["aug_realval"]["Y_te"]).all()

configs = {  # name: (scheme, threshold method)
    "baseline (repo)": ("baseline", "fold_avg"),
    "augmented (repo)": ("aug_original", "fold_avg"),
    "augmented, real-only tuning [PRIMARY]": ("aug_realval", "fold_avg"),
    "baseline, pooled OOF tuning": ("baseline", "pooled_real"),
    "augmented, real-only pooled OOF tuning": ("aug_realval", "pooled_real"),
}
TH, PRED = {}, {}
for name, (s, m) in configs.items():
    TH[name] = thresholds(Z[s], s, m); PRED[name] = apply(Z[s]["test_prob"], TH[name])

# ---- 1. reproduction check against p_value_computations_1 ----
glob = pd.DataFrame({n: global_metrics(Y, PRED[n], Z[configs[n][0]]["test_prob"]) for n in configs}).T
print("\n=== TEST SET GLOBAL METRICS (n=%d) ===" % len(Y)); print(glob.round(3).to_string())

# ---- 2. per-protocol sensitivity & precision on test ----
def per_class(yp):
    tp = ((Y == 1) & (yp == 1)).sum(0); fn = ((Y == 1) & (yp == 0)).sum(0); fp = ((Y == 0) & (yp == 1)).sum(0)
    return tp / (tp + fn), np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), np.nan)

def paired(ref, new):
    rows = []
    for j, L in enumerate(labels):
        pos = Y[:, j] == 1
        lost = int((pos & (PRED[ref][:, j] == 1) & (PRED[new][:, j] == 0)).sum())
        gain = int((pos & (PRED[ref][:, j] == 0) & (PRED[new][:, j] == 1)).sum())
        p = binomtest(gain, gain + lost).pvalue if gain + lost else 1.0
        sr, pr = per_class(PRED[ref]); sn, pn = per_class(PRED[new])
        rows.append(dict(protocol=L, augmented=L in AUG, n_test=int(pos.sum()), sens_ref=sr[j], sens_new=sn[j],
                         prec_ref=pr[j], prec_new=pn[j], lost=lost, gained=gain, mcnemar_p=p))
    t = pd.DataFrame(rows); t["holm_p"] = holm(t.mcnemar_p); return t.sort_values("n_test", ascending=False)

pd.set_option("display.width", 250)
for ref, new in [("baseline (repo)", "augmented (repo)"),
                 ("baseline (repo)", "augmented, real-only tuning [PRIMARY]"),
                 ("baseline, pooled OOF tuning", "augmented, real-only pooled OOF tuning")]:
    t = paired(ref, new)
    print(f"\n=== PER-PROTOCOL TEST: {ref}  ->  {new} ==="); print(t.round(3).to_string(index=False))
    t.to_csv(f"{R}/per_protocol__{new.split(' [')[0].replace(' ', '_').replace(',', '')}.csv", index=False)
    print("mean sensitivity, 11 augmented:", round(t[t.augmented].sens_ref.mean(), 3), "->", round(t[t.augmented].sens_new.mean(), 3),
          "| 10 non-augmented:", round(t[~t.augmented].sens_ref.mean(), 3), "->", round(t[~t.augmented].sens_new.mean(), 3))

# ---- 3. thresholds ----
th = pd.DataFrame(TH, index=labels); print("\n=== THRESHOLDS ==="); print(th.round(3).to_string())

# ---- 4. cross-validation check on REAL training rows (much larger n than test) ----
# thresholds for each fold are tuned on the OTHER folds of the same repeat (no self-tuning)
def cv_sens(s, method):
    z = Z[s]; oof, Yt, syn = z["oof"], z["Y_tr"], z["syn"]; tp = np.zeros(len(labels)); pos = np.zeros(len(labels))
    for r in range(oof.shape[0]):
        fl = list(folds(z, s, r))
        for k, v in enumerate(fl):
            others = [w for i, w in enumerate(fl) if i != k]
            if method == "real_fold_avg": others = [w[~syn[w]] for w in others]
            t_ = np.mean([tune(Yt[w], oof[r][w]) for w in others], axis=0)
            v = v[~syn[v]]; yp = apply(oof[r][v], t_)
            tp += ((Yt[v] == 1) & (yp == 1)).sum(0); pos += (Yt[v] == 1).sum(0)
    return tp / pos, pos / oof.shape[0]

cvb, npos = cv_sens("baseline", "fold_avg"); cva, _ = cv_sens("aug_original", "fold_avg"); cvr, _ = cv_sens("aug_realval", "real_fold_avg")
cv = pd.DataFrame(dict(real_train_n=npos.astype(int), baseline=cvb, aug_repo=cva, aug_realval=cvr), index=labels).sort_values("real_train_n", ascending=False)
print("\n=== CV SENSITIVITY ON REAL TRAINING ROWS (5x4 folds, averaged) ==="); print(cv.round(3).to_string())
cv.to_csv(f"{R}/cv_sensitivity.csv"); glob.to_csv(f"{R}/test_global_metrics.csv"); th.to_csv(f"{R}/thresholds.csv")
