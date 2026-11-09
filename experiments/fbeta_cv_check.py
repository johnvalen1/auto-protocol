"""CV-only check: what happens when per-protocol thresholds maximise F-beta instead of F1?
Reads only out-of-fold predictions on real training rows; test arrays are blocked.
usage: python fbeta_cv_check.py <final_out_dir>"""
import sys
import numpy as np, pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

R = sys.argv[1]
GRID = np.linspace(0.05, 0.95, 19)

class CVOnly:
    OK = {"oof", "Y_tr", "syn", "labels", "aug"}
    def __init__(self, p): self.z = np.load(p, allow_pickle=True)
    def __getitem__(self, k):
        if k not in self.OK: raise KeyError(f"test data blocked: {k}")
        return self.z[k]

def fbeta(y, pred, b):  # vectorised over columns of pred (n x G)
    tp = (pred & y[:, None]).sum(0); fp = (pred & ~y[:, None]).sum(0); fn = (~pred & y[:, None]).sum(0)
    den = (1 + b * b) * tp + b * b * fn + fp
    return np.where(den > 0, (1 + b * b) * tp / np.maximum(den, 1), 0)

def tune(Y, P, b):  # same rule as the repo (grid 0.05-0.95, strict improvement, default 0.5)
    th = np.full(Y.shape[1], 0.5)
    for j in range(Y.shape[1]):
        s = fbeta(Y[:, j].astype(bool), P[:, j][:, None] >= GRID[None, :], b)
        if s.max() > 0: th[j] = GRID[int(np.argmax(s))]
    return th

def apply(p, th, lo=1, hi=3):
    out = np.zeros_like(p, dtype=int)
    for i in range(len(p)):
        L = np.where(p[i] >= th)[0]
        if len(L) < lo: L = np.argsort(p[i])[-lo:]
        if len(L) > hi: L = L[np.argsort(p[i][L])[-hi:]]
        out[i, L] = 1
    return out

def folds(z, r):
    rows = np.where(~z["syn"])[0]
    for a, b in MultilabelStratifiedKFold(4, shuffle=True, random_state=r).split(rows, z["Y_tr"][rows]): yield rows[b]

out = []
for arm, name in [("baseline", "Baseline"), ("gpt4o", "GPT-4o augmentation")]:
    z = CVOnly(f"{R}/{arm}.npz"); Y = z["Y_tr"]; oof = z["oof"]
    A = np.array([l in list(z["aug"]) for l in z["labels"]])
    for b in [1.0, 1.5, 2.0, 2.5, 3.0]:
        tp = np.zeros(Y.shape[1]); fn = tp.copy(); fp = tp.copy(); adv = n = 0; ths = []
        for r in range(oof.shape[0]):
            fl = list(folds(z, r))
            for k, v in enumerate(fl):
                th = np.mean([tune(Y[w], oof[r][w], b) for i, w in enumerate(fl) if i != k], axis=0); ths.append(th)
                yp = apply(oof[r][v], th); y = Y[v]
                tp += ((y == 1) & (yp == 1)).sum(0); fn += ((y == 1) & (yp == 0)).sum(0); fp += ((y == 0) & (yp == 1)).sum(0)
                adv += yp.sum(); n += len(v)
        out.append(dict(model=name, beta=b, advised_per_case=adv / n, rare_sens=tp[A].sum() / (tp[A] + fn[A]).sum(),
                        common_sens=tp[~A].sum() / (tp[~A] + fn[~A]).sum(), precision=tp.sum() / (tp + fp).sum(),
                        mean_threshold=np.mean(ths)))
        print(out[-1], flush=True)
t = pd.DataFrame(out); pd.set_option("display.width", 200)
print(t.round(3).to_string(index=False))
true = Y[~z["syn"]].sum(1).mean(); print("true protocols per real training case:", round(true, 3))
t.to_csv(f"{R}/fbeta_cv_check.csv", index=False)
