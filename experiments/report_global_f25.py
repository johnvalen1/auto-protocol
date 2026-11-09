"""Final operating rule: per-protocol thresholds maximise F1 (averaged over 20 CV folds),
then ONE lowering factor per model maximises micro F2.5 over all suggestions in
training-data CV (each fold's thresholds from the other folds). Baseline vs GPT-4o.
Test set: 984 studies (V7 test set, ambiguous indications included, Neck NC removed);
EXCLUDE_AMBIG_TEST=1 evaluates on the 956 studies without ambiguous indications.
usage: python report_global_f25.py <final_out_dir>"""
import sys, json
import numpy as np, pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score, roc_auc_score, hamming_loss
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

R = sys.argv[1]; BETA = 2.5
GRID = np.linspace(0.05, 0.95, 19)
def fb(tp, fp, fn, b): return (1 + b * b) * tp / max((1 + b * b) * tp + b * b * fn + fp, 1)
def tune_f1(Y, P):
    th = np.full(Y.shape[1], 0.5)
    for j in range(Y.shape[1]):
        y = Y[:, j].astype(bool); pr = P[:, j][:, None] >= GRID[None, :]
        tp = (pr & y[:, None]).sum(0); fp = (pr & ~y[:, None]).sum(0); fn = (~pr & y[:, None]).sum(0)
        s = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0)
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
def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); adj = np.empty_like(ps); run = 0
    for k, i in enumerate(o): run = max(run, min(1, (len(ps) - k) * ps[i])); adj[i] = run
    return adj
mcn = lambda g, l: binomtest(g, g + l).pvalue if g + l else 1.0

ARMS = {"baseline": "Baseline", "gpt4o": "GPT-4o augmentation"}
Z = {a: np.load(f"{R}/{a}.npz", allow_pickle=True) for a in ARMS}
labels = list(Z["baseline"]["labels"]); AUG = list(Z["baseline"]["aug"]); A = np.array([l in AUG for l in labels])
FACTOR, TH = {}, {}
for a, z in Z.items():
    Yt, oof = z["Y_tr"], z["oof"]; cache = []
    for r in range(oof.shape[0]):
        fl = list(folds(z, r))
        for k, v in enumerate(fl):
            cache.append((oof[r][v], Yt[v], np.mean([tune_f1(Yt[w], oof[r][w]) for i, w in enumerate(fl) if i != k], axis=0)))
    best = None
    for s in np.round(np.arange(0.20, 1.001, 0.01), 2):
        tp = fp = fn = 0
        for p, y, th in cache:
            yp = apply(p, th * s); tp += ((y == 1) & (yp == 1)).sum(); fp += ((y == 0) & (yp == 1)).sum(); fn += ((y == 1) & (yp == 0)).sum()
        f = fb(tp, fp, fn, BETA)
        if best is None or f > best[0]: best = (f, s)
    FACTOR[a] = best[1]
    TH[a] = np.mean([tune_f1(Yt[v], oof[r][v]) for r in range(oof.shape[0]) for v in folds(z, r)], axis=0) * FACTOR[a]
print("factors chosen on CV (micro F2.5):", FACTOR)

import os
KEEP = ~Z["baseline"]["ambiguous"] if os.environ.get("EXCLUDE_AMBIG_TEST") == "1" else np.ones(len(Z["baseline"]["ambiguous"]), bool)
TAG = "_ambiguous_excluded" if os.environ.get("EXCLUDE_AMBIG_TEST") == "1" else ""
Y = Z["baseline"]["Y_te"][KEEP]  # 984 test studies, or 956 with EXCLUDE_AMBIG_TEST=1
P = {a: apply(Z[a]["test_prob"], TH[a])[KEEP] for a in Z}; PR = {a: Z[a]["test_prob"][KEEP] for a in Z}
print("test studies:", len(Y))
t3 = lambda pr: np.array([len(set(np.argsort(q)[-3:]) & set(np.where(t)[0])) > 0 for q, t in zip(pr, Y)])
auc = lambda y, p: roc_auc_score(y[:, y.sum(0) > 0], p[:, y.sum(0) > 0], average="macro")
def boot(fn, PA, PB, n=2000):
    rng = np.random.default_rng(42); d = []
    for _ in range(n):
        i = rng.integers(0, len(Y), len(Y)); d.append(fn(Y[i], PA[i]) - fn(Y[i], PB[i]))
    d = np.array(d); return fn(Y, PA) - fn(Y, PB), np.percentile(d, 2.5), np.percentile(d, 97.5), min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
pg, pb = P["gpt4o"], P["baseline"]
rows = []
def add(name, b, g, diff, lo, hi, gained, lost, p, test): rows.append(dict(metric=name, baseline=b, gpt4o=g, difference=diff, ci_lo=lo, ci_hi=hi, gained=gained, lost=lost, p=p, test=test))
for name, m in [("Rare-protocol sensitivity", A), ("Common-protocol sensitivity", ~A), ("Overall sensitivity", np.ones(len(labels), bool))]:
    f = lambda y, p, mm=m: ((y[:, mm] == 1) & (p[:, mm] == 1)).sum() / (y[:, mm] == 1).sum()
    pos = Y[:, m] == 1; x = (pg[:, m] == 1) & pos; yy = (pb[:, m] == 1) & pos; g, l = int((x & ~yy).sum()), int((~x & yy).sum())
    d, lo, hi, _ = boot(f, pg, pb); add(name, f(Y, pb), f(Y, pg), d, lo, hi, g, l, mcn(g, l), "Exact McNemar (case-protocol pairs)")
for name, f in [("All reference protocols suggested", lambda y, p: (((y == 1) & (p == 0)).sum(1) == 0)), ("Exact match", lambda y, p: (p == y).all(1))]:
    a_, b_ = f(Y, pg), f(Y, pb); g, l = int((a_ & ~b_).sum()), int((~a_ & b_).sum()); d, lo, hi, _ = boot(lambda y, p, ff=f: ff(y, p).mean(), pg, pb)
    add(name, b_.mean(), a_.mean(), d, lo, hi, g, l, mcn(g, l), "Exact McNemar (cases)")
a_, b_ = t3(PR["gpt4o"]), t3(PR["baseline"]); g, l = int((a_ & ~b_).sum()), int((~a_ & b_).sum())
add("Top-3 accuracy", b_.mean(), a_.mean(), a_.mean() - b_.mean(), np.nan, np.nan, g, l, mcn(g, l), "Exact McNemar (cases)")
for name, f in [("Precision", lambda y, p: ((y == 1) & (p == 1)).sum() / p.sum()), ("Micro F1", lambda y, p: f1_score(y, p, average="micro", zero_division=0)),
                ("Macro F1", lambda y, p: f1_score(y, p, average="macro", zero_division=0)), ("Weighted F1", lambda y, p: f1_score(y, p, average="weighted", zero_division=0)),
                ("Hamming loss", lambda y, p: hamming_loss(y, p))]:
    d, lo, hi, p_ = boot(f, pg, pb); add(name, f(Y, pb), f(Y, pg), d, lo, hi, None, None, p_, "Case bootstrap, 2,000 resamples")
d, lo, hi, p_ = boot(auc, PR["gpt4o"], PR["baseline"]); add("Macro AUC", auc(Y, PR["baseline"]), auc(Y, PR["gpt4o"]), d, lo, hi, None, None, p_, "Case bootstrap, 2,000 resamples")
add("Suggestions per case", pb.sum() / len(Y), pg.sum() / len(Y), None, None, None, None, None, None, "")
G = pd.DataFrame(rows)
pp = []
for j, L in enumerate(labels):
    pos = Y[:, j] == 1; g = int((pos & (pg[:, j] == 1) & (pb[:, j] == 0)).sum()); l = int((pos & (pg[:, j] == 0) & (pb[:, j] == 1)).sum())
    r = dict(protocol=L, augmented=L in AUG, test_cases=int(pos.sum()))
    for a, nm in ARMS.items():
        tp = int(((P[a][:, j] == 1) & pos).sum()); adv = int(P[a][:, j].sum())
        r[f"{nm} correct"] = tp; r[f"{nm} sensitivity"] = tp / pos.sum(); r[f"{nm} suggested"] = adv
        r[f"{nm} precision"] = tp / adv if adv else np.nan; r[f"{nm} F1"] = f1_score(Y[:, j], P[a][:, j], zero_division=0)
    r.update(gained=g, lost=l, mcnemar_p=mcn(g, l)); pp.append(r)
PP = pd.DataFrame(pp); PP["holm_p"] = holm(PP.mcnemar_p); PP = PP.sort_values(["augmented", "test_cases"], ascending=[False, False])
pd.set_option("display.width", 260)
print(G.round(4).to_string(index=False))
print(PP[["protocol", "test_cases", "Baseline correct", "GPT-4o augmentation correct", "Baseline precision", "GPT-4o augmentation precision", "gained", "lost", "mcnemar_p", "holm_p"]].round(3).to_string(index=False))
with pd.ExcelWriter(f"{R}/final_global_f25_results{TAG}.xlsx") as w:
    pd.DataFrame({"Item": ["Rule", "Factors (CV)", "Test set", "Tests"], "Detail": [
        "Per-protocol thresholds maximise F1 (mean of 20 CV folds); one factor per model, chosen in training-data CV to maximise micro F2.5, multiplies all thresholds; 1-3 suggestions per case",
        json.dumps(FACTOR), f"{len(Y)} V7 test studies (Neck NC removed){', ambiguous indications excluded' if TAG else ', ambiguous indications included'}",
        "Exact McNemar for sensitivities and case-level metrics; case bootstrap (2,000 resamples, seed 42) for precision, F1, AUC, Hamming; Holm across 20 protocols"]}).to_excel(w, sheet_name="About", index=False)
    G.to_excel(w, sheet_name="Global", index=False); PP.to_excel(w, sheet_name="Per protocol", index=False)
json.dump(dict(factors=FACTOR, global_=G.to_dict("records"), perprot=PP.to_dict("records")), open(f"{R}/final_global_f25{TAG}.json", "w"), default=float)
print("saved", f"{R}/final_global_f25_results{TAG}.xlsx")
