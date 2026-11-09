"""Baseline vs GPT-4o augmentation with per-protocol thresholds tuned to maximise F2.5
(chosen by the authors on 2026-11-09 after the CV-only check). No retraining: same models.
Thresholds: F2.5-optimal per protocol on each of the 20 CV folds (real rows), averaged.
Evaluated on the test set with ambiguous studies included (984) and excluded (956).
usage: python report_f25.py <final_out_dir>"""
import sys
import numpy as np, pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score, roc_auc_score, hamming_loss
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

R = sys.argv[1]; BETA = 2.5
GRID = np.linspace(0.05, 0.95, 19)
def fbeta(y, pred, b):
    tp = (pred & y[:, None]).sum(0); fp = (pred & ~y[:, None]).sum(0); fn = (~pred & y[:, None]).sum(0)
    den = (1 + b * b) * tp + b * b * fn + fp
    return np.where(den > 0, (1 + b * b) * tp / np.maximum(den, 1), 0)
def tune(Y, P, b):
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
def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); adj = np.empty_like(ps); run = 0
    for rank, i in enumerate(o): run = max(run, min(1, (len(ps) - rank) * ps[i])); adj[i] = run
    return adj
mcn = lambda g, l: binomtest(g, g + l).pvalue if g + l else 1.0

ARMS = {"baseline": "Baseline", "gpt4o": "GPT-4o augmentation"}
Z = {a: np.load(f"{R}/{a}.npz", allow_pickle=True) for a in ARMS}
labels = list(Z["baseline"]["labels"]); AUG = list(Z["baseline"]["aug"]); A = np.array([l in AUG for l in labels])
TH = {a: np.mean([tune(z["Y_tr"][v], z["oof"][r][v], BETA) for r in range(5) for v in folds(z, r)], axis=0) for a, z in Z.items()}
YF = Z["baseline"]["Y_te"]; amb = Z["baseline"]["ambiguous"]

def boot(Y, Pn, Pr, fn, n=2000):
    rng = np.random.default_rng(42); d = []
    for _ in range(n):
        i = rng.integers(0, len(Y), len(Y))
        try: d.append(fn(Y[i], Pn[i]) - fn(Y[i], Pr[i]))
        except ValueError: pass
    d = np.array(d); return fn(Y, Pn) - fn(Y, Pr), np.percentile(d, 2.5), np.percentile(d, 97.5), min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))

G, T, PP = [], [], []
for tag, mask in [("Ambiguous included (984)", np.ones(len(amb), bool)), ("Ambiguous excluded (956)", ~amb)]:
    Y = YF[mask]; P = {a: apply(Z[a]["test_prob"], TH[a])[mask] for a in Z}; PR = {a: Z[a]["test_prob"][mask] for a in Z}
    ok = Y.sum(0) > 0; auc = lambda Yb, Pb: roc_auc_score(Yb[:, Yb.sum(0) > 0], Pb[:, Yb.sum(0) > 0], average="macro")
    caseok = {"Exact match": lambda Y_, P_: (P_ == Y_).all(1), "All reference protocols suggested": lambda Y_, P_: (((Y_ == 1) & (P_ == 0)).sum(1) == 0)}
    top3 = lambda Y_, pr: np.array([len(set(np.argsort(q)[-3:]) & set(np.where(t)[0])) > 0 for q, t in zip(pr, Y_)])
    for a in Z:
        p = P[a]; tp = (Y == 1) & (p == 1)
        G.append({"Test set": tag, "Model": ARMS[a],
                  "Rare sensitivity": tp[:, A].sum() / (Y[:, A] == 1).sum(), "Rare correct": int(tp[:, A].sum()), "Rare labels": int((Y[:, A] == 1).sum()),
                  "Common sensitivity": tp[:, ~A].sum() / (Y[:, ~A] == 1).sum(), "Common correct": int(tp[:, ~A].sum()), "Common labels": int((Y[:, ~A] == 1).sum()),
                  "Overall sensitivity": tp.sum() / Y.sum(), "Precision": tp.sum() / p.sum(), "Correct suggestions": int(tp.sum()), "Total suggestions": int(p.sum()),
                  "Exact match": caseok["Exact match"](Y, p).mean(), "Exact match (cases)": int(caseok["Exact match"](Y, p).sum()),
                  "All reference suggested": caseok["All reference protocols suggested"](Y, p).mean(),
                  "All reference suggested (cases)": int(caseok["All reference protocols suggested"](Y, p).sum()),
                  "Top-3 accuracy": top3(Y, PR[a]).mean(), "Micro F1": f1_score(Y, p, average="micro", zero_division=0),
                  "Macro F1": f1_score(Y, p, average="macro", zero_division=0), "Weighted F1": f1_score(Y, p, average="weighted", zero_division=0),
                  "Macro AUC": auc(Y, PR[a]), "Hamming loss": hamming_loss(Y, p), "Suggestions per case": p.sum() / len(Y), "Cases": len(Y)})
    Pn, Pr = P["gpt4o"], P["baseline"]
    for name, m in [("Rare sensitivity", A), ("Common sensitivity", ~A), ("Overall sensitivity", np.ones(len(labels), bool))]:
        pos = Y[:, m] == 1; x = (Pn[:, m] == 1) & pos; y = (Pr[:, m] == 1) & pos; g, l = int((x & ~y).sum()), int((~x & y).sum())
        o, lo, hi, _ = boot(Y, Pn, Pr, lambda Yb, Pb, mm=m: ((Yb[:, mm] == 1) & (Pb[:, mm] == 1)).sum() / (Yb[:, mm] == 1).sum())
        T.append(dict(test_set=tag, metric=name, difference=o, ci_lo=lo, ci_hi=hi, gained=g, lost=l, test="Exact McNemar (case-protocol pairs)", p=mcn(g, l)))
    for name, fn in caseok.items():
        a_, b_ = fn(Y, Pn), fn(Y, Pr); g, l = int((a_ & ~b_).sum()), int((~a_ & b_).sum()); o, lo, hi, _ = boot(Y, Pn, Pr, lambda Yb, Pb, f=fn: f(Yb, Pb).mean())
        T.append(dict(test_set=tag, metric=name, difference=o, ci_lo=lo, ci_hi=hi, gained=g, lost=l, test="Exact McNemar (cases)", p=mcn(g, l)))
    a_, b_ = top3(Y, PR["gpt4o"]), top3(Y, PR["baseline"]); g, l = int((a_ & ~b_).sum()), int((~a_ & b_).sum())
    T.append(dict(test_set=tag, metric="Top-3 accuracy", difference=a_.mean() - b_.mean(), ci_lo=np.nan, ci_hi=np.nan, gained=g, lost=l, test="Exact McNemar (cases)", p=mcn(g, l)))
    for name, fn in [("Precision", lambda Y_, P_: ((Y_ == 1) & (P_ == 1)).sum() / P_.sum()), ("Micro F1", lambda Y_, P_: f1_score(Y_, P_, average="micro", zero_division=0)),
                     ("Macro F1", lambda Y_, P_: f1_score(Y_, P_, average="macro", zero_division=0)), ("Weighted F1", lambda Y_, P_: f1_score(Y_, P_, average="weighted", zero_division=0)),
                     ("Hamming loss", lambda Y_, P_: hamming_loss(Y_, P_))]:
        o, lo, hi, p_ = boot(Y, Pn, Pr, fn); T.append(dict(test_set=tag, metric=name, difference=o, ci_lo=lo, ci_hi=hi, gained=None, lost=None, test="Case bootstrap, 2,000 resamples", p=p_))
    o, lo, hi, p_ = boot(Y, PR["gpt4o"], PR["baseline"], auc)
    T.append(dict(test_set=tag, metric="Macro AUC", difference=o, ci_lo=lo, ci_hi=hi, gained=None, lost=None, test="Case bootstrap, 2,000 resamples", p=p_))
    rows = []
    for j, L in enumerate(labels):
        pos = Y[:, j] == 1; r = {"Test set": tag, "Protocol": L, "Augmented": L in AUG, "Test cases": int(pos.sum())}
        for a in Z:
            tpj = int(((P[a][:, j] == 1) & pos).sum()); adv = int(P[a][:, j].sum())
            r[f"{ARMS[a]} correct"] = tpj; r[f"{ARMS[a]} sensitivity"] = tpj / pos.sum() if pos.sum() else np.nan
            r[f"{ARMS[a]} suggested"] = adv; r[f"{ARMS[a]} precision"] = tpj / adv if adv else np.nan
            r[f"{ARMS[a]} F1"] = f1_score(Y[:, j], P[a][:, j], zero_division=0)
        g = int((pos & (Pn[:, j] == 1) & (Pr[:, j] == 0)).sum()); l = int((pos & (Pn[:, j] == 0) & (Pr[:, j] == 1)).sum())
        r.update({"Gained": g, "Lost": l, "McNemar p": mcn(g, l)}); rows.append(r)
    rows = pd.DataFrame(rows); rows["Holm p (20 protocols)"] = holm(rows["McNemar p"]); PP.append(rows)
G = pd.DataFrame(G); T = pd.DataFrame(T); PP = pd.concat(PP)
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40)
print("thresholds mean:", {a: round(TH[a].mean(), 3) for a in TH})
print(G.set_index(["Test set", "Model"]).T.round(4).to_string())
print(T.round(4).to_string(index=False))
x = PP[PP["Test set"].str.startswith("Ambiguous included")]
print(x[["Protocol", "Augmented", "Test cases", "Baseline correct", "GPT-4o augmentation correct", "Baseline precision", "GPT-4o augmentation precision", "Gained", "Lost", "McNemar p", "Holm p (20 protocols)"]].sort_values(["Augmented", "Test cases"], ascending=[False, False]).round(3).to_string(index=False))
with pd.ExcelWriter(f"{R}/f25_baseline_vs_gpt4o.xlsx") as w:
    pd.DataFrame({"Item": ["Comparison", "Thresholds", "Training", "Test sets", "Tests", "Status"],
                  "Detail": ["Baseline vs GPT-4o augmentation (no class weighting)",
                             "Per protocol, maximising F2.5 (a missed protocol weighted 6.25x an extra suggestion), grid 0.05-0.95, averaged over 20 CV folds of real training studies; 1-3 suggestions per case",
                             "Unchanged models: real training studies without ambiguous indications; GPT-4o arm adds 701 synthetic indications",
                             "Main: 984 studies with ambiguous indications included. Also: 956 with them excluded",
                             "Exact McNemar for sensitivities and case-level metrics; case bootstrap (2,000 resamples, seed 42) for precision, F1, AUC, Hamming loss; Holm across 20 protocols",
                             "Threshold metric chosen by the authors after the earlier analyses (post hoc)"]}).to_excel(w, sheet_name="About", index=False)
    G.to_excel(w, sheet_name="Global metrics", index=False); T.to_excel(w, sheet_name="Global tests", index=False)
    PP.to_excel(w, sheet_name="Per-protocol", index=False)
print("saved", f"{R}/f25_baseline_vs_gpt4o.xlsx")
