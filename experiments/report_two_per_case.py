"""Full metrics and tests for the secondary 'about 2 suggestions per case' analysis
(CV-calibrated threshold scaling, see EXPERIMENT_LOG.md). No retraining.
usage: python report_two_per_case.py <final_out_dir>"""
import sys
import numpy as np, pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score, roc_auc_score, hamming_loss
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

R = sys.argv[1]
src = open(__file__.replace("report_two_per_case.py", "analyze_operating_points.py")).read()
import os
here = os.path.dirname(os.path.abspath(__file__))
block = src[src.index("R = sys.argv[1]"):src.index("TARGETS =")]
block = block.replace('__file__.replace("analyze_operating_points.py", "analyze_final.py")', f'"{here}/analyze_final.py"')
exec(block)  # Z, labels, AUG, A, YF, amb, TH, topk, cv_scale
ARMS = {"baseline": "Baseline", "class_weight": "Class weighting", "gpt4o": "GPT-4o augmentation", "gpt4o_cw": "GPT-4o + class weighting"}
SCALE = {a: cv_scale(a, 2.0) for a in Z}
PRED = {a: apply(Z[a]["test_prob"], TH[a] * SCALE[a]) for a in Z}
PROB = {a: Z[a]["test_prob"] for a in Z}
mcn = lambda g, l: binomtest(g, g + l).pvalue if g + l else 1.0

def global_row(Y, P, prob):
    tp = (Y == 1) & (P == 1); ok = Y.sum(0) > 0
    top3 = sum(len(set(np.argsort(q)[-3:]) & set(np.where(y)[0])) > 0 for q, y in zip(prob, Y))
    return {"Rare sensitivity": tp[:, A].sum() / (Y[:, A] == 1).sum(), "Rare correct": int(tp[:, A].sum()), "Rare labels": int((Y[:, A] == 1).sum()),
            "Common sensitivity": tp[:, ~A].sum() / (Y[:, ~A] == 1).sum(), "Common correct": int(tp[:, ~A].sum()), "Common labels": int((Y[:, ~A] == 1).sum()),
            "Overall sensitivity (micro recall)": tp.sum() / Y.sum(), "Precision (micro)": tp.sum() / P.sum(),
            "Correct suggestions": int(tp.sum()), "Total suggestions": int(P.sum()),
            "Micro F1": f1_score(Y, P, average="micro", zero_division=0), "Macro F1": f1_score(Y, P, average="macro", zero_division=0),
            "Weighted F1": f1_score(Y, P, average="weighted", zero_division=0),
            "Exact match (cases)": int((P == Y).all(1).sum()), "Exact match rate": (P == Y).all(1).mean(),
            "All reference protocols suggested (cases)": int((((Y == 1) & (P == 0)).sum(1) == 0).sum()),
            "All reference suggested rate": (((Y == 1) & (P == 0)).sum(1) == 0).mean(),
            "Top-3 (threshold-independent)": top3 / len(Y), "Macro AUC (threshold-independent)": roc_auc_score(Y[:, ok], prob[:, ok], average="macro"),
            "Hamming loss": hamming_loss(Y, P), "Suggestions per case": P.sum() / len(Y), "Cases": len(Y)}

def boot(Y, Pn, Pr, fn, n=2000):
    rng = np.random.default_rng(42); d = []
    for _ in range(n):
        i = rng.integers(0, len(Y), len(Y)); d.append(fn(Y[i], Pn[i]) - fn(Y[i], Pr[i]))
    d = np.array(d); obs = fn(Y, Pn) - fn(Y, Pr)
    p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))  # percentile bootstrap, two-sided
    return obs, np.percentile(d, 2.5), np.percentile(d, 97.5), p

def tests(Y, Pn, Pr):
    out = []
    for grp, m in [("Rare-protocol sensitivity", A), ("Common-protocol sensitivity", ~A)]:
        pos = Y[:, m] == 1; x = (Pn[:, m] == 1) & pos; y = (Pr[:, m] == 1) & pos
        g, l = int((x & ~y).sum()), int((~x & y).sum())
        def f(Yb, Pb, mm=m): return ((Yb[:, mm] == 1) & (Pb[:, mm] == 1)).sum() / max((Yb[:, mm] == 1).sum(), 1)
        o, lo, hi, _ = boot(Y, Pn, Pr, f)
        out.append(dict(metric=grp, difference=o, ci_lo=lo, ci_hi=hi, gained=g, lost=l, test="Exact McNemar (case-protocol pairs)", p=mcn(g, l)))
    for name, case_ok in [("Exact match", lambda Y_, P_: (P_ == Y_).all(1)), ("All reference protocols suggested", lambda Y_, P_: (((Y_ == 1) & (P_ == 0)).sum(1) == 0))]:
        a, b = case_ok(Y, Pn), case_ok(Y, Pr); g, l = int((a & ~b).sum()), int((~a & b).sum())
        o, lo, hi, _ = boot(Y, Pn, Pr, lambda Y_, P_: case_ok(Y_, P_).mean())
        out.append(dict(metric=name, difference=o, ci_lo=lo, ci_hi=hi, gained=g, lost=l, test="Exact McNemar (cases)", p=mcn(g, l)))
    for name, fn in [("Precision (micro)", lambda Y_, P_: ((Y_ == 1) & (P_ == 1)).sum() / P_.sum()),
                     ("Micro F1", lambda Y_, P_: f1_score(Y_, P_, average="micro", zero_division=0)),
                     ("Macro F1", lambda Y_, P_: f1_score(Y_, P_, average="macro", zero_division=0)),
                     ("Weighted F1", lambda Y_, P_: f1_score(Y_, P_, average="weighted", zero_division=0)),
                     ("Hamming loss", lambda Y_, P_: hamming_loss(Y_, P_))]:
        o, lo, hi, p = boot(Y, Pn, Pr, fn)
        out.append(dict(metric=name, difference=o, ci_lo=lo, ci_hi=hi, gained=None, lost=None, test="Case bootstrap, 2,000 resamples", p=p))
    return out

def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); adj = np.empty_like(ps); run = 0
    for rank, i in enumerate(o): run = max(run, min(1, (len(ps) - rank) * ps[i])); adj[i] = run
    return adj

G, T, PP, PT = [], [], [], []
for tag, mask in [("Primary (ambiguous excluded)", ~amb), ("Secondary (ambiguous included)", np.ones(len(amb), bool))]:
    Y = YF[mask]; P = {a: PRED[a][mask] for a in Z}; pr = {a: PROB[a][mask] for a in Z}
    for a in Z: G.append({"Test set": tag, "Model": ARMS[a], "Threshold factor (CV)": SCALE[a], **global_row(Y, P[a], pr[a])})
    rows = []
    for a in ["class_weight", "gpt4o", "gpt4o_cw"]:
        for t in tests(Y, P[a], P["baseline"]): rows.append({"Test set": tag, "Comparison": f"{ARMS[a]} vs Baseline", **t})
    for t in tests(Y, P["gpt4o_cw"], P["class_weight"]): rows.append({"Test set": tag, "Comparison": "GPT-4o + class weighting vs Class weighting", **t})
    t = pd.DataFrame(rows)
    t["holm_p (within metric, 4 comparisons)"] = t.groupby("metric").p.transform(lambda s: holm(s.values))
    T.append(t)
    for j, L in enumerate(labels):
        pos = Y[:, j] == 1; r = {"Test set": tag, "Protocol": L, "Augmented": L in AUG, "Test cases": int(pos.sum())}
        for a in Z:
            tp = int(((P[a][:, j] == 1) & pos).sum()); adv = int(P[a][:, j].sum())
            r[f"{ARMS[a]}: correct"] = tp; r[f"{ARMS[a]}: sensitivity"] = tp / pos.sum() if pos.sum() else np.nan
            r[f"{ARMS[a]}: suggested"] = adv; r[f"{ARMS[a]}: precision"] = tp / adv if adv else np.nan
            r[f"{ARMS[a]}: F1"] = f1_score(Y[:, j], P[a][:, j], zero_division=0)
        PP.append(r)
        for a in ["gpt4o", "gpt4o_cw", "class_weight"]:
            g = int((pos & (P[a][:, j] == 1) & (P["baseline"][:, j] == 0)).sum()); l = int((pos & (P[a][:, j] == 0) & (P["baseline"][:, j] == 1)).sum())
            PT.append({"Test set": tag, "Comparison": f"{ARMS[a]} vs Baseline", "Protocol": L, "Augmented": L in AUG, "Test cases": int(pos.sum()),
                       "Baseline correct": int(((P["baseline"][:, j] == 1) & pos).sum()), "Model correct": int(((P[a][:, j] == 1) & pos).sum()),
                       "Gained": g, "Lost": l, "McNemar p": mcn(g, l)})
G = pd.DataFrame(G); T = pd.concat(T); PP = pd.DataFrame(PP); PT = pd.DataFrame(PT)
PT["Holm p (20 protocols)"] = PT.groupby(["Test set", "Comparison"])["McNemar p"].transform(lambda s: holm(s.values))
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40)
for tag in G["Test set"].unique():
    print(f"\n######## {tag}\nGLOBAL"); print(G[G["Test set"] == tag].drop(columns="Test set").set_index("Model").T.to_string())
    print("TESTS"); print(T[T["Test set"] == tag].drop(columns="Test set").round(4).to_string(index=False))
print("\nPER-PROTOCOL TESTS (primary)")
x = PT[PT["Test set"].str.startswith("Primary")].drop(columns="Test set")
print(x.round(4).to_string(index=False))
with pd.ExcelWriter(f"{R}/two_per_case_full_results.xlsx") as w:
    pd.DataFrame({"Item": ["Analysis", "Rule", "Models", "Test sets", "Sensitivity tests", "Case-level tests", "Other metric tests", "Multiplicity", "Status"],
                  "Detail": ["Secondary operating-point analysis: about 2 suggested protocols per case",
                             "Each model's protocol thresholds multiplied by one factor chosen on training-data cross-validation so the mean is ~2.0 suggestions per case (1-3 per case)",
                             "Baseline, class weighting, GPT-4o augmentation, GPT-4o + class weighting (no retraining)",
                             "Primary: 956 studies, ambiguous excluded. Secondary: 984 studies, ambiguous included",
                             "Exact McNemar on discordant case-protocol pairs; 95% CI by case bootstrap (2,000 resamples, seed 42)",
                             "Exact McNemar on discordant cases (exact match; all reference protocols suggested)",
                             "Precision, F1 and Hamming loss: percentile case bootstrap, 2,000 resamples; two-sided bootstrap p",
                             "Holm across the 4 comparisons within each metric; Holm across 20 protocols within each comparison",
                             "Post hoc: requested after the pre-specified results were seen"]}).to_excel(w, sheet_name="About", index=False)
    G.to_excel(w, sheet_name="Global metrics", index=False); T.to_excel(w, sheet_name="Global tests", index=False)
    PP.to_excel(w, sheet_name="Per-protocol metrics", index=False); PT.to_excel(w, sheet_name="Per-protocol tests", index=False)
print("saved", f"{R}/two_per_case_full_results.xlsx")
