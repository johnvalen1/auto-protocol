"""Secondary operating-point analysis (see EXPERIMENT_LOG.md). No retraining.
usage: python analyze_operating_points.py <final_out_dir>"""
import sys, json
import numpy as np, pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score, hamming_loss
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

R = sys.argv[1]
src = open(__file__.replace("analyze_operating_points.py", "analyze_final.py")).read()
exec(src[src.index("GRID ="):src.index("ARMS =")])  # tune, apply, holm, folds
ARMS = {"baseline": "Baseline", "class_weight": "Class weighting", "gpt4o": "GPT-4o augmentation", "gpt4o_cw": "GPT-4o + class weighting"}
Z = {a: np.load(f"{R}/{a}.npz", allow_pickle=True) for a in ARMS}
labels = list(Z["baseline"]["labels"]); AUG = list(Z["baseline"]["aug"]); A = np.array([l in AUG for l in labels])
YF = Z["baseline"]["Y_te"]; amb = Z["baseline"]["ambiguous"]
TH = {a: np.mean([tune(z["Y_tr"][v], z["oof"][r][v]) for r in range(5) for v in folds(z, r)], axis=0) for a, z in Z.items()}

def topk(p, k):
    out = np.zeros_like(p, dtype=int)
    for i in range(len(p)): out[i, np.argsort(p[i])[-k:]] = 1
    return out

def cv_scale(a, target):
    """Factor s so that mean advised on real out-of-fold rows (all repeats) ~= target. CV only."""
    z = Z[a]; real = ~z["syn"]; oof = z["oof"][:, real]
    best, bs = 9, 1.0
    for s in np.round(np.arange(0.05, 1.501, 0.01), 2):
        m = np.mean([apply(oof[r], TH[a] * s).sum(1).mean() for r in range(oof.shape[0])])
        if abs(m - target) < best: best, bs = abs(m - target), s
    return bs

def rule_pred(a, rule):
    p = Z[a]["test_prob"]
    if rule == "current": return apply(p, TH[a])
    if rule.startswith("top"): return topk(p, int(rule[3:]))
    return apply(p, TH[a] * SCALE[(a, rule)])

TARGETS = [1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75]
SCALE = {(a, f"cv{t}"): cv_scale(a, t) for a in Z for t in TARGETS}
print("CV-calibrated threshold factors:", {f"{a}@{r}": s for (a, r), s in SCALE.items() if r == "cv2.0"})
RULES = ["current", "top1", "top2", "top3"] + [f"cv{t}" for t in TARGETS]

def metrics(Y, P):
    tp = (Y == 1) & (P == 1)
    return dict(rare_correct=int(tp[:, A].sum()), rare_n=int((Y[:, A] == 1).sum()),
                common_correct=int(tp[:, ~A].sum()), common_n=int((Y[:, ~A] == 1).sum()),
                rare_sens=tp[:, A].sum() / (Y[:, A] == 1).sum(), common_sens=tp[:, ~A].sum() / (Y[:, ~A] == 1).sum(),
                precision=tp.sum() / P.sum(), exact_match=int((P == Y).all(1).sum()),
                all_ref_advised=int((((Y == 1) & (P == 0)).sum(1) == 0).sum()), n_cases=len(Y),
                macro_f1=f1_score(Y, P, average="macro", zero_division=0), micro_f1=f1_score(Y, P, average="micro", zero_division=0),
                hamming=hamming_loss(Y, P), advised_per_case=P.sum() / len(Y))

def compare(Y, Pn, Pr, mask):
    pos = Y[:, mask] == 1; x = (Pn[:, mask] == 1) & pos; y = (Pr[:, mask] == 1) & pos
    g, l = int((x & ~y).sum()), int((~x & y).sum()); rng = np.random.default_rng(42); d = []
    for _ in range(2000):
        i = rng.integers(0, len(Y), len(Y)); d.append((x[i].sum() - y[i].sum()) / pos[i].sum())
    return dict(diff=(x.sum() - y.sum()) / pos.sum(), ci_lo=np.percentile(d, 2.5), ci_hi=np.percentile(d, 97.5),
                gained=g, lost=l, p=binomtest(g, g + l).pvalue if g + l else 1.0)

rows, comps, perprot = [], [], []
for tag, mask in [("primary (ambiguous excluded)", ~amb), ("secondary (ambiguous included)", np.ones(len(amb), bool))]:
    Y = YF[mask]
    PR = {(a, r): rule_pred(a, r)[mask] for a in Z for r in RULES}
    for r in RULES:
        for a in Z: rows.append(dict(test_set=tag, rule=r, arm=ARMS[a], **metrics(Y, PR[(a, r)])))
        for a in ["gpt4o", "gpt4o_cw"]:
            for grp, m in [("rare", A), ("common", ~A)]:
                comps.append(dict(test_set=tag, rule=r, comparison=f"{ARMS[a]} vs Baseline", group=grp, **compare(Y, PR[(a, r)], PR[("baseline", r)], m)))
    if tag.startswith("primary"):
        for j, L in enumerate(labels):
            pos = Y[:, j] == 1; r_ = dict(protocol=L, augmented=L in AUG, n_test=int(pos.sum()))
            for rule in ["current", "top2"]:
                for a in Z: r_[f"{ARMS[a]} | {rule}"] = int(((PR[(a, rule)][:, j] == 1) & pos).sum())
            perprot.append(r_)

M = pd.DataFrame(rows); C = pd.DataFrame(comps); PP = pd.DataFrame(perprot).sort_values(["augmented", "n_test"], ascending=[True, False])
pd.set_option("display.width", 280)
for tag in M.test_set.unique():
    print(f"\n######## {tag}")
    print(M[(M.test_set == tag) & M.rule.isin(["current", "top2", "cv2.0"])].drop(columns="test_set").round(3).to_string(index=False))
    print(C[(C.test_set == tag) & C.rule.isin(["current", "top2", "cv2.0"])].drop(columns="test_set").round(4).to_string(index=False))
print("\n######## CURVE (primary): rare / common sensitivity by rule")
cv = M[M.test_set.str.startswith("primary")].pivot_table(index="rule", columns="arm", values=["advised_per_case", "rare_sens", "common_sens", "precision"])
print(cv.round(3).to_string())
print("\n######## PER PROTOCOL (primary)"); print(PP.to_string(index=False))
with pd.ExcelWriter(f"{R}/operating_points.xlsx") as w:
    M.to_excel(w, sheet_name="metrics", index=False); C.to_excel(w, sheet_name="comparisons", index=False)
    PP.to_excel(w, sheet_name="per_protocol_primary", index=False)
    pd.DataFrame([dict(arm=a, rule=r, factor=s) for (a, r), s in SCALE.items()]).to_excel(w, sheet_name="cv_factors", index=False)
print("saved", f"{R}/operating_points.xlsx")
