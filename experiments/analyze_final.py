"""Analysis for the final four-arm study (pre-registered in EXPERIMENT_LOG.md).
usage: python analyze_final.py <out_dir>"""
import sys, json
import numpy as np, pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score, roc_auc_score, hamming_loss, accuracy_score
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

R = sys.argv[1]
GRID = np.linspace(0.05, 0.95, 19)
def tune(y, p):
    th = np.full(y.shape[1], 0.5)
    for i in range(y.shape[1]):
        best = 0
        for t in GRID:
            f = f1_score(y[:, i], (p[:, i] >= t).astype(int), zero_division=0)
            if f > best: best, th[i] = f, t
    return th
def apply(p, th, lo=1, hi=3):
    out = np.zeros_like(p, dtype=int)
    for i in range(len(p)):
        L = np.where(p[i] >= th)[0]
        if len(L) < lo: L = np.argsort(p[i])[-lo:]
        if len(L) > hi: L = L[np.argsort(p[i][L])[-hi:]]
        out[i, L] = 1
    return out
def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); adj = np.empty_like(ps); run = 0
    for rank, i in enumerate(o): run = max(run, min(1, (len(ps) - rank) * ps[i])); adj[i] = run
    return adj
def folds(z, r):
    rows = np.where(~z["syn"])[0]
    for a, b in MultilabelStratifiedKFold(4, shuffle=True, random_state=r).split(rows, z["Y_tr"][rows]): yield rows[b]

ARMS = {"baseline": "Baseline", "class_weight": "Class weighting", "gpt4o": "GPT-4o augmentation", "gpt4o_cw": "GPT-4o augmentation + class weighting"}
Z = {a: np.load(f"{R}/{a}.npz", allow_pickle=True) for a in ARMS}
labels = list(Z["baseline"]["labels"]); AUG = list(Z["baseline"]["aug"]); A = np.array([l in AUG for l in labels])
YF = Z["baseline"]["Y_te"]; amb = Z["baseline"]["ambiguous"]
for a in Z: assert list(Z[a]["labels"]) == labels and (Z[a]["Y_te"] == YF).all()
TH = {a: np.mean([tune(z["Y_tr"][v], z["oof"][r][v]) for r in range(5) for v in folds(z, r)], axis=0) for a, z in Z.items()}
PF = {a: apply(Z[a]["test_prob"], TH[a]) for a in Z}

def evaluate(mask_rows, tag):
    Y = YF[mask_rows]; P = {a: PF[a][mask_rows] for a in Z}; prob = {a: Z[a]["test_prob"][mask_rows] for a in Z}
    def pooled(p, m): return ((Y[:, m] == 1) & (p[:, m] == 1)).sum(), (Y[:, m] == 1).sum()
    def comp(new, ref, m):
        pos = Y[:, m] == 1; x = (P[new][:, m] == 1) & pos; y = (P[ref][:, m] == 1) & pos
        g, l = int((x & ~y).sum()), int((~x & y).sum()); rng = np.random.default_rng(42); d = []
        for _ in range(2000):
            i = rng.integers(0, len(Y), len(Y)); d.append((x[i].sum() - y[i].sum()) / pos[i].sum())
        return dict(new=ARMS[new], ref=ARMS[ref], correct_new=int(x.sum()), correct_ref=int(y.sum()), n=int(pos.sum()),
                    sens_new=x.sum() / pos.sum(), sens_ref=y.sum() / pos.sum(), diff=(x.sum() - y.sum()) / pos.sum(),
                    ci_lo=np.percentile(d, 2.5), ci_hi=np.percentile(d, 97.5), gained=g, lost=l,
                    mcnemar_p=binomtest(g, g + l).pvalue if g + l else 1.0)
    prim = pd.DataFrame([comp("gpt4o_cw", "baseline", A)])
    common = pd.DataFrame([comp("gpt4o_cw", "baseline", ~A)]); common["meets_minus2"] = common.ci_lo > -0.02
    sec_pairs = [("gpt4o", "baseline"), ("class_weight", "baseline"), ("gpt4o_cw", "class_weight")]
    sec = pd.DataFrame([comp(n, r, A) for n, r in sec_pairs]); sec["holm_p"] = holm(sec.mcnemar_p)
    secc = pd.DataFrame([comp(n, r, ~A) for n, r in sec_pairs]); secc["holm_p"] = holm(secc.mcnemar_p)
    rows = []
    for a in Z:
        p = P[a]; tp = (Y == 1) & (p == 1)
        ca, na = pooled(p, A); cc, nc = pooled(p, ~A)
        rows.append(dict(arm=ARMS[a], rare_correct=int(ca), rare_n=int(na), common_correct=int(cc), common_n=int(nc),
                         correct_advised=int(tp.sum()), advised=int(p.sum()), precision=tp.sum() / p.sum(),
                         exact_match=int((p == Y).all(1).sum()), all_ref_advised=int((((Y == 1) & (p == 0)).sum(1) == 0).sum()),
                         top3=int(sum(len(set(np.argsort(q)[-3:]) & set(np.where(y)[0])) > 0 for q, y in zip(prob[a], Y))),
                         n_cases=len(Y), macro_f1=f1_score(Y, p, average="macro", zero_division=0),
                         micro_f1=f1_score(Y, p, average="micro", zero_division=0),
                         weighted_f1=f1_score(Y, p, average="weighted", zero_division=0),
                         macro_auc=roc_auc_score(Y[:, Y.sum(0) > 0], prob[a][:, Y.sum(0) > 0], average="macro"),
                         hamming=hamming_loss(Y, p), advised_per_case=p.sum() / len(Y)))
    allarms = pd.DataFrame(rows)
    pp = []
    for j, L in enumerate(labels):
        pos = Y[:, j] == 1; r_ = dict(protocol=L, augmented=L in AUG, n_test=int(pos.sum()))
        for a in Z: r_[ARMS[a]] = int(((P[a][:, j] == 1) & pos).sum())
        g = int((pos & (P["gpt4o_cw"][:, j] == 1) & (P["baseline"][:, j] == 0)).sum())
        l = int((pos & (P["gpt4o_cw"][:, j] == 0) & (P["baseline"][:, j] == 1)).sum())
        r_.update(gained=g, lost=l, mcnemar_p=binomtest(g, g + l).pvalue if g + l else 1.0); pp.append(r_)
    pp = pd.DataFrame(pp); pp["holm_p"] = holm(pp.mcnemar_p); pp = pp.sort_values(["augmented", "n_test"], ascending=[True, False])
    pd.set_option("display.width", 260)
    print(f"\n################ {tag}: {len(Y)} cases ################")
    print("PRIMARY (rare protocols, GPT-4o+CW vs baseline)"); print(prim.round(4).to_string(index=False))
    print("COMMON PROTOCOLS (no-cost check, margin -2 points)"); print(common.round(4).to_string(index=False))
    print("SECONDARY COMPARISONS, rare protocols (Holm)"); print(sec.round(4).to_string(index=False))
    print("SECONDARY COMPARISONS, common protocols (Holm)"); print(secc.round(4).to_string(index=False))
    print("ALL ARMS"); print(allarms.round(3).to_string(index=False))
    print("PER PROTOCOL (correct of n_test)"); print(pp.round(3).to_string(index=False))
    return dict(primary=prim, common=common, secondary_rare=sec, secondary_common=secc, all_arms=allarms, per_protocol=pp)

res_p = evaluate(~amb, "PRIMARY TEST SET (ambiguous excluded)")
res_s = evaluate(np.ones(len(amb), bool), "SECONDARY TEST SET (ambiguous included)")

cv = []
for a, z in Z.items():
    oof, Yt, syn = z["oof"], z["Y_tr"], z["syn"]; tp = np.zeros(len(labels)); fn = tp.copy(); fp = tp.copy()
    for r in range(5):
        fl = list(folds(z, r))
        for k, v in enumerate(fl):
            t_ = np.mean([tune(Yt[w], oof[r][w]) for i, w in enumerate(fl) if i != k], axis=0); yp = apply(oof[r][v], t_); y = Yt[v]
            tp += ((y == 1) & (yp == 1)).sum(0); fn += ((y == 1) & (yp == 0)).sum(0); fp += ((y == 0) & (yp == 1)).sum(0)
    cv.append(dict(arm=ARMS[a], rare_sens=tp[A].sum() / (tp[A] + fn[A]).sum(), common_sens=tp[~A].sum() / (tp[~A] + fn[~A]).sum(),
                   precision=tp.sum() / (tp + fp).sum()))
cv = pd.DataFrame(cv); print("\nCV (real training rows, pooled 5x4 folds)"); print(cv.round(4).to_string(index=False))

with pd.ExcelWriter(f"{R}/final_results.xlsx") as w:
    pd.DataFrame([json.load(open(f"{R}/data_summary.json"))]).T.to_excel(w, sheet_name="data_summary")
    for tag, res in [("P", res_p), ("S", res_s)]:
        for k, v in res.items(): v.to_excel(w, sheet_name=f"{tag}_{k}"[:31], index=False)
    cv.to_excel(w, sheet_name="cv", index=False)
print("saved", f"{R}/final_results.xlsx")
