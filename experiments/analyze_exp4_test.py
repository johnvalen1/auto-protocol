"""Experiment 4: seed check (CV) and Stage 2 test evaluation, as pre-registered.
usage: python analyze_exp4_test.py <runs_dir> seeds   -> CV seed-stability only (no test arrays)
       python analyze_exp4_test.py <runs_dir> test    -> Stage 2 test evaluation"""
import sys, json, glob, os
import numpy as np, pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score, roc_auc_score, hamming_loss, accuracy_score
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

R, MODE = sys.argv[1], sys.argv[2]
here = os.path.dirname(os.path.abspath(__file__))
sel_src = open(f"{here}/analyze_exp4_select.py").read()
block = sel_src[sel_src.index("R = sys.argv[1]"):sel_src.index("rows, sds = {}")]
block = block.replace('__file__.replace("analyze_exp4_select.py", "analyze.py")', f'"{here}/analyze.py"')
exec(block)  # CVOnly, folds, cv_counts, metrics, Z, labels, A
SEL = json.load(open(f"{R}/exp4_selected.json")); S, SF = SEL["selected"], SEL["file"]
pd.set_option("display.width", 250)

if MODE == "seeds":
    out = []
    for name, f in [("baseline", "baseline"), (S, SF)]:
        for seed, path in [(0, f"{R}/{f}.npz"), (10, f"{R}/{f}_seed10.npz"), (20, f"{R}/{f}_seed20.npz")]:
            c = cv_counts(CVOnly(path))
            m = metrics(*[sum(x[i] for x in c) for i in range(3)], np.mean([x[3] for x in c]))
            out.append(dict(config=name, chain_seeds=f"{seed}-{seed+2}", **m))
    t = pd.DataFrame(out); print("=== SEED CHECK, CV ONLY ==="); print(t.round(4).to_string(index=False))
    rng = t.groupby("config")[["S_aug", "S_non", "P_micro", "macro_F1"]].agg(["min", "max"])
    print("\nrange across seeds:"); print((rng * 100).round(2).to_string())
    t.to_csv(f"{R}/exp4_seed_cv.csv", index=False); sys.exit()

# ---------------- Stage 2: test ----------------
an = open(f"{here}/analyze.py").read(); exec(an[an.index("def topk"):an.index("Z = {")])  # topk, global_metrics, holm
def thr(z):
    return np.mean([tune(z["Y_tr"][v], z["oof"][r][v]) for r in range(z["oof"].shape[0]) for v in folds(z, r)], axis=0)
FULL = {n: np.load(f"{R}/{f}.npz", allow_pickle=True) for n, f in CAND.items()}
Y = FULL["baseline"]["Y_te"]
PRED = {n: apply(z["test_prob"], thr(z)) for n, z in FULL.items()}

def pooled(yp, mask): return ((Y[:, mask] == 1) & (yp[:, mask] == 1)).sum() / (Y[:, mask] == 1).sum()
def compare(new, ref, mask, n_boot=2000):
    pos = Y[:, mask] == 1; a = (PRED[new][:, mask] == 1) & pos; b = (PRED[ref][:, mask] == 1) & pos
    g, l = int((a & ~b).sum()), int((~a & b).sum()); rng = np.random.default_rng(42); d = []
    for _ in range(n_boot):
        i = rng.integers(0, len(Y), len(Y)); d.append((a[i].sum() - b[i].sum()) / pos[i].sum())
    return dict(selected=new, comparator=ref, sens_selected=pooled(PRED[new], mask), sens_comparator=pooled(PRED[ref], mask),
                diff=pooled(PRED[new], mask) - pooled(PRED[ref], mask), ci_lo=np.percentile(d, 2.5), ci_hi=np.percentile(d, 97.5),
                gained=g, lost=l, mcnemar_p=binomtest(g, g + l).pvalue if g + l else 1.0)

prim = pd.DataFrame([compare(S, "baseline", A)])
print(f"=== STAGE 2 PRIMARY: {S} vs baseline, pooled sensitivity, 11 augmented protocols (n={int((Y[:, A]==1).sum())} pairs) ===")
print(prim.round(4).to_string(index=False))
others = [n for n in CAND if n not in (S, "baseline")]
for mask, nm in [(A, "augmented"), (~A, "non-augmented")]:
    t = pd.DataFrame([compare(S, o, mask) for o in (["baseline"] + others)]); t["holm_p"] = holm(t.mcnemar_p)
    print(f"\n=== {S} vs every candidate, pooled sensitivity, {nm} protocols (Holm) ==="); print(t.round(4).to_string(index=False))
    t.to_csv(f"{R}/exp4_test_vs_all_{nm}.csv", index=False)
g = pd.DataFrame({n: global_metrics(Y, PRED[n], FULL[n]["test_prob"]) for n in CAND}).T
g.insert(0, "S_aug", [pooled(PRED[n], A) for n in CAND]); g.insert(1, "S_non", [pooled(PRED[n], ~A) for n in CAND])
g.insert(2, "P_micro", [((Y == 1) & (PRED[n] == 1)).sum() / PRED[n].sum() for n in CAND])
print("\n=== TEST, ALL CANDIDATES ==="); print(g.round(3).to_string())
pp = pd.DataFrame({n: ((Y == 1) & (PRED[n] == 1)).sum(0) / (Y == 1).sum(0) for n in CAND}, index=labels)
pp.insert(0, "n_test", (Y == 1).sum(0)); pp = pp.sort_values("n_test", ascending=False)
print("\n=== TEST PER-PROTOCOL SENSITIVITY ==="); print(pp.round(3).to_string())
seed_rows = []
for name, f in [("baseline", "baseline"), (S, SF)]:
    for seed, sfx in [(0, ""), (10, "_seed10"), (20, "_seed20")]:
        z = np.load(f"{R}/{f}{sfx}.npz", allow_pickle=True); yp = apply(z["test_prob"], thr(z))
        seed_rows.append(dict(config=name, seeds=f"{seed}-{seed+2}", S_aug=pooled(yp, A), S_non=pooled(yp, ~A),
                              P_micro=((Y == 1) & (yp == 1)).sum() / yp.sum(), macro_F1=f1_score(Y, yp, average="macro", zero_division=0)))
sr = pd.DataFrame(seed_rows); print("\n=== SEED CHECK, TEST ==="); print(sr.round(4).to_string(index=False))
with pd.ExcelWriter(f"{R}/exp4_results.xlsx") as w:
    pd.read_csv(f"{R}/exp4_stage1_selection.csv").to_excel(w, sheet_name="stage1_cv_selection", index=False)
    pd.read_csv(f"{R}/exp4_seed_cv.csv").to_excel(w, sheet_name="seed_check_cv", index=False)
    prim.to_excel(w, sheet_name="primary_test", index=False)
    for nm in ["augmented", "non-augmented"]:
        pd.read_csv(f"{R}/exp4_test_vs_all_{nm}.csv").to_excel(w, sheet_name=f"vs_all_{nm}"[:31], index=False)
    g.to_excel(w, sheet_name="test_all_candidates"); pp.to_excel(w, sheet_name="test_per_protocol"); sr.to_excel(w, sheet_name="seed_check_test", index=False)
print("\nsaved", f"{R}/exp4_results.xlsx")
