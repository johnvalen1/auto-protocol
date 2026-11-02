"""
Threshold-tuning experiment for the synthetic-augmentation trade-off.

Same model as autoprotocol.py / autoprotocol_with_augments.py (word+char TF-IDF,
3-chain ClassifierChain ensemble of logistic regressions, per-protocol F1-tuned
thresholds, 1-3 labels per case). Only the cross-validation / threshold step changes.

Schemes
  baseline     : real rows only (as in autoprotocol.py)
  aug_original : CV folds drawn over real + synthetic rows, so thresholds are
                 tuned partly on synthetic validation rows (as in autoprotocol_with_augments.py)
  aug_realval  : CV folds drawn over REAL rows only; every synthetic row is added
                 to every training fold; thresholds tuned on real validation rows only
  aug_realval_cb : aug_realval plus counterbalancing (Experiment 2). Real training rows
                 whose indication matches CUE_REGEX and that carry at least one
                 non-augmented protocol are repeated CUE_K times in each training fold
                 (and in the final training set). Validation rows are never repeated.
                 CUE_K defaults to the value that restores the pre-augmentation share
                 of non-augmented labels among cue-matching training rows.

Out-of-fold probabilities are saved so thresholds can be set either by averaging
per-fold thresholds (repo method) or once on pooled out-of-fold predictions.
The test set is only scored by the final model of each scheme.

usage: python threshold_fix.py <V7 xlsx> <scheme> <out_dir>
"""
import sys, os, time
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import ClassifierChain
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import clone
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

INPUT, SCHEME, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
N_SPLITS, N_REPEATS, N_CHAINS = 4, int(os.environ.get("N_REPEATS", 5)), 3
TEXT, LABEL, SPLIT = "indication_x1", "protocol_y", "split"

df = pd.read_excel(INPUT, sheet_name="data")
df = df[df["exclude_record"].astype(str).str.upper() != "TRUE"].reset_index(drop=True)
df = df[df[SPLIT].isin(["train", "test"])].reset_index(drop=True)
df[LABEL] = df[LABEL].apply(lambda x: [i.strip() for i in str(x).split(",")])
df["synthetic"] = df["original_or_augmented"] == "augmented"
if SCHEME == "baseline":
    df = df[~df["synthetic"]].reset_index(drop=True)

train = df[df[SPLIT] == "train"].reset_index(drop=True)
test = df[df[SPLIT] == "test"].reset_index(drop=True)
mlb = MultiLabelBinarizer()
Y_tr = mlb.fit_transform(train[LABEL]); Y_te = mlb.transform(test[LABEL])
X_tr, X_te = train[TEXT].astype(str), test[TEXT].astype(str)
syn = train["synthetic"].values

model = Pipeline([
    ("features", FeatureUnion([
        ("word_tfidf", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=20000, stop_words="english")),
        ("char_tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=30000)),
    ])),
    ("clf", ClassifierChain(LogisticRegression(max_iter=2000), order="random", random_state=42)),
])

def fit_predict(Xa, Ya, Xb):
    p = None
    for i in range(N_CHAINS):
        m = clone(model)
        m.set_params(clf=ClassifierChain(LogisticRegression(max_iter=2000), order="random", random_state=i))
        m.fit(Xa, Ya)
        q = m.predict_proba(Xb)
        p = q if p is None else p + q
    return p / N_CHAINS

AUG = ['T Spine NC', 'Facial Bones C+', 'Orbits C+', 'Neck NC', 'Orbits NC', 'Temporal Bones NC',
       'Neck Angiogram', 'Temporal Bones C+', 'C Spine C+', 'L Spine C+', 'Sinus NC']
CUE_REGEX = os.environ.get("CUE_REGEX", r"mastoid|infect|drainage")
cue = X_tr.str.contains(CUE_REGEX, case=False, regex=True).values
nonaug = train[LABEL].map(lambda l: any(x not in AUG for x in l)).values
boost = cue & nonaug & ~syn
if SCHEME == "aug_realval_cb":
    n_na, n_a, n_s = boost.sum(), (cue & ~nonaug & ~syn).sum(), (cue & syn).sum()
    s0 = n_na / (n_na + n_a)
    k_formula = s0 * (n_a + n_s) / (n_na * (1 - s0))
    CUE_K = int(round(float(os.environ.get("CUE_K", k_formula))))
    print(f"cue rows: real non-aug {n_na}, real aug-only {n_a}, synthetic {n_s}; k_formula={k_formula:.2f}; using k={CUE_K}", flush=True)
else:
    CUE_K = 1

def with_boost(idx):  # repeat boosted rows (k-1) extra times, training rows only
    extra = idx[boost[idx]]
    return np.concatenate([idx] + [extra] * (CUE_K - 1)) if CUE_K > 1 else idx

# which rows are split into folds / which rows are always-train
if SCHEME in ("aug_realval", "aug_realval_cb"):
    fold_rows = np.where(~syn)[0]; always_train = np.where(syn)[0]
else:
    fold_rows = np.arange(len(train)); always_train = np.array([], dtype=int)

oof = np.full((N_REPEATS, len(train), Y_tr.shape[1]), np.nan)
t0 = time.time()
for r in range(N_REPEATS):
    mskf = MultilabelStratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=r)
    for f, (a, b) in enumerate(mskf.split(fold_rows, Y_tr[fold_rows])):
        tr_idx = with_boost(np.concatenate([fold_rows[a], always_train])); va_idx = fold_rows[b]
        oof[r, va_idx] = fit_predict(X_tr.iloc[tr_idx], Y_tr[tr_idx], X_tr.iloc[va_idx])
        print(f"{SCHEME} repeat {r} fold {f} done {time.time()-t0:.0f}s", flush=True)

full = with_boost(np.arange(len(train)))
test_prob = fit_predict(X_tr.iloc[full], Y_tr[full], X_te)
os.makedirs(OUT, exist_ok=True)
TAG = SCHEME + (f"_k{CUE_K}" if SCHEME == "aug_realval_cb" else "")
np.savez_compressed(f"{OUT}/{TAG}.npz", oof=oof, cue_k=CUE_K, boost=boost, Y_tr=Y_tr, syn=syn, test_prob=test_prob, Y_te=Y_te,
                    labels=np.array(mlb.classes_), test_text=X_te.values,
                    test_id=test["Identifier"].values)
print("saved", f"{OUT}/{TAG}.npz", f"{time.time()-t0:.0f}s")
