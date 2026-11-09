"""
Final four-arm analysis (pre-registered in EXPERIMENT_LOG.md, "Final analysis").

Data
  - Base: V7 workbook, sheet `data` (split, exclusions, and the 1,227 synthetic
    indications used for the ARRS submission). Rows with exclude_record = TRUE dropped.
  - Ambiguous studies: real studies flagged
    exclude_record_based_on_anatomical_ambiguity = 1 in the V8 workbook (matched on
    Identifier) are removed from training and from the primary test set.
  - Label fix: "Head Angiogram or COW" merged into "Head angiogram or COW".
  - Neck NC removed (67 of its 69 real studies are ambiguous; 2 remain, none in test).
    All rows carrying Neck NC, real and synthetic, are dropped.
  - INCLUDE_AMBIG_TRAIN=1 keeps ambiguous studies in training (default: removed).
    TARGET=<n> overrides the fixed training total.
  - Synthetic dose: every augmented protocol is brought to the same total of real
    training + synthetic cases, TARGET = the largest total all of them can reach
    from the V7 synthetic rows (T Spine NC limits it). Synthetic rows are subsampled
    without replacement, seed 42.

Arms: baseline | class_weight | gpt4o | gpt4o_cw
  Model, CV (5 x 4 multilabel stratified folds over real training rows), real-only
  threshold tuning and 1-3 advised protocols as in threshold_fix.py.

Outputs <out_dir>/<arm>.npz with out-of-fold probabilities and probabilities for
the full test set (primary rows + ambiguous rows), and data_summary.json.

usage: python final_four_arm.py <V7.xlsx> <V8.xlsx> <arm> <out_dir>
"""
import sys, os, time, json
import numpy as np, pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import ClassifierChain
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import clone
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

V7, V8, ARM, OUT = sys.argv[1:5]
assert ARM in ("baseline", "class_weight", "gpt4o", "gpt4o_cw")
TEXT, LABEL, SPLIT = "indication_x1", "protocol_y", "split"
AUG = ['T Spine NC', 'Facial Bones C+', 'Orbits C+', 'Orbits NC', 'Temporal Bones NC',
       'Neck Angiogram', 'Temporal Bones C+', 'C Spine C+', 'L Spine C+', 'Sinus NC']  # Neck NC removed
FIX = {"Head Angiogram or COW": "Head angiogram or COW"}

# ---------------- data ----------------
df = pd.read_excel(V7, sheet_name="data")
df = df[df["exclude_record"].astype(str).str.upper() != "TRUE"]
df = df[df[SPLIT].isin(["train", "test"])].reset_index(drop=True)
df[LABEL] = df[LABEL].apply(lambda x: sorted({FIX.get(t.strip(), t.strip()) for t in str(x).split(",")}))
df["synthetic"] = df["original_or_augmented"] == "augmented"
v8 = pd.read_excel(V8, sheet_name="data")
amb_ids = set(v8[(v8.original_or_augmented == "original") & (v8.exclude_record_based_on_anatomical_ambiguity == 1)].Identifier)
df["ambiguous"] = (~df.synthetic) & df.Identifier.isin(amb_ids)
summary = dict(v7_real=int((~df.synthetic).sum()), v7_synthetic=int(df.synthetic.sum()),
               v8_flagged_total=len(amb_ids), ambiguous_in_v7=int(df.ambiguous.sum()),
               ambiguous_train=int((df.ambiguous & (df[SPLIT] == "train")).sum()),
               ambiguous_test=int((df.ambiguous & (df[SPLIT] == "test")).sum()))
neck_nc = df[LABEL].map(lambda l: "Neck NC" in l)
summary.update(neck_nc_real_nonambiguous=int((neck_nc & ~df.synthetic & ~df.ambiguous).sum()),
               neck_nc_real_ambiguous=int((neck_nc & df.ambiguous).sum()), neck_nc_synthetic=int((neck_nc & df.synthetic).sum()))
df = df[~neck_nc].reset_index(drop=True)

AMB_TRAIN = os.environ.get("INCLUDE_AMBIG_TRAIN", "0") == "1"   # 1 = keep ambiguous studies in training
train_real = df[(df[SPLIT] == "train") & ~df.synthetic & (AMB_TRAIN | ~df.ambiguous)].reset_index(drop=True)
test_all = df[(df[SPLIT] == "test") & ~df.synthetic].reset_index(drop=True)          # primary + ambiguous
syn_pool = df[df.synthetic & (df[SPLIT] == "train")].reset_index(drop=True)

real_n = {p: int(train_real[LABEL].map(lambda l: p in l).sum()) for p in AUG}
syn_n = {p: int(syn_pool[LABEL].map(lambda l: p in l).sum()) for p in AUG}
TARGET = min(real_n[p] + syn_n[p] for p in AUG)
if os.environ.get("TARGET"): TARGET = int(os.environ["TARGET"])  # override the fixed training total
rng = np.random.default_rng(42); keep = []
for p in AUG:
    rows = np.where(syn_pool[LABEL].map(lambda l: l == [p]))[0]
    need = TARGET - real_n[p]; assert 0 <= need <= len(rows), p
    keep += list(rng.choice(rows, need, replace=False))
syn_used = syn_pool.iloc[sorted(keep)].reset_index(drop=True)
summary.update(ambiguous_in_training=AMB_TRAIN, target_total=TARGET, limiting=min(AUG, key=lambda p: real_n[p] + syn_n[p]),
               real_train_per_aug=real_n, synthetic_available=syn_n,
               synthetic_used={p: TARGET - real_n[p] for p in AUG}, synthetic_used_total=len(syn_used),
               train_real=len(train_real), test_primary=int((~test_all.ambiguous).sum()),
               test_ambiguous=int(test_all.ambiguous.sum()))

train = pd.concat([train_real, syn_used], ignore_index=True) if ARM in ("gpt4o", "gpt4o_cw") else train_real
mlb = MultiLabelBinarizer(); mlb.fit(train_real[LABEL])
Y_tr = mlb.transform(train[LABEL]); Y_te = mlb.transform(test_all[LABEL])
X_tr, X_te = train[TEXT].astype(str), test_all[TEXT].astype(str)
syn = train["synthetic"].values
summary.update(n_protocols=len(mlb.classes_), protocols=list(mlb.classes_))

# ---------------- model ----------------
CW = "balanced" if ARM in ("class_weight", "gpt4o_cw") else None
model = Pipeline([
    ("features", FeatureUnion([
        ("word_tfidf", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=20000, stop_words="english")),
        ("char_tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=30000)),
    ])),
    ("clf", ClassifierChain(LogisticRegression(max_iter=2000), order="random", random_state=0)),
])
def fit_predict(Xa, Ya, Xb):
    p = None
    for i in range(3):
        m = clone(model)
        m.set_params(clf=ClassifierChain(LogisticRegression(max_iter=2000, class_weight=CW), order="random", random_state=i))
        m.fit(Xa, Ya); q = m.predict_proba(Xb); p = q if p is None else p + q
    return p / 3

fold_rows = np.where(~syn)[0]; always = np.where(syn)[0]
oof = np.full((5, len(train), Y_tr.shape[1]), np.nan); t0 = time.time()
for r in range(5):
    for f, (a, b) in enumerate(MultilabelStratifiedKFold(4, shuffle=True, random_state=r).split(fold_rows, Y_tr[fold_rows])):
        tr = np.concatenate([fold_rows[a], always]); va = fold_rows[b]
        oof[r, va] = fit_predict(X_tr.iloc[tr], Y_tr[tr], X_tr.iloc[va])
        print(f"{ARM} repeat {r} fold {f} {time.time()-t0:.0f}s", flush=True)
test_prob = fit_predict(X_tr, Y_tr, X_te)
os.makedirs(OUT, exist_ok=True)
np.savez_compressed(f"{OUT}/{ARM}.npz", oof=oof, Y_tr=Y_tr, syn=syn, test_prob=test_prob, Y_te=Y_te,
                    ambiguous=test_all.ambiguous.values, labels=np.array(mlb.classes_), aug=np.array(AUG),
                    test_text=X_te.values, test_id=test_all.Identifier.values)
json.dump(summary, open(f"{OUT}/data_summary.json", "w"), indent=1)
print("saved", f"{OUT}/{ARM}.npz", f"{time.time()-t0:.0f}s")
