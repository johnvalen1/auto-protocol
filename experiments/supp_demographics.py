"""Supplementary analysis: baseline model + patient age (standardized) and sex (one-hot).
Same data prep, CV, seeds and outputs as final_four_arm.py; text-only baseline is unchanged.
usage: python supp_demographics.py <V7.xlsx> <V8.xlsx> baseline_demo <out_dir>"""

import sys, os, time, json
import numpy as np, pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import ClassifierChain
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

V7, V8, ARM, OUT = sys.argv[1:5]
assert ARM == "baseline_demo"  # supplementary: baseline + age and sex
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

# ---- optional: larger synthetic pool (V7 + V8 synthetic rows + extra CSV), duplicates dropped ----
norm = lambda s: " ".join(str(s).lower().split())
if os.environ.get("USE_V8_SYN") == "1" or os.environ.get("EXTRA_SYN"):
    extra = []
    if os.environ.get("USE_V8_SYN") == "1":
        e = v8[v8.original_or_augmented == "augmented"][[TEXT, LABEL]].copy(); e["src"] = "V8"; extra.append(e)
    if os.environ.get("EXTRA_SYN"):
        e = pd.read_csv(os.environ["EXTRA_SYN"])[[TEXT, LABEL]]; e["src"] = "extra"; extra.append(e)
    extra = pd.concat(extra, ignore_index=True)
    extra[LABEL] = extra[LABEL].apply(lambda x: sorted({FIX.get(t.strip(), t.strip()) for t in str(x).split(",")}))
    extra = extra[extra[LABEL].map(lambda l: len(l) == 1 and l[0] in AUG)]
    seen = set(df[df.synthetic][TEXT].map(norm)); add = []
    for _, r in extra.iterrows():
        k = norm(r[TEXT])
        if k not in seen: seen.add(k); add.append(r)
    add = pd.DataFrame(add)
    add = add.assign(synthetic=True, ambiguous=False, **{SPLIT: "train", "Identifier": [f"syn-{s}-{i}" for i, s in enumerate(add.src)]})
    summary.update(extra_synthetic_added={k: int(v) for k, v in add.src.value_counts().items()})
    df = pd.concat([df, add.drop(columns="src")], ignore_index=True)

# ---- optional: rebalance the test set to exactly BALANCE_TEST studies per augmented protocol ----
# Only single-protocol studies are moved (random, seed 42), so no other protocol's counts change.
if os.environ.get("BALANCE_TEST"):
    k = int(os.environ["BALANCE_TEST"]); rng_b = np.random.default_rng(42); moves = {}
    for p in AUG:
        real = ~df.synthetic; single = df[LABEL].map(lambda l: l == [p]); has = df[LABEL].map(lambda l: p in l)
        n_test = int((real & has & (df[SPLIT] == "test")).sum())
        if n_test > k:
            idx = rng_b.choice(np.where(real & single & (df[SPLIT] == "test"))[0], n_test - k, replace=False); df.loc[idx, SPLIT] = "train"
        elif n_test < k:
            idx = rng_b.choice(np.where(real & single & (df[SPLIT] == "train"))[0], k - n_test, replace=False); df.loc[idx, SPLIT] = "test"
        if n_test != k: moves[p] = {"from": n_test, "to": k, "identifiers": [str(x) for x in df.loc[idx, "Identifier"]]}
    summary.update(test_rebalancing=moves)

AMB_TRAIN = os.environ.get("INCLUDE_AMBIG_TRAIN", "0") == "1"   # 1 = keep ambiguous studies in training
train_real = df[(df[SPLIT] == "train") & ~df.synthetic & (AMB_TRAIN | ~df.ambiguous)].reset_index(drop=True)
test_all = df[(df[SPLIT] == "test") & ~df.synthetic].reset_index(drop=True)          # primary + ambiguous
syn_pool = df[df.synthetic & (df[SPLIT] == "train")].reset_index(drop=True)

real_n = {p: int(train_real[LABEL].map(lambda l: p in l).sum()) for p in AUG}
syn_n = {p: int(syn_pool[LABEL].map(lambda l: p in l).sum()) for p in AUG}
TARGET = min(real_n[p] + syn_n[p] for p in AUG)
# never exceed (least frequent non-augmented protocol's real training count) - 1, to keep the frequency order
nonaug_min = min(int(train_real[LABEL].map(lambda l: q in l).sum()) for q in {l for ls in train_real[LABEL] for l in ls} - set(AUG))
CAP = nonaug_min - 1
if os.environ.get("CAP_BELOW_COMMON") == "1": TARGET = min(TARGET, CAP)
if os.environ.get("TARGET"): TARGET = int(os.environ["TARGET"])  # override the fixed training total
summary.update(cap_below_common=CAP)
rng = np.random.default_rng(42); keep = []
for p in AUG:
    rows = np.where(syn_pool[LABEL].map(lambda l: l == [p]))[0]
    need = max(0, TARGET - real_n[p]); assert need <= len(rows), p  # protocols already above TARGET get no synthetic rows
    keep += list(rng.choice(rows, need, replace=False))
syn_used = syn_pool.iloc[sorted(keep)].reset_index(drop=True)
summary.update(ambiguous_in_training=AMB_TRAIN, target_total=TARGET, limiting=min(AUG, key=lambda p: real_n[p] + syn_n[p]),
               real_train_per_aug=real_n, synthetic_available=syn_n,
               synthetic_used={p: max(0, TARGET - real_n[p]) for p in AUG}, synthetic_used_total=len(syn_used),
               train_real=len(train_real), test_primary=int((~test_all.ambiguous).sum()),
               test_ambiguous=int(test_all.ambiguous.sum()))
os.makedirs(OUT, exist_ok=True)
syn_used.assign(**{LABEL: syn_used[LABEL].map(", ".join)})[["Identifier", TEXT, LABEL]].to_csv(f"{OUT}/synthetic_used.csv", index=False)
if os.environ.get("PREP_ONLY") == "1": json.dump(summary, open(f"{OUT}/data_summary.json", "w"), indent=1); sys.exit(0)

train = train_real  # no synthetic rows (they have no age or sex)
mlb = MultiLabelBinarizer(); mlb.fit(train_real[LABEL])
Y_tr = mlb.transform(train[LABEL]); Y_te = mlb.transform(test_all[LABEL])
AGE, SEX = "patient_age_x3", "patient_sex_x2"
def frame(d): return pd.DataFrame({TEXT: d[TEXT].astype(str).values, AGE: d[AGE].astype(float).values, SEX: d[SEX].astype(str).values})
X_tr, X_te = frame(train), frame(test_all)
assert X_tr[AGE].notna().all() and X_te[AGE].notna().all()
syn = train["synthetic"].values
summary.update(n_protocols=len(mlb.classes_), protocols=list(mlb.classes_))

# ---------------- model ----------------
CW = None
text_features = FeatureUnion([
        ("word_tfidf", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=20000, stop_words="english")),
        ("char_tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=30000)),
    ])
model = Pipeline([
    ("features", ColumnTransformer([
        ("text", text_features, TEXT),
        ("age", StandardScaler(), [AGE]),
        ("sex", OneHotEncoder(handle_unknown="ignore"), [SEX]),
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
                    test_text=X_te[TEXT].values, test_id=test_all.Identifier.values)
json.dump(summary, open(f"{OUT}/data_summary.json", "w"), indent=1)
print("saved", f"{OUT}/{ARM}.npz", f"{time.time()-t0:.0f}s")
