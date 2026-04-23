import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import ClassifierChain
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import clone

from sklearn.metrics import f1_score, roc_auc_score, hamming_loss, accuracy_score
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

# -------------------------
# SETTINGS
# -------------------------
INPUT_FILE = "./datasets/CT_Neuro_Gold_Standard_V6.xlsx"
INPUT_FILE_WITH_PREDICTIONS = "./datasets/CT_Neuro_Gold_Standard_V6_WITH_PREDICTIONS_age_and_gender.xlsx"
OUTPUT_FILE = "./datasets/model_results_age_and_gender.xlsx"
MODEL_FILE = "./models/protocol_model_age_and_gender.joblib"

TEXT_COL = "indication_x1"
AGE_COL = "patient_age_x3"
GENDER_COL = "patient_sex_x2"
LABEL_COL = "protocol_y"
SPLIT_COL = "split"

N_SPLITS = 4
N_REPEATS = 5
TOP_K = 3
N_CHAINS = 3

FEATURE_COLS = [TEXT_COL, AGE_COL, GENDER_COL]

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_excel(INPUT_FILE, sheet_name="data")

# Ensure protocol labels are lists
df[LABEL_COL] = df[LABEL_COL].apply(lambda x: [i.strip() for i in str(x).split(",")])

# Clean gender and fill missing values
df[GENDER_COL] = df[GENDER_COL].fillna("Unspecified").str.strip()

# Split data
train_df = df[df[SPLIT_COL] == "train"].reset_index(drop=True)
test_df = df[df[SPLIT_COL] == "test"].reset_index(drop=True)

# -------------------------
# MULTI-LABEL BINARIZATION
# -------------------------
mlb = MultiLabelBinarizer()
Y_train = mlb.fit_transform(train_df[LABEL_COL])
Y_test = mlb.transform(test_df[LABEL_COL])

X_train = train_df[FEATURE_COLS]
X_test = test_df[FEATURE_COLS]

label_names = mlb.classes_
n_labels = len(label_names)

# -------------------------
# ENSEMBLE TRAINING FUNCTIONS
# -------------------------
def train_chain_ensemble(X, Y, base_pipeline, n_chains=N_CHAINS):
    chains = []
    for i in range(n_chains):
        chain_model = clone(base_pipeline)
        chain_model.set_params(
            clf=ClassifierChain(
                LogisticRegression(max_iter=2000),
                order="random",
                random_state=i
            )
        )
        chain_model.fit(X, Y)
        chains.append(chain_model)
    return chains

def ensemble_predict_proba(chains, X):
    probs = None
    for model in chains:
        p = model.predict_proba(X)
        if probs is None:
            probs = p
        else:
            probs += p
    probs /= len(chains)
    return probs

# -------------------------
# MODEL PIPELINE
# -------------------------
model = Pipeline([
    ("features", ColumnTransformer([
        ("word_tfidf", TfidfVectorizer(analyzer="word", ngram_range=(1,2), max_features=20000, stop_words="english"), TEXT_COL),
        ("char_tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), max_features=30000), TEXT_COL),
        ("age", StandardScaler(), [AGE_COL]),
        ("gender", OneHotEncoder(handle_unknown="ignore"), [GENDER_COL])
    ])),
    ("clf", ClassifierChain(
        LogisticRegression(max_iter=2000),
        order="random",
        random_state=42
    ))
])

# -------------------------
# METRICS
# -------------------------
def top_k_accuracy(y_true, y_prob, k):
    correct = 0
    for i in range(len(y_true)):
        top_k = np.argsort(y_prob[i])[-k:]
        true_labels = np.where(y_true[i] == 1)[0]
        if len(set(top_k).intersection(true_labels)) > 0:
            correct += 1
    return correct / len(y_true)

def sensitivity_per_class(y_true, y_pred):
    sens = {}
    for i, label in enumerate(label_names):
        tp = np.sum((y_true[:,i]==1) & (y_pred[:,i]==1))
        fn = np.sum((y_true[:,i]==1) & (y_pred[:,i]==0))
        sens[label] = tp/(tp+fn) if (tp+fn) > 0 else np.nan
    return sens

def compute_global_metrics(y_true, y_pred, y_prob):
    metrics = {}
    metrics["weighted_f1"] = f1_score(y_true, y_pred, average="weighted")
    metrics["micro_f1"] = f1_score(y_true, y_pred, average="micro")
    metrics["macro_f1"] = f1_score(y_true, y_pred, average="macro")
    metrics["hamming_loss"] = hamming_loss(y_true, y_pred)
    metrics["subset_accuracy"] = accuracy_score(y_true, y_pred)
    metrics["macro_auc"] = roc_auc_score(y_true, y_prob, average="macro")
    return metrics

def bootstrap_metrics(y_true, y_pred, y_prob, n_boot=1000):
    rng = np.random.default_rng(42)
    results = []
    n = len(y_true)
    for i in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        y_t = y_true[idx]
        y_p = y_pred[idx]
        y_pr = y_prob[idx]
        metrics = compute_global_metrics(y_t, y_p, y_pr)
        results.append(metrics)
    boot_df = pd.DataFrame(results)
    ci = boot_df.quantile([0.025,0.975]).T
    ci.columns = ["ci_lower","ci_upper"]
    mean_vals = boot_df.mean()
    summary = pd.concat([mean_vals, ci], axis=1)
    summary.columns = ["mean","ci_lower","ci_upper"]
    return summary, boot_df

def protocol_confusion_stats(y_true, y_pred):
    rows = []
    for i, label in enumerate(label_names):
        tp = np.sum((y_true[:,i]==1) & (y_pred[:,i]==1))
        fp = np.sum((y_true[:,i]==0) & (y_pred[:,i]==1))
        fn = np.sum((y_true[:,i]==1) & (y_pred[:,i]==0))
        tn = np.sum((y_true[:,i]==0) & (y_pred[:,i]==0))
        sensitivity = tp/(tp+fn) if (tp+fn)>0 else np.nan
        precision = tp/(tp+fp) if (tp+fp)>0 else np.nan
        f1 = (2*precision*sensitivity/(precision+sensitivity)) if (precision+sensitivity)>0 else np.nan
        rows.append({
            "protocol":label,
            "tp":tp,
            "fp":fp,
            "fn":fn,
            "tn":tn,
            "sensitivity":sensitivity,
            "precision":precision,
            "f1":f1
        })
    return pd.DataFrame(rows)

def optimize_thresholds(y_true, y_prob):
    thresholds = np.zeros(n_labels)
    for i in range(n_labels):
        best_f1 = 0
        best_t = 0.5
        for t in np.linspace(0.05,0.95,19):
            preds = (y_prob[:,i] >= t).astype(int)
            f1 = f1_score(y_true[:,i], preds)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        thresholds[i] = best_t
    return thresholds

def apply_thresholds_with_limits(y_prob, thresholds, min_labels=1, max_labels=3):
    preds = np.zeros_like(y_prob, dtype=int)
    for i in range(len(y_prob)):
        probs = y_prob[i]
        labels = np.where(probs >= thresholds)[0]
        if len(labels) < min_labels:
            labels = np.argsort(probs)[-min_labels:]
        if len(labels) > max_labels:
            labels = labels[np.argsort(probs[labels])[-max_labels:]]
        preds[i, labels] = 1
    return preds

# -------------------------
# CROSS VALIDATION
# -------------------------
cv_results = []
all_thresholds = []

for repeat in range(N_REPEATS):
    mskf = MultilabelStratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=repeat
    )
    for fold, (train_idx, val_idx) in enumerate(mskf.split(X_train, Y_train)):
        X_tr = X_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        y_tr = Y_train[train_idx]
        y_val = Y_train[val_idx]

        chains = train_chain_ensemble(X_tr, y_tr, model, N_CHAINS)
        y_prob = ensemble_predict_proba(chains, X_val)

        thresholds = optimize_thresholds(y_val, y_prob)
        all_thresholds.append(thresholds)

        y_pred = apply_thresholds_with_limits(y_prob, thresholds)

        results = {
            "repeat":repeat,
            "fold":fold,
            "top_k_accuracy":top_k_accuracy(y_val, y_prob, TOP_K),
            "weighted_f1":f1_score(y_val, y_pred, average="weighted"),
            "micro_f1":f1_score(y_val, y_pred, average="micro"),
            "macro_f1":f1_score(y_val, y_pred, average="macro"),
            "hamming_loss":hamming_loss(y_val, y_pred),
            "subset_accuracy":accuracy_score(y_val, y_pred),
            "macro_auc":roc_auc_score(y_val, y_prob, average="macro")
        }
        cv_results.append(results)

cv_results_df = pd.DataFrame(cv_results)
cv_summary = cv_results_df.describe().loc[["mean","std"]]

# -------------------------
# FINAL THRESHOLDS
# -------------------------
final_thresholds = np.mean(np.vstack(all_thresholds), axis=0)

# -------------------------
# TRAIN FINAL MODEL
# -------------------------
chains = train_chain_ensemble(X_train, Y_train, model, N_CHAINS)

# -------------------------
# TEST SET EVALUATION
# -------------------------
y_test_prob = ensemble_predict_proba(chains, X_test)
y_test_pred = apply_thresholds_with_limits(y_test_prob, final_thresholds)

test_metrics = compute_global_metrics(Y_test, y_test_pred, y_test_prob)
test_metrics["top_k_accuracy"] = top_k_accuracy(Y_test, y_test_prob, TOP_K)

test_results_df = pd.DataFrame(list(test_metrics.items()), columns=["metric","value"])
test_ci_df, boot_dist = bootstrap_metrics(Y_test, y_test_pred, y_test_prob)
test_ci_df = test_ci_df.reset_index()
test_ci_df.rename(columns={"index":"metric"}, inplace=True)

sens_df = pd.DataFrame(sensitivity_per_class(Y_test, y_test_pred).items(), columns=["protocol","sensitivity"])
protocol_conf_df = protocol_confusion_stats(Y_test, y_test_pred)

# -------------------------
# GENERATE PREDICTIONS FOR TEST SET ONLY
# -------------------------
df["predicted_protocols"] = ""
test_indices = df[df[SPLIT_COL] == "test"].index
X_test_full = df.loc[test_indices, FEATURE_COLS]
y_test_prob_full = ensemble_predict_proba(chains, X_test_full)
y_test_pred_full = apply_thresholds_with_limits(y_test_prob_full, final_thresholds)

predicted_labels = mlb.inverse_transform(y_test_pred_full)
df.loc[test_indices, "predicted_protocols"] = [
    ", ".join(labels) if len(labels) > 0 else ""
    for labels in predicted_labels
]

df.to_excel(INPUT_FILE_WITH_PREDICTIONS, index=False)

# -------------------------
# SAVE RESULTS
# -------------------------
with pd.ExcelWriter(OUTPUT_FILE) as writer:
    cv_results_df.to_excel(writer,"CV_Folds",index=False)
    cv_summary.to_excel(writer,"CV_Summary")
    test_results_df.to_excel(writer,"Test_Metrics",index=False)
    test_ci_df.to_excel(writer,"Test_Metrics_CI",index=False)
    sens_df.to_excel(writer,"Protocol_Sensitivity",index=False)
    protocol_conf_df.to_excel(writer,"Protocol_Confusion",index=False)

# -------------------------
# SAVE MODEL
# -------------------------
model_package = {
    "chains": chains,
    "thresholds": final_thresholds,
    "mlb": mlb,
    "labels": label_names
}

joblib.dump(model_package, MODEL_FILE)

print("Model saved to:", MODEL_FILE)
print("Results saved to:", OUTPUT_FILE)