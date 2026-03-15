import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

# ---- SETTINGS ----
input_file = "./datasets/CT_Neuro_Gold_Standard_V6.xlsx"
output_file = "./datasets/CT_Neuro_Gold_Standard_V6_with_train_test_split.xlsx"
label_column = "protocol_y"
test_size = 0.1
random_state = 42
# -------------------

# Load Excel
df = pd.read_excel(input_file,
    sheet_name="data")   # exact sheet name)
print(len(df))
# exclude augmented records
df = df[
    (df["original_or_augmented"] == "original")
]
print(len(df))
# exclude the records that have too few examples
df = df[
    (df["exclude_record"] == False)
]
print(len(df))

# Convert label string to list
df["label_list"] = df[label_column].apply(
    lambda x: [label.strip() for label in str(x).split(",")]
)
df = df.reset_index(drop=True)
# Multi-hot encode labels
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(df["label_list"])

# Prepare dummy X (we only need indices)
X = df.index.values

# Multilabel stratified split
msss = MultilabelStratifiedShuffleSplit(
    n_splits=1,
    test_size=test_size,
    random_state=random_state
)
print(df)
for train_idx, test_idx in msss.split(X, Y):
    df["split"] = "train"
    df.loc[test_idx, "split"] = "test"

# Drop helper column
df.drop(columns=["label_list"], inplace=True)

# Save back to Excel
df.to_excel(output_file, index=False)

print("Done. File saved as:", output_file)

# -------------------------------------------------------
# Label distribution validation (ACTUAL counts + percent)
# -------------------------------------------------------

import numpy as np
import pandas as pd

# Convert label matrix to DataFrame
label_df = pd.DataFrame(Y, columns=mlb.classes_)

# Add split column from dataframe
label_df["split"] = df["split"].values

# Separate splits
train_df = label_df[label_df["split"] == "train"].drop(columns="split")
test_df = label_df[label_df["split"] == "test"].drop(columns="split")

# ---- Counts ----
total_counts = label_df.drop(columns="split").sum()
train_counts = train_df.sum()
test_counts = test_df.sum()

# ---- Percentages ----
overall_pct = total_counts / len(label_df) * 100
train_pct = train_counts / len(train_df) * 100
test_pct = test_counts / len(test_df) * 100

# ---- Build validation table ----
results = pd.DataFrame({
    "Label": mlb.classes_,
    "Total Count": total_counts.values,
    "Train Count": train_counts.values,
    "Test Count": test_counts.values,
    "Overall (%)": overall_pct.values,
    "Train (%)": train_pct.values,
    "Test (%)": test_pct.values,
    "Train Diff (%)": (train_pct - overall_pct).values,
    "Test Diff (%)": (test_pct - overall_pct).values
})

results = results.round(3)

# Sort by rarest labels
results = results.sort_values("Total Count")

output_validation_file = "./doc/label_distribution_validation.xlsx"

results.to_excel(
    output_validation_file,
    index=False,
    sheet_name="Label Distribution"
)

print(f"Validation report written to: {output_validation_file}")