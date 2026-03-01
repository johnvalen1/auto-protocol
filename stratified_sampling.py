import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

# ---- SETTINGS ----
input_file = "./datasets/CT_Neuro_Gold_Standard_V5.xlsx"
output_file = "./datasets/CT_Neuro_Gold_Standard_V5_with_train_test_split.xlsx"
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

for train_idx, test_idx in msss.split(X, Y):
    df["split"] = "train"
    df.loc[test_idx, "split"] = "test"

# Drop helper column
df.drop(columns=["label_list"], inplace=True)

# Save back to Excel
df.to_excel(output_file, index=False)

print("Done. File saved as:", output_file)


def label_distribution(Y):
    return Y.mean(axis=0)

Y_train = Y[train_idx]
Y_test = Y[test_idx]

import numpy as np
import pandas as pd

def label_distribution(Y):
    return Y.mean(axis=0)

# Compute distributions
overall = label_distribution(Y)
train_dist = label_distribution(Y_train)
test_dist = label_distribution(Y_test)

# Convert to percentage
results = pd.DataFrame({
    "Label": mlb.classes_,
    "Overall (%)": overall * 100,
    "Train (%)": train_dist * 100,
    "Test (%)": test_dist * 100,
    "Train Diff (%)": (train_dist - overall) * 100,
    "Test Diff (%)": (test_dist - overall) * 100
})

# Round for readability
results = results.round(3)

# print(results)

# print("\nMaximum absolute difference:")
# print("Train:", np.abs(train_dist - overall).max() * 100, "%")
# print("Test:", np.abs(test_dist - overall).max() * 100, "%")

output_validation_file = "./doc/label_distribution_validation.xlsx"

# Write to Excel
results.to_excel(output_validation_file,
                 index=False,
                 sheet_name="Label Distribution")

print(f"Validation report written to: {output_validation_file}")