""" This script returns the class distribution of the input file's protocol column. """

import archive_scripts.helpers as helpers
import pandas as pd

# input_excel_train_aug = "./train test sets/CT_Neuro_Gold_Standard - Train Set.xlsx"
# gold_standard_CT_neuro_train_aug = helpers.readData(filepath=input_excel_train_aug)
# # get only the train rows which are original, not augmented
# gold_standard_CT_neuro_train = gold_standard_CT_neuro_train_aug[gold_standard_CT_neuro_train_aug["original_or_augmented"]=="original"]
# print(gold_standard_CT_neuro_train)

input_excel_train = "./doc/train/df_train.xlsx"
gold_standard_CT_neuro_train = helpers.readData(filepath=input_excel_train)

input_excel_train_aug = "./doc/train/augmented_df_train.xlsx"
gold_standard_CT_neuro_train_aug = helpers.readData(filepath=input_excel_train_aug)

#input_excel_test = "./train test sets/CT_Neuro_Gold_Standard - Test Set.xlsx"
input_excel_test = "./doc/test/Test_Set_Results.xlsx"
gold_standard_CT_neuro_test = helpers.readData(filepath=input_excel_test)

input_excel_test_aug = "./doc/test/Augmented_Test_Set_Results.xlsx"
gold_standard_CT_neuro_test_aug = helpers.readData(filepath=input_excel_test_aug)

# input_excel_combined = "./CT_Neuro_Gold_Standard.xlsx"
# gold_standard_CT_neuro = helpers.readData(filepath=input_excel_combined)

# input_excel_combined_aug = "./CT_Neuro_Gold_Standard - Augmented.xlsx"
# gold_standard_CT_neuro_aug = helpers.readData(filepath=input_excel_combined_aug)

def PivotTableCounts(df):
    list_of_protocols = []
    for protocols in df['protocol']:
        protocol_list_in_row = protocols.split(",")
        for protocol in protocol_list_in_row:
            list_of_protocols.append(protocol.strip().lower())

    # Create DataFrame
    df = pd.DataFrame(list_of_protocols, columns=['protocol'])

    # Get count (value counts)
    pivot_table = df['protocol'].value_counts().reset_index()
    pivot_table.columns = ['protocol', 'Count']

    # Add percentage column
    pivot_table['pct of grand total'] = (pivot_table['Count'] / pivot_table['Count'].sum()) * 100

    # Optional: round to 2 decimal places
    pivot_table['pct of grand total'] = pivot_table['pct of grand total'].round(2)

    return pivot_table

train_set_pivot_aug = PivotTableCounts(gold_standard_CT_neuro_train_aug)
train_set_pivot_aug.to_excel("./doc/protocol_distributions/train_aug_distribution.xlsx")

train_set_pivot = PivotTableCounts(gold_standard_CT_neuro_train)
train_set_pivot.to_excel("./doc/protocol_distributions/train_distribution.xlsx")

test_set_pivot = PivotTableCounts(gold_standard_CT_neuro_test)
test_set_pivot.to_excel("./doc/protocol_distributions/test_distribution.xlsx")

test_set_pivot_aug = PivotTableCounts(gold_standard_CT_neuro_test_aug)
test_set_pivot_aug.to_excel("./doc/protocol_distributions/test_aug_distribution.xlsx")

# combined_pivot = PivotTableCounts(gold_standard_CT_neuro)
# combined_pivot.to_excel("./doc/protocol_distributions/combined_distribution.xlsx")

# combined_pivot_aug = PivotTableCounts(gold_standard_CT_neuro_aug)
# combined_pivot_aug.to_excel("./doc/protocol_distributions/combined_aug_distribution.xlsx")
