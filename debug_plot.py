import pandas as pd
from pathlib import Path

# Load files
res_csv = "/home/qyx/ancestor-inference/results/hard_gatk_admixture_1x_20251211_155403/admixture_result/admixture_final_result.csv"
test_samples_csv = "/space/s1/qyx/data/test_samples.csv"
labels_csv = "/space/s1/qyx/data/simulate/labels.tsv"

print("Loading Data...")
df_res = pd.read_csv(res_csv)
df_test = pd.read_csv(test_samples_csv)
try:
    df_labels = pd.read_csv(labels_csv, sep='\t')
except:
    df_labels = pd.read_csv(labels_csv)

# Standardize
df_res['sample'] = df_res['sample'].astype(str).str.strip()
df_test['sample'] = df_test['sample'].astype(str).str.strip()
df_labels['sample'] = df_labels['sample'].astype(str).str.strip()

# Check Merge
print(f"Res samples: {len(df_res)}")
print(f"Label samples: {len(df_labels)}")
merged_df = pd.merge(df_labels, df_res, on='sample', how='inner')
print(f"Merged (Labels+Res) size: {len(merged_df)}")

if merged_df.empty:
    print("MERGE FAILED! checking sample overlap:")
    print("First 5 Res samples:", df_res['sample'].head().tolist())
    print("First 5 Label samples:", df_labels['sample'].head().tolist())
else:
    print("Merge OK.")

# Check Test Filter
test_samples_set = set(df_test['sample'])
print(f"Test samples count: {len(test_samples_set)}")
print("First 5 Test samples:", list(test_samples_set)[:5])

filtered_df = merged_df[merged_df['sample'].isin(test_samples_set)]
print(f"Filtered DF size: {len(filtered_df)}")

if filtered_df.empty:
    print("FILTER FAILED! Checking overlap between merged and test set:")
    merged_samples = set(merged_df['sample'])
    overlap = merged_samples.intersection(test_samples_set)
    print(f"Overlap count: {len(overlap)}")
    
    print("\n--- Diagnostic ---")
    print("Sample in Result but not in Test (First 5):", list(merged_samples - test_samples_set)[:5])
    print("Sample in Test but not in Result (First 5):", list(test_samples_set - merged_samples)[:5])


