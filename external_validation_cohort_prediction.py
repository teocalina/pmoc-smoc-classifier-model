"""
MOC/EOM Hierarchical Classifier — Prediction Pipeline

Reads IDAT files from a directory, runs the 3-step hierarchical classifier,
and outputs a final prediction per sample.

Steps:
  1. LR_MOC: if 1 → "MOC", else →
  2. SVM_EOM: if 0 → "Other Lesion", else →
  3. LR_COAD / LR_STAD: COAD=1 → "EOM-COAD", STAD=1 → "EOM-STAD", else → "EOM-Other"
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from mepylome import MethylData
import joblib

# ============================================================
# CONFIGURATION — point this to your IDAT folder
# ============================================================
IDAT_DIR = "data/EPICv2"

# ============================================================
# 1. Read IDAT files
# ============================================================
basenames = list(set(
    f.replace("_Grn.idat", "").replace("_Red.idat", "")
    for f in os.listdir(IDAT_DIR)
    if f.endswith(".idat")
))

file_paths = [os.path.join(IDAT_DIR, b) for b in basenames]

print(f"Found {len(basenames)} samples in {IDAT_DIR}")

betas = MethylData(file=file_paths, prep="noob").betas
betas = betas.loc[~betas.index.duplicated(keep="first")]

# ============================================================
# 2. Load feature CpG lists
# ============================================================
with open("data/feature_cpg_list_step_1.json") as f:
    cpg_step_1 = json.load(f)
with open("data/feature_cpg_list_step_2.json") as f:
    cpg_step_2 = json.load(f)
with open("data/feature_cpg_list_step_3.json") as f:
    cpg_step_3 = json.load(f)

features_step_1 = betas.loc[cpg_step_1].T
features_step_2 = betas.loc[cpg_step_2].T
features_step_3 = betas.loc[cpg_step_3].T

# ============================================================
# 2b. EPICv2-to-EPICv1 bias correction
# ============================================================
print("Fetching EPICv2-to-EPICv1 bias conversion table...")
url = "https://github.com/zhou-lab/InfiniumAnnotationV1/raw/main/Anno/EPICv2/EPICv2ToEPIC_conversion.tsv.gz"
conversion_df = pd.read_csv(url, sep='\t')

delta_columns = [
    'EPIC1m2_GM12878_rep1', 'EPIC1m2_GM12878_rep2',
    'EPIC1m2_K562_rep1', 'EPIC1m2_K562_rep2',
    'EPIC1m2_LNCaP_rep1', 'EPIC1m2_LNCaP_rep2',
]
conversion_df['mean_bias'] = conversion_df[delta_columns].mean(axis=1)

# Group by EPICv1 probe ID to collapse duplicates
shift_series = conversion_df.groupby('ID_EPIC1')['mean_bias'].mean()

# Align shifts to probe order and apply correction
aligned_shifts = shift_series.reindex(betas.index).fillna(0)
betas_corrected = betas.add(aligned_shifts, axis=0).clip(lower=0, upper=1)

features_corrected_step_1 = betas_corrected.loc[cpg_step_1].T
features_corrected_step_2 = betas_corrected.loc[cpg_step_2].T
features_corrected_step_3 = betas_corrected.loc[cpg_step_3].T

# ============================================================
# 3. Load models (only what the pipeline needs)
# ============================================================
def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# Step 1: LR MOC
pca_lr_MOC = joblib.load("models/STEP1_pca_model_lr.joblib")
lr_MOC = joblib.load("models/STEP1_lr_model.joblib")

# Step 2: SVM EOM
pca_svm_EOM = joblib.load("models/STEP2_pca_model_svm.joblib")
svm_EOM = joblib.load("models/STEP2_svm_model.joblib")

# Step 3: LR COAD + LR STAD
pca_lr_coad = joblib.load("models/STEP3_pca_model_lr_COAD.joblib")
lr_coad = joblib.load("models/STEP3_lr_model_COAD.joblib")
pca_lr_stad = joblib.load("models/models/STEP3_pca_model_lr_STAD.joblib")
lr_stad = joblib.load("models/STEP3_lr_model_STAD.joblib")

# ============================================================
# 4. Run predictions
# ============================================================
results = pd.DataFrame(index=betas.columns)

# Step 1: LR MOC
results["LR_MOC"] = lr_MOC.predict(pca_lr_MOC.transform(features_step_1))

# Step 2: SVM EOM
results["SVM_EOM"] = svm_EOM.predict(pca_svm_EOM.transform(features_step_2))

# Step 3: LR COAD, LR STAD
results["LR_COAD"] = lr_coad.predict(pca_lr_coad.transform(features_step_3))
results["LR_STAD"] = lr_stad.predict(pca_lr_stad.transform(features_step_3))

# ============================================================
# 5. Hierarchical decision logic
# ============================================================
def classify(row):
    if row["LR_MOC"] == 1:
        return "MOC"
    if row["SVM_EOM"] == 0:
        return "Other Lesion"
    if row["LR_COAD"] == 1:
        return "EOM-COAD"
    if row["LR_STAD"] == 1:
        return "EOM-STAD"
    return "EOM-Other"

results["Prediction"] = results.apply(classify, axis=1)

# ============================================================
# 6. Bias-corrected predictions
# ============================================================
results_corrected = pd.DataFrame(index=betas_corrected.columns)

results_corrected["LR_MOC"] = lr_MOC.predict(pca_lr_MOC.transform(features_corrected_step_1))
results_corrected["SVM_EOM"] = svm_EOM.predict(pca_svm_EOM.transform(features_corrected_step_2))
results_corrected["LR_COAD"] = lr_coad.predict(pca_lr_coad.transform(features_corrected_step_3))
results_corrected["LR_STAD"] = lr_stad.predict(pca_lr_stad.transform(features_corrected_step_3))
results_corrected["Prediction"] = results_corrected.apply(classify, axis=1)

# ============================================================
# 7. Output
# ============================================================
print("\n" + "=" * 60)
print("PREDICTIONS (raw betas)")
print("=" * 60)
print(results[["LR_MOC", "SVM_EOM", "LR_COAD", "LR_STAD", "Prediction"]].to_string())
print("\n" + results["Prediction"].value_counts().to_string())

print("\n" + "=" * 60)
print("PREDICTIONS (EPICv2-to-EPICv1 bias-corrected betas)")
print("=" * 60)
print(results_corrected[["LR_MOC", "SVM_EOM", "LR_COAD", "LR_STAD", "Prediction"]].to_string())
print("\n" + results_corrected["Prediction"].value_counts().to_string())
