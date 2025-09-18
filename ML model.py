# ================================
# AMP Prediction with XGBoost + SHAP + Correct Feature Extraction
# ================================

# Install if missing:
# !pip install scikit-learn pandas xgboost joblib biopython modlamp shap

import os
import pandas as pd # Data handling
from modlamp.descriptors import GlobalDescriptor, PeptideDescriptor #Global - used to analysis the physiochemical , Peptide - Computes amino acid composition, transition, and distribution descriptors (like dipeptide composition, amphiphilicity, etc.).
from Bio.SeqUtils.ProtParam import ProteinAnalysis #
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# ======== Boman Scale ======== to analyze with boman index
BOMAN_SCALE = {
    'A': 0.62, 'R': -2.53, 'N': -0.78, 'D': -0.90, 'C': 0.29,
    'E': -0.74, 'Q': -0.85, 'G': 0.48, 'H': -0.40, 'I': 1.38,
    'L': 1.06, 'K': -1.50, 'M': 0.64, 'F': 1.19, 'P': 0.12,
    'S': -0.18, 'T': -0.05, 'W': 0.81, 'Y': 0.26, 'V': 1.08
}

def calculate_boman(sequence):
    if not sequence:
        return float('nan')
    return sum(BOMAN_SCALE.get(aa,0) for aa in sequence) / len(sequence)

# ======== Load AMP and Non-AMP datasets ========
amp = pd.read_csv("AMP.csv")
nonamp = pd.read_csv("NONAMP.csv")
amp["Label"] = 1
nonamp["Label"] = 0
data = pd.concat([amp, nonamp], ignore_index=True)

# ======== Sequence Cleaning ========
valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
def clean_sequence(seq):
    return "".join([aa for aa in str(seq).upper().strip() if aa in valid_aas])

data['Sequence'] = data['Sequence'].apply(clean_sequence)
sequences = data['Sequence'].tolist()

# ======== Feature Extraction Function ========
def extract_features(sequences):
    # 1. Net Charge
    desc_charge = GlobalDescriptor(sequences)
    desc_charge.calculate_charge()
    net_charge = desc_charge.descriptor.flatten()

    # 2. Hydrophobicity
    desc_hydro = PeptideDescriptor(sequences, "eisenberg")
    desc_hydro.calculate_global()
    hydrophobicity = desc_hydro.descriptor.flatten()

    # 3. Amphipathicity
    desc_peptide = PeptideDescriptor(sequences, "eisenberg")
    desc_peptide.calculate_moment()
    amphipathicity = desc_peptide.descriptor.flatten()

    # 4. Length
    length = [len(seq) for seq in sequences]

    # 5. Isoelectric Point, Instability, Aromaticity, AACs
    isoelectric_point = []
    instability_index = []
    aromaticity = []
    aac_list = []
    aliphatic_index = []

    for seq in sequences:
        if seq:
            analysed_seq = ProteinAnalysis(seq)
            isoelectric_point.append(analysed_seq.isoelectric_point())
            instability_index.append(analysed_seq.instability_index())
            aromaticity.append(analysed_seq.aromaticity())
            aa_percent = analysed_seq.amino_acids_percent
            aac_list.append(aa_percent)
            AI = (aa_percent.get("A",0)*100 +
                  aa_percent.get("V",0)*100*2.9 +
                  (aa_percent.get("I",0)+aa_percent.get("L",0))*100*3.9)
            aliphatic_index.append(AI)
        else:
            isoelectric_point.append(float('nan'))
            instability_index.append(float('nan'))
            aromaticity.append(float('nan'))
            aac_list.append({aa:0.0 for aa in valid_aas})
            aliphatic_index.append(float('nan'))

    # 6. Boman Index
    boman_index = [calculate_boman(seq) for seq in sequences]

    # Build feature dataframe
    output_df = pd.DataFrame({
        'Net_Charge': net_charge,
        'Hydrophobicity': hydrophobicity,
        'Amphipathicity': amphipathicity,
        'Length': length,
        'Isoelectric_Point': isoelectric_point,
        'Boman_Index': boman_index,
        'Instability_Index': instability_index,
        'Aliphatic_Index': aliphatic_index,
        'Aromaticity': aromaticity
    })

    # Add AAC features
    aac_df = pd.DataFrame(aac_list).add_prefix("AAC_")
    output_df = pd.concat([output_df, aac_df], axis=1)
    return output_df

# ======== Extract Features ========
features_df = extract_features(sequences)
features_df["Label"] = data["Label"].astype(int)

# ======== Train/Test Split ========
X = features_df.drop(columns=["Label"])
y = features_df["Label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ======== Train XGBoost ========
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)
xgb_model.fit(X_train, y_train)

# ======== Evaluate Model ========
xgb_pred = xgb_model.predict(X_test)
xgb_prob = xgb_model.predict_proba(X_test)[:,1]
print("\n🔹 XGBoost Results")
print(classification_report(y_test, xgb_pred))
print("ROC-AUC:", roc_auc_score(y_test, xgb_prob))

# Save model and feature columns
joblib.dump(xgb_model, "xgb_amp_model.pkl")
joblib.dump(list(X.columns), "feature_columns.pkl")

# ======== Predict New Sequence ========
def predict_new_sequence(seq, model, feature_columns):
    seq_clean = "".join([aa for aa in seq.upper() if aa in valid_aas])
    features = extract_features([seq_clean]).iloc[0].to_dict()
    f = pd.DataFrame([features])[feature_columns]
    prob = model.predict_proba(f)[:,1][0]
    label = "AMP" if prob >= 0.5 else "Non-AMP"
    return label, prob, features

# ======== SHAP Explanation ========
def explain_shap(model, X_sample):
    import shap
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_sample)
    print("\n🔹 SHAP Waterfall Plot")
    shap.plots.waterfall(shap_values[0])
    print("\n🔹 SHAP Feature Importance Bar")
    shap.plots.bar(shap_values)

# ======== User Input ========
sequence_input = input("\nEnter peptide sequence: ").strip()
if sequence_input:
    feature_columns = joblib.load("feature_columns.pkl")
    label, prob, features = predict_new_sequence(sequence_input, xgb_model, feature_columns)

    print("\n🔹 Prediction")
    print(f"Sequence: {sequence_input}")
    print(f"Prediction: {label}, Probability: {prob:.3f}")

    print("\n🔹 Extracted Features:")
    for k,v in features.items():
        print(f"{k}: {v}")

    X_sample = pd.DataFrame([features])[feature_columns]
    explain_shap(xgb_model, X_sample)
else:
    print("⚠ No valid sequence entered.")