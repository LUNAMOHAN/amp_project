import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap
from model import predict_new_sequence, extract_features

# Load saved model and features
model = joblib.load("xgb_amp_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# Streamlit layout
st.set_page_config(layout="wide")

# ===== First Section: Left-Right Layout =====
col1, col2 = st.columns([1, 2])

with col1:
    st.title("🧬 AMP Prediction Tool")
    st.write("Predict whether a peptide sequence is an **Antimicrobial Peptide (AMP)** or NON AMP.")

    # Text input
    sequence_input = st.text_area(
        "Enter peptide sequence (ACDEFGHIKLMNPQRSTVWY):",
        height=150
    )

    # File upload
    uploaded_file = st.file_uploader(
        "Or upload a CSV/Excel file with a 'Sequence' column:",
        type=["csv", "xlsx"]
    )

    if st.button("Predict"):
        if sequence_input:
            # ===== Single sequence prediction =====
            label, prob, features = predict_new_sequence(sequence_input, model, feature_columns)

            st.subheader("🔹 Prediction Result (Single Sequence)")
            st.write(f"**Sequence:** {sequence_input}")
            st.write(f"**Prediction:** {label}")
            st.write(f"**Probability:** {prob:.3f}")

            st.session_state['features'] = features
            st.session_state['shap_single'] = True
            st.session_state['shap_features'] = features
            st.session_state['shap_batch'] = False  # reset batch flag

        elif uploaded_file:
            # ===== Batch prediction =====
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            # Normalize column names (force to string, lowercase)
            df.columns = [str(c).strip().lower() for c in df.columns]

            # Try to detect sequence column
            possible_cols = ["sequence", "sequences", "seq", "sequence_sample"]
            seq_col = None
            for c in df.columns:
                if any(keyword in c for keyword in possible_cols):
                    seq_col = c
                    break

            # If not found, default to 3rd column (C column = index 2)
            if seq_col is None:
                if df.shape[1] >= 3:  # at least 3 columns
                    seq_col = df.columns[2]
                    st.info(f"⚠ No column named 'Sequence' found. Using column C: **{seq_col}** as sequences.")
                else:
                    st.error("⚠ Could not find a sequence column and file does not have at least 3 columns.")
                    seq_col = None

            if seq_col is not None:
                st.subheader("🔹 Batch Prediction Results")
                results = []
                feature_list = []

                for seq in df[seq_col]:
                    label, prob, features = predict_new_sequence(seq, model, feature_columns)
                    results.append({"Sequence": seq, "Prediction": label, "Probability": prob})
                    feature_list.append(features)

                results_df = pd.DataFrame(results)
                st.dataframe(results_df, height=400)

                # Download predictions only
                st.download_button(
                    "📥 Download Predictions",
                    results_df.to_csv(index=False),
                    "predictions.csv",
                    "text/csv"
                )

                # Store for later use
                st.session_state['shap_batch'] = True
                st.session_state['batch_features'] = feature_list
                st.session_state['batch_results'] = results
                st.session_state['shap_single'] = False  # reset single flag

        else:
            st.warning("⚠ Please enter a sequence or upload a file.")

with col2:
    # Single sequence features
    if "features" in st.session_state and st.session_state.get('shap_single', False):
        st.subheader("🔹 Extracted Features (Single Sequence)")
        features = st.session_state['features']
        df = pd.DataFrame([features])
        st.dataframe(df, height=400)

        # Feature importance visualization
        st.subheader("🔹 Top Feature Values")
        df_numeric = df.apply(pd.to_numeric, errors='coerce')
        top_features = df_numeric.T.abs().sort_values(by=0, ascending=False).iloc[:10]

        fig, ax = plt.subplots(figsize=(6, 4), facecolor="white")
        top_features.plot(kind="barh", legend=False, ax=ax)
        ax.set_title("Top 10 Extracted Feature Values")
        ax.set_xlabel("Value")
        st.pyplot(fig)

    # Batch features
    elif st.session_state.get('shap_batch', False):
        st.subheader("🔹 Extracted Features (All Sequences from File)")
        feature_list = st.session_state['batch_features']
        results = st.session_state['batch_results']

        # Features DataFrame
        features_df = pd.DataFrame(feature_list)

        # Results DataFrame
        results_df = pd.DataFrame(results)

        # Merge predictions + features
        merged_df = pd.concat([results_df, features_df], axis=1)

        # Show full feature table
        st.dataframe(merged_df, height=500)

        # Download full features + predictions
        st.download_button(
            "📥 Download Features + Predictions",
            merged_df.to_csv(index=False),
            "features_predictions.csv",
            "text/csv"
        )

# ===== Second Section: SHAP Plots in Separate Layout =====
if st.session_state.get('shap_single', False):
    st.subheader("🔹 SHAP Explanation (Single Sequence)")
    shap_left, shap_right = st.columns(2)

    features = st.session_state['shap_features']
    X_sample = pd.DataFrame([features])[feature_columns]
    explainer = shap.Explainer(model, X_sample)
    shap_values = explainer(X_sample)

    # Waterfall (left)
    fig1, ax1 = plt.subplots(facecolor="white")
    shap.plots.waterfall(shap_values[0], show=False)
    shap_left.pyplot(fig1)

    # Bar plot (right)
    fig2, ax2 = plt.subplots(facecolor="white")
    shap.plots.bar(shap_values, show=False)
    shap_right.pyplot(fig2)

if st.session_state.get('shap_batch', False):
    st.subheader("🔹 SHAP Summary (All Sequences from File)")
    X_batch = pd.DataFrame(st.session_state['batch_features'])[feature_columns]
    explainer = shap.Explainer(model, X_batch)
    shap_values = explainer(X_batch)

    fig3, ax3 = plt.subplots(facecolor="white")
    shap.summary_plot(shap_values, X_batch, show=False)
    st.pyplot(fig3)
