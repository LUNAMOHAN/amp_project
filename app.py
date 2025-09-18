import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from model import predict_new_sequence

# Load saved model and features
model = joblib.load("xgb_amp_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# Set wide layout
st.set_page_config(layout="wide")

# Columns
col1, col2 = st.columns([1, 2])  # col2 is twice as wide

with col1:
    st.title("🧬 AMP Prediction Tool")
    st.write("Predict whether a peptide sequence is an **Antimicrobial Peptide (AMP)** or not.")

    # Input
    sequence_input = st.text_area(
        "Enter peptide sequence (ACDEFGHIKLMNPQRSTVWY):",
        height=250  # only height can be set
    )

    if st.button("Predict"):
        if sequence_input:
            label, prob, features = predict_new_sequence(sequence_input, model, feature_columns)

            st.subheader("🔹 Prediction Result")
            st.write(f"**Sequence:** {sequence_input}")
            st.write(f"**Prediction:** {label}")
            st.write(f"**Probability:** {prob:.3f}")

            # Save prediction info in session state for right side
            st.session_state['features'] = features
        else:
            st.warning("⚠ Please enter a valid peptide sequence.")

with col2:
    if "features" in st.session_state:
        st.subheader("🔹 Extracted Features")
        features = st.session_state['features']
        df = pd.DataFrame([features])
        st.dataframe(df, height=400)  # Set height for dataframe

        # Feature importance visualization
        st.subheader("🔹 Feature Importance Visualization")
        df_numeric = df.apply(pd.to_numeric, errors='coerce')
        top_features = df_numeric.T.abs().sort_values(by=0, ascending=False).iloc[:10]

        fig, ax = plt.subplots(figsize=(6,4))  # Adjust chart size
        top_features.plot(kind="barh", legend=False, ax=ax)
        ax.set_title("Top 10 Extracted Feature Values")
        ax.set_xlabel("Value")
        st.pyplot(fig)
