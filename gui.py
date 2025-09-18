import tkinter as tk
from tkinter import messagebox
import joblib
import pandas as pd
from model import predict_new_sequence

# Load model & features
model = joblib.load("xgb_amp_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

def predict():
    seq = entry.get().strip()
    if not seq:
        messagebox.showwarning("Warning", "Enter a valid sequence!")
        return
    label, prob, features = predict_new_sequence(seq, model, feature_columns)
    result.set(f"Prediction: {label} (Prob: {prob:.3f})")

# GUI setup
root = tk.Tk()
root.title("AMP Prediction Tool")

tk.Label(root, text="Enter Peptide Sequence:").pack(pady=5)
entry = tk.Entry(root, width=50)
entry.pack(pady=5)

tk.Button(root, text="Predict", command=predict).pack(pady=10)

result = tk.StringVar()
tk.Label(root, textvariable=result, font=("Arial", 12), fg="blue").pack(pady=10)

root.mainloop()
