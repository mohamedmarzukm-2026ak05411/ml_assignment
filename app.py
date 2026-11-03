# ML Assignment 2 - Streamlit app
# loads the saved models from the model folder and tests them on the uploaded csv
# (no training here, only prediction)

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score, roc_curve)

APP_DIR = Path(__file__).parent
MODEL_DIR = APP_DIR / "model"
info = json.loads((MODEL_DIR / "model_info.json").read_text())
SENSORS = info["features"]
TARGET = info["target"]
CLASSES = info["classes"]
MODEL_FILES = info["models"]
# short names for the plots
SHORT_NAMES = {"Move-Forward": "Forward", "Sharp-Right-Turn": "Sharp right",
               "Slight-Right-Turn": "Slight right", "Slight-Left-Turn": "Slight left"}

st.set_page_config(page_title="ML Assignment 2", layout="wide")


@st.cache_resource
def get_model(name):
    return joblib.load(MODEL_DIR / f"{MODEL_FILES[name]}.joblib")


# same 6 metrics as the notebook (macro average, AUC one-vs-rest)
def get_metrics(y_true, y_pred, proba, model_classes):
    try:
        auc = roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=model_classes)
    except ValueError:
        auc = float("nan")  # if the file has only one class, AUC can't be calculated
    return {"Accuracy": accuracy_score(y_true, y_pred),
            "AUC": auc,
            "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "MCC": matthews_corrcoef(y_true, y_pred)}


# ---- sidebar ----
st.sidebar.header("Settings")
uploaded_file = st.sidebar.file_uploader("Upload test data (CSV)", type="csv")
sample_file = APP_DIR / "test_data.csv"
if sample_file.exists():
    st.sidebar.download_button("Download sample test_data.csv", sample_file.read_bytes(), "test_data.csv", "text/csv")
model_name = st.sidebar.selectbox("Choose a model", list(MODEL_FILES))
st.sidebar.caption("Precision, Recall, F1 = macro average. AUC = one-vs-rest (macro).")

st.title("Wall Following Robot - Movement Prediction")
st.write("ML Assignment 2. Dataset: UCI Wall-Following Robot Navigation (24 ultrasound sensors). "
         "The model predicts the robot's move (Move-Forward, Sharp-Right-Turn, Slight-Right-Turn "
         "or Slight-Left-Turn) from the 24 sensor readings.")

if uploaded_file is None:
    st.info("Please upload a CSV file from the sidebar. You can use the sample test_data.csv "
            "(last 20% of the data, not used for training).")
    st.stop()

# ---- check the uploaded file ----
data = pd.read_csv(uploaded_file)
missing = [c for c in SENSORS + [TARGET] if c not in data.columns]
if missing:
    st.error("These columns are missing from the file: " + ", ".join(missing))
    st.stop()
unknown = sorted(set(data[TARGET].astype(str)) - set(CLASSES))
if unknown:
    st.error(f"Unknown values in '{TARGET}': {unknown}. Expected one of {CLASSES}.")
    st.stop()
if data[SENSORS].isna().any().any():
    st.warning("Some sensor values are missing, they will be filled with the training mean.")

# ---- predict with the selected model ----
X = data[SENSORS]
y = data[TARGET].astype(str)
model = get_model(model_name)
pred = model.predict(X)
proba = model.predict_proba(X)
results = get_metrics(y, pred, proba, model.classes_)

tab1, tab2, tab3 = st.tabs(["Model results", "Compare models", "Single row"])

with tab1:
    st.subheader("Results for " + model_name)
    metric_cols = st.columns(6)
    for i, (metric, value) in enumerate(results.items()):
        if np.isnan(value):
            metric_cols[i].metric(metric, "n/a")
        else:
            metric_cols[i].metric(metric, f"{value:.4f}")

    col1, col2 = st.columns(2)
    labels = [SHORT_NAMES[c] for c in CLASSES]
    with col1:
        st.write("Confusion Matrix")
        cm = confusion_matrix(y, pred, labels=CLASSES)
        fig, ax = plt.subplots(figsize=(5, 4.2))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", cbar=False, ax=ax,
                    xticklabels=labels, yticklabels=labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
    with col2:
        st.write("ROC Curve (one-vs-rest)")
        fig, ax = plt.subplots(figsize=(5, 4.2))
        for i, cls in enumerate(model.classes_):
            # need both positive and negative rows to draw the curve
            if (y == cls).any() and (y != cls).any():
                fpr, tpr, _ = roc_curve(y == cls, proba[:, i])
                ax.plot(fpr, tpr, label=SHORT_NAMES.get(cls, cls))
        ax.plot([0, 1], [0, 1], "--", color="grey")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right", fontsize=8)
        st.pyplot(fig)

    st.write("Classification Report")
    report = classification_report(y, pred, labels=CLASSES, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T.round(4), width="stretch")

with tab2:
    st.subheader("All 5 models on the uploaded file")
    rows = []
    for name in MODEL_FILES:
        mdl = get_model(name)
        rows.append({"Model": name, **get_metrics(y, mdl.predict(X), mdl.predict_proba(X), mdl.classes_)})
    table = pd.DataFrame(rows).set_index("Model")
    st.caption("Best value in each column is highlighted.")
    st.dataframe(table.style.format("{:.4f}").highlight_max(axis=0, color="#ffe0b2"), width="stretch")
    fig, ax = plt.subplots(figsize=(9, 3.5))
    table[["Accuracy", "F1", "MCC"]].plot(kind="bar", ax=ax, rot=0)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    st.pyplot(fig)

with tab3:
    st.subheader("Sensor readings for one row")
    row_no = int(st.number_input("Row number", 0, len(data) - 1, 0))
    row = data.iloc[row_no]
    st.write(f"Actual: {row[TARGET]}, Predicted ({model_name}): {pred[row_no]}")
    # sensors are 15 degrees apart, sensor 1 is at the front (180 deg)
    angles = np.deg2rad(180 + 15 * np.arange(24))
    dist = row[SENSORS].astype(float).values
    fig = plt.figure(figsize=(4.5, 4.5))
    ax = fig.add_subplot(projection="polar")
    ax.set_theta_zero_location("S")   # so the front of the robot is at the top
    ax.plot(np.append(angles, angles[0]), np.append(dist, dist[0]), marker="o", markersize=3)
    ax.fill(angles, dist, alpha=0.15)
    ax.set_rmax(5.2)
    ax.set_title("Sensor distances (m)", fontsize=9)
    st.pyplot(fig)
    st.caption("5 m means the sensor did not detect anything.")

# ---- download predictions ----
out = data.copy()
out["Predicted"] = pred
out["Probability"] = proba.max(axis=1).round(4)
st.sidebar.download_button("Download predictions", out.to_csv(index=False).encode(),
                           "predictions_" + MODEL_FILES[model_name] + ".csv", "text/csv")
