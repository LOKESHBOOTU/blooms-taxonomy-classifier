import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import gradio as gr
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
from scipy.special import softmax

MODEL_DIR = "bloom_bert_model"

# -------------------------------
# Load Saved Models
# -------------------------------

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
bert_model = TFDistilBertForSequenceClassification.from_pretrained(MODEL_DIR)

with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
    label_encoder = pickle.load(f)

with open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
    vectorizer = pickle.load(f)

with open(os.path.join(MODEL_DIR, "ml_models.pkl"), "rb") as f:
    ml_models = pickle.load(f)

# -------------------------------
# Prediction Logic
# -------------------------------

def predict_all_models(question, true_label=None):
    preds = {}
    comparison = {}

    # ---------------- BERT ----------------
    inputs = tokenizer(question, return_tensors="tf", truncation=True, padding=True)
    logits = bert_model(inputs).logits.numpy()
    probs = softmax(logits, axis=1)
    pred_id = np.argmax(probs)
    bert_label = label_encoder.inverse_transform([pred_id])[0]
    bert_conf = float(np.max(probs))

    preds["DistilBERT"] = {
        "label": bert_label,
        "confidence": bert_conf,
        "metrics": {"accuracy": "-", "precision": "-", "recall": "-", "f1": "-"}
    }

    if true_label and true_label.strip() != "":
        comparison["DistilBERT"] = 1 if bert_label == true_label else 0

    # ---------------- Classical Models ----------------
    question_tfidf = vectorizer.transform([question])

    for name, model in ml_models.items():
        prob = model.predict_proba(question_tfidf)[0]
        pred_id = np.argmax(prob)
        label = label_encoder.inverse_transform([pred_id])[0]
        confidence = float(np.max(prob))

        preds[name] = {
            "label": label,
            "confidence": confidence,
            "metrics": {"accuracy": "-", "precision": "-", "recall": "-", "f1": "-"}
        }

        if true_label and true_label.strip() != "":
            comparison[name] = 1 if label == true_label else 0

    return preds, comparison

# -------------------------------
# Confidence Bar Chart
# -------------------------------

def plot_confidence_bar(preds):
    models = list(preds.keys())
    confidences = [info['confidence'] for info in preds.values()]

    fig, ax = plt.subplots(figsize=(8,6))
    bars = ax.bar(models, confidences)

    for bar, conf in zip(bars, confidences):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f"{conf:.4f}",
                ha="center", va="bottom")

    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Confidence")
    ax.set_title("Confidence Comparison")
    plt.xticks(rotation=30)

    return fig

# -------------------------------
# Gradio Interface
# -------------------------------

label_choices = [""] + list(label_encoder.classes_)

def gradio_interface(question, true_label_choice):
    preds, comparison = predict_all_models(question, true_label_choice)

    bert_info = preds["DistilBERT"]

    summary = (
        f"Predicted Bloom's Level (DistilBERT): {bert_info['label']}\n"
        f"Confidence: {bert_info['confidence']:.4f}"
    )

    if true_label_choice and true_label_choice.strip() != "":
        summary += f"\nTrue Bloom's Level: {true_label_choice}"

    # -------- Model Prediction Table --------
    pred_rows = []
    for model_name, info in preds.items():
        row = {
            "Model": model_name,
            "Label": info['label'],
            "Confidence": f"{info['confidence']:.4f}"
        }

        if true_label_choice and true_label_choice.strip() != "":
            row["Correct?"] = "✅" if comparison.get(model_name) == 1 else "❌"

        pred_rows.append(row)

    df_preds = pd.DataFrame(pred_rows)

    fig_conf = plot_confidence_bar(preds)

    return summary, df_preds, fig_conf

# -------------------------------
# Launch App
# -------------------------------

interface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Textbox(label="Enter a Question", lines=3),
        gr.Dropdown(
            label="(Optional) True Bloom's Level",
            choices=label_choices,
            value=""
        )
    ],
    outputs=[
        gr.Textbox(label="DistilBERT Prediction Summary"),
        gr.Dataframe(label="Model-wise Predictions"),
        gr.Plot(label="Confidence Comparison Chart")
    ],
    title="📘 Bloom's Taxonomy Classifier",
    description="Enter a question to classify its Bloom's level."
)

if __name__ == "__main__":
    interface.launch()