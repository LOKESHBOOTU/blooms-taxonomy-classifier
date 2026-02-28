import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
import tensorflow as tf
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
from scipy.special import softmax
from huggingface_hub import hf_hub_download

# -------------------------------
# CONFIG
# -------------------------------

MODEL_NAME = "Lokeshlokey/blooms-taxonomy-distilbert"

# -------------------------------
# Load Model & Artifacts
# -------------------------------

print("🔄 Loading tokenizer...")
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

print("🔄 Loading DistilBERT model...")
bert_model = TFDistilBertForSequenceClassification.from_pretrained(MODEL_NAME)

print("🔄 Downloading classical ML artifacts...")
label_path = hf_hub_download(repo_id=MODEL_NAME, filename="label_encoder.pkl")
vectorizer_path = hf_hub_download(repo_id=MODEL_NAME, filename="tfidf_vectorizer.pkl")
ml_models_path = hf_hub_download(repo_id=MODEL_NAME, filename="ml_models.pkl")

with open(label_path, "rb") as f:
    label_encoder = pickle.load(f)

with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)

with open(ml_models_path, "rb") as f:
    ml_models = pickle.load(f)

print("✅ All models loaded successfully!")

# -------------------------------
# Prediction Logic
# -------------------------------

def predict_all_models(question, true_label=None):

    if not question or not question.strip():
        return "⚠ Please enter a valid question.", pd.DataFrame(), None

    preds = {}
    comparison = {}

    # -------- DistilBERT --------
    inputs = tokenizer(
        question,
        return_tensors="tf",
        truncation=True,
        padding=True,
        max_length=128
    )

    logits = bert_model(**inputs).logits.numpy()
    probs = softmax(logits, axis=1)

    pred_id = np.argmax(probs)
    bert_label = label_encoder.inverse_transform([pred_id])[0]
    bert_conf = float(np.max(probs))

    preds["DistilBERT"] = {
        "label": bert_label,
        "confidence": bert_conf
    }

    if true_label:
        comparison["DistilBERT"] = 1 if bert_label == true_label else 0

    # -------- Classical ML Models --------
    question_tfidf = vectorizer.transform([question])

    for name, model in ml_models.items():
        prob = model.predict_proba(question_tfidf)[0]
        pred_id = np.argmax(prob)
        label = label_encoder.inverse_transform([pred_id])[0]
        confidence = float(np.max(prob))

        preds[name] = {
            "label": label,
            "confidence": confidence
        }

        if true_label:
            comparison[name] = 1 if label == true_label else 0

    # -------- Summary --------
    summary = (
        f"Predicted Bloom's Level (DistilBERT): {bert_label}\n"
        f"Confidence: {bert_conf:.4f}"
    )

    if true_label:
        summary += f"\nTrue Bloom's Level: {true_label}"

    # -------- Table --------
    rows = []
    for model_name, info in preds.items():
        row = {
            "Model": model_name,
            "Label": info["label"],
            "Confidence": f"{info['confidence']:.4f}"
        }

        if true_label:
            row["Correct?"] = "✅" if comparison.get(model_name) == 1 else "❌"

        rows.append(row)

    df = pd.DataFrame(rows)

    # -------- Confidence Chart --------
    fig, ax = plt.subplots(figsize=(8, 5))
    models = list(preds.keys())
    confidences = [info["confidence"] for info in preds.values()]

    bars = ax.bar(models, confidences)

    for bar, conf in zip(bars, confidences):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01,
            f"{conf:.4f}",
            ha="center"
        )

    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Confidence")
    ax.set_title("Confidence Comparison")
    plt.xticks(rotation=30)

    return summary, df, fig

# -------------------------------
# Gradio UI
# -------------------------------

label_choices = list(label_encoder.classes_)

interface = gr.Interface(
    fn=predict_all_models,
    inputs=[
        gr.Textbox(label="Enter a Question", lines=3),
        gr.Dropdown(
            label="(Optional) True Bloom's Level",
            choices=label_choices,
            value=None
        )
    ],
    outputs=[
        gr.Textbox(label="DistilBERT Prediction Summary"),
        gr.Dataframe(label="Model-wise Predictions"),
        gr.Plot(label="Confidence Comparison Chart")
    ],
    title="📘 Bloom's Taxonomy Classifier",
    description="Deep Learning + Traditional ML comparison using trained Bloom's Taxonomy model."
)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)
