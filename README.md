# Bloom's Taxonomy Classifier

🚀 Live Demo:
[Click Here to Launch App](https://huggingface.co/spaces/Lokeshlokey/blooms-taxonomy-classifier-final)

# 📘 Bloom's Taxonomy Classifier

This project classifies questions into Bloom's Taxonomy levels using:

- DistilBERT (Deep Learning)
- Traditional ML models (Logistic Regression, SVM, Random Forest, etc.)
- Gradio Web Interface

## 🚀 How to Run

### 1️⃣ Create Virtual Environment
python -m venv venv

### 2️⃣ Activate
Windows:
venv\Scripts\activate

### 3️⃣ Install Requirements
pip install -r requirements.txt

### 4️⃣ Train Model
python train.py

### 5️⃣ Run App
python app.py

---

## 📊 Models Used
- DistilBERT
- Logistic Regression
- Naive Bayes
- Random Forest
- Gradient Boosting
- SVM
- KNN

---

## 📂 Project Structure
- train.py → Training pipeline
- app.py → Gradio inference app
- bloom_bert_model/ → Saved model files (generated after training)
