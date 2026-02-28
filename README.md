<img width="1920" height="1187" alt="image" src="https://github.com/user-attachments/assets/20ea6e22-d7f0-45c0-b600-e4c1b9496041" /># Bloom's Taxonomy Classifier

🚀 Live Demo:
[Click Here to Launch Demo](https://huggingface.co/spaces/Lokeshlokey/blooms-taxonomy-classifier-final)

# 📘 Bloom's Taxonomy Classifier

An AI-powered web application that classifies questions into Bloom's Taxonomy levels using:

- DistilBERT (Deep Learning)
- Traditional ML models (Logistic Regression, SVM, Random Forest, etc.)
- Gradio Web Interface

## 🧠 What is Bloom’s Taxonomy?

Bloom’s Taxonomy is a framework used to classify educational learning objectives into levels of cognitive complexity.

It helps educators design exams, assignments, and assessments based on difficulty and depth of understanding.

---

## 📊 Bloom’s Taxonomy Hierarchy

Bloom’s Revised Taxonomy consists of **six levels**, arranged from lower-order thinking to higher-order thinking:

[Uploading image.png…]()



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
