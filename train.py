import os
os.environ["TF_USE_LEGACY_KERAS"] = "True"
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
from scipy.special import softmax

# ---------------- CONFIG ----------------
DATA_PATH = "blooms_taxonomy_dataset.csv"
MODEL_DIR = "bloom_bert_model"
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 5e-5
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------- LOAD DATA ----------------
try:
    raw_df = pd.read_csv(DATA_PATH, encoding="utf-8")
except UnicodeDecodeError:
    raw_df = pd.read_csv(DATA_PATH, encoding="latin-1")

raw_df = raw_df.dropna(subset=["Question", "blooms_level"])
raw_df["Question"] = raw_df["Question"].astype(str).str.strip()
raw_df["blooms_level"] = raw_df["blooms_level"].astype(str).str.strip()

# ---------------- LABEL ENCODING ----------------
label_encoder = LabelEncoder()
raw_df["label_enc"] = label_encoder.fit_transform(raw_df["blooms_level"])
num_labels = len(label_encoder.classes_)
print(f"Detected {num_labels} classes: {list(label_encoder.classes_)}")

# ---------------- TRAIN TEST SPLIT ----------------
X = raw_df["Question"].values
y = raw_df["label_enc"].values
X_train_texts, X_test_texts, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

# ---------------- TOKENIZER ----------------
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

train_encodings = tokenizer(
    list(X_train_texts),
    truncation=True,
    padding="max_length",
    max_length=MAX_LEN,
)

test_encodings = tokenizer(
    list(X_test_texts),
    truncation=True,
    padding="max_length",
    max_length=MAX_LEN,
)

def convert_to_tf_dataset(encodings, labels=None):
    inputs = {
        "input_ids": tf.constant(encodings["input_ids"], dtype=tf.int32),
        "attention_mask": tf.constant(encodings["attention_mask"], dtype=tf.int32),
    }
    if labels is None:
        dataset = tf.data.Dataset.from_tensor_slices(inputs)
    else:
        dataset = tf.data.Dataset.from_tensor_slices((inputs, tf.constant(labels)))
    return dataset.batch(BATCH_SIZE)

train_dataset = convert_to_tf_dataset(train_encodings, y_train)
val_dataset = convert_to_tf_dataset(test_encodings, y_test)

# ---------------- BERT MODEL ----------------
model = TFDistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=num_labels,
)

optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=LEARNING_RATE)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

history = model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS)

# ---------------- SAVE BERT ----------------
model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)

with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "wb") as f:
    pickle.dump(label_encoder, f)

# ---------------- CLASSICAL MODELS ----------------
vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train_texts)
X_test_tfidf = vectorizer.transform(X_test_texts)

ml_models = {
    "LogisticRegression": LogisticRegression(max_iter=200, random_state=SEED),
    "MultinomialNB": MultinomialNB(),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=SEED),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=200, random_state=SEED),
    "SVM": SVC(probability=True, random_state=SEED),
    "KNN": KNeighborsClassifier(),
}

ml_results = {}

for name, clf in ml_models.items():
    clf.fit(X_train_tfidf, y_train)
    y_pred = clf.predict(X_test_tfidf)

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    ml_results[name] = {
        "model": clf,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }

# ---------------- SAVE CLASSICAL MODELS ----------------
with open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
    pickle.dump(vectorizer, f)

with open(os.path.join(MODEL_DIR, "ml_models.pkl"), "wb") as f:
    pickle.dump({k: v["model"] for k, v in ml_results.items()}, f)

print("✅ Training complete. Models saved in bloom_bert_model/")