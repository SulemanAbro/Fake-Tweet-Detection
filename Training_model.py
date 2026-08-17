import pandas as pd
import re
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


# ---------------- CLEAN TEXT ----------------
def clean_text(text):
    text = str(text).lower()
    # Remove source names like (Reuters) - aksar true news mein ye hota hai
    text = re.sub(r'\(reuters\)|reuters', '', text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------- LOAD & PROCESS DATA ----------------
def load_data(files, label):
    df_list = []
    for file in files:
        try:
            df = pd.read_csv(file, encoding='latin1', on_bad_lines='skip', engine='python')
            # Dataset specificity handle karne ke liye
            df['content'] = df.get('title', '').fillna('') + " " + df.get('text', '').fillna('')
            df['text'] = df['content'].apply(clean_text)
            df['label'] = label
            df = df[df['text'].str.len() > 20]  # Short snippets hata dein
            df_list.append(df[['text', 'label']])
            print(f"Loaded {file} with {len(df)} rows")
        except Exception as e:
            print(f"Error loading {file}: {e}")

    return pd.concat(df_list, ignore_index=True)


# ---------------- PREPARE DATA ----------------
true_files = ["true12.csv", "true1234.csv"]
fake_files = ["Fake.csv", "fake1.csv"]

true_data = load_data(true_files, 1)
fake_data = load_data(fake_files, 0)

# 50/50 Perfect Balance
min_size = min(len(true_data), len(fake_data))
data = pd.concat([true_data.sample(min_size, random_state=42),
                  fake_data.sample(min_size, random_state=42)], ignore_index=True)

data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# ---------------- TRAINING ----------------
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['label'], test_size=0.2, random_state=42,
    stratify=data['label']
)

# TF-IDF Improvements
vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words='english',
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.8  # Ignore words appearing in >80% of documents
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Naive Bayes
model = MultinomialNB(alpha=1.0, class_prior=[0.5, 0.5])
model.fit(X_train_vec, y_train)


# ---------------- PREDICTION LOGIC ----------------
def hybrid_predict(text):
    cleaned = clean_text(text)

    if len(cleaned.split()) < 4:
        return "NEUTRAL/TOO SHORT", 0.0, "Input is too short."

    vec = vectorizer.transform([cleaned])
    if vec.sum() == 0:
        return "UNCERTAIN", 0.0, "No keywords matched."

    # Model probability
    probs = model.predict_proba(vec)[0]
    fake_prob, real_prob = probs[0], probs[1]

    # Dynamic Threshold
    diff = abs(real_prob - fake_prob)

    # Increase "Uncertain" zone to avoid false labels on random text
    if diff < 0.25:
        return "UNCERTAIN", round(diff * 100, 2), "Mixed signals from content."

    label = "REAL" if real_prob > fake_prob else "FAKE"
    return label, round(max(probs) * 100, 2), "Successful detection."


# Save
with open("model.pkl", "wb") as f: pickle.dump(model, f)
with open("vectorizer.pkl", "wb") as f: pickle.dump(vectorizer, f)

# Performance metrics
y_pred = model.predict(X_test_vec)
print("\n--- Model Evaluation ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))