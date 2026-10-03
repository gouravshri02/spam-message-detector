import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

DATA_FILE = "data/spam.csv"

print("Loading dataset...")
df = pd.read_csv(DATA_FILE)

df = df.dropna()
df["label"] = df["label"].str.lower().str.strip()
df["message"] = df["message"].astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    df["message"],
    df["label"],
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)

print("Converting text into TF-IDF features...")
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=5000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

predictions = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, predictions)

print(f"\nModel Accuracy: {accuracy:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, predictions))

with open("spam_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\nModel saved as spam_model.pkl")
print("Vectorizer saved as tfidf_vectorizer.pkl")
print("Training completed successfully!")
