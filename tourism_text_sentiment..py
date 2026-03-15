import pandas as pd
import numpy as np
import re
import string
import torch
import torch.nn as nn
import torch.optim as optim

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import shap
from lime.lime_text import LimeTextExplainer


print("Loading tourism reviews dataset")

dataset = load_dataset("amazon_polarity", split="train[:6000]")

df = pd.DataFrame(dataset)

print("Dataset shape:", df.shape)


# -----------------------------
# TEXT CLEANING
# -----------------------------

def clean_text(text):

    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"http\S+", "", text)

    text = text.lower()

    text = text.translate(str.maketrans("", "", string.punctuation))

    return text


df["clean_text"] = df["content"].apply(clean_text)


# -----------------------------
# NORMALIZATION
# -----------------------------

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def normalize(text):

    words = text.split()

    words = [w for w in words if w not in stop_words]

    words = [lemmatizer.lemmatize(w) for w in words]

    return " ".join(words)


df["normalized"] = df["clean_text"].apply(normalize)


# -----------------------------
# TF-IDF FEATURES
# -----------------------------

tfidf = TfidfVectorizer(max_features=2000)

X = tfidf.fit_transform(df["normalized"]).toarray()

y = df["label"].values

print("TFIDF shape:", X.shape)


# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# convert to tensor
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)

y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)


# -----------------------------
# LSTM MODEL
# -----------------------------

class LSTMModel(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):

        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        out, _ = self.lstm(x)

        out = out[:, -1, :]

        out = self.fc(out)

        return out


model = LSTMModel(X_train.shape[2], 64, 2)


# -----------------------------
# TRAIN MODEL
# -----------------------------

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 5

for epoch in range(epochs):

    outputs = model(X_train)

    loss = criterion(outputs, y_train)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    print("Epoch", epoch + 1, "Loss:", loss.item())


# -----------------------------
# EVALUATION
# -----------------------------

with torch.no_grad():

    preds = model(X_test)

    predicted = torch.argmax(preds, dim=1)

accuracy = accuracy_score(y_test, predicted)

precision = precision_score(y_test, predicted, zero_division=0)

recall = recall_score(y_test, predicted, zero_division=0)

f1 = f1_score(y_test, predicted, zero_division=0)

print("\nModel Performance")

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)


# -----------------------------
# SHAP EXPLANATION (WORKING)
# -----------------------------

print("\nRunning SHAP explanation")

def predict_fn(data):

    data = torch.tensor(data, dtype=torch.float32).unsqueeze(1)

    with torch.no_grad():

        outputs = model(data)

        probs = torch.softmax(outputs, dim=1)

    return probs.numpy()


background = X[:50]

explainer = shap.KernelExplainer(predict_fn, background)

shap_values = explainer.shap_values(X[:10])

shap.summary_plot(shap_values, X[:10])


# -----------------------------
# LIME EXPLANATION
# -----------------------------

print("\nRunning LIME explanation")

explainer = LimeTextExplainer(class_names=["Negative", "Positive"])

idx = 5

exp = explainer.explain_instance(
    df["normalized"].iloc[idx],
    lambda x: np.array([[0.5, 0.5]] * len(x)),
    num_features=10
)

exp.save_to_file("lime_explanation.html")

print("LIME explanation saved as lime_explanation.html")