# pandas loads our test CSV file into a DataFrame
# Analogy: pandas is like Excel for Python — rows, columns, and easy filtering
import pandas as pd

# numpy is our numerical computing library
# Analogy: numpy is like a calculator that works on entire lists at once
import numpy as np

# torch is PyTorch — we need it to load and run the DAN model
import torch

# nn is PyTorch's neural network module — we need it to rebuild the DAN architecture
# We have to recreate the exact same architecture we used in Script 04
# so we can load the saved weights back into it
# Analogy: you need the same filing cabinet to put the files back into —
# you can't load files into a different shaped cabinet
import torch.nn as nn

# DataLoader and Dataset let us batch the test data for the DAN model
from torch.utils.data import DataLoader, Dataset

# joblib loads the saved TF-IDF model, vectorizer, and DAN vocabulary from disk
import joblib

# matplotlib creates our comparison bar chart
# Analogy: matplotlib is like a graph paper and pencil — it draws charts from numbers
import matplotlib.pyplot as plt

# os lets us create folders and build file paths
import os


# This script is the final script of the project
# It loads both trained models, runs them on the same test set,
# and produces a side-by-side comparison of their performance
# Analogy: like a final judging panel that takes two contestants
# and puts them through the exact same challenge to compare them fairly


# Force CPU — same as Script 04
device = torch.device("cpu")


class DANClassifier(nn.Module):
    # We have to redefine the exact same DAN architecture from Script 04
    # so we can load the saved weights back into it
    # torch.save() only saves the weights, not the blueprint —
    # so we need to provide the blueprint again here
    # Analogy: the saved file is like a filled-in form —
    # but you still need the blank form template to know where each answer goes

    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(DANClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size + 2, embed_dim, padding_idx=0)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        averaged = embedded.mean(dim=1)
        out = self.relu(self.fc1(averaged))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        return self.sigmoid(out).squeeze()


class TweetDataset(Dataset):
    # Same TweetDataset class from Script 04 — wraps encoded tweets for the DataLoader
    # Analogy: same standardized containers for the factory conveyor belt

    def __init__(self, X, y):
        self.X = torch.tensor(np.array(X), dtype=torch.long)
        self.y = torch.tensor(np.array(y), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_test_data():
    # This function's job is to load the test set that Script 02 saved
    # Both models will be evaluated on this exact same set of tweets
    # so the comparison is completely fair
    # Analogy: both contestants sit the same exam on the same day

    print("Loading test data...")
    test_df = pd.read_csv("data/test.csv")
    test_df = test_df.dropna(subset=["text", "sentiment"])
    test_df["sentiment"] = test_df["sentiment"].astype(int)
    print(f"Test set: {test_df.shape[0]:,} tweets")
    return test_df


def load_tfidf_model():
    # This function's job is to load the saved TF-IDF model and vectorizer from disk
    # joblib.load() is the reverse of joblib.dump() — it reads the file and
    # reconstructs the Python object exactly as it was when we saved it
    # Analogy: opening the filing cabinet and pulling out the stored notes

    print("\nLoading TF-IDF model...")
    model = joblib.load("models/tfidf_model.joblib")
    vectorizer = joblib.load("models/tfidf_vectorizer.joblib")
    print("TF-IDF model loaded")
    return model, vectorizer


def load_dan_model():
    # This function's job is to load the saved DAN model and vocabulary from disk
    # We have to rebuild the architecture first, then load the weights into it
    # Analogy: assembling the filing cabinet first, then putting the files back in

    print("Loading DAN model...")

    # Load the vocabulary that Script 04 saved
    word2idx = joblib.load("models/dan_vocab.joblib")

    # Rebuild the exact same architecture we used in Script 04
    model = DANClassifier(
        vocab_size=20000,
        embed_dim=128,
        hidden_dim=256
    ).to(device)

    # Load the saved weights into the architecture
    # map_location="cpu" ensures the weights load onto CPU even if they were saved on GPU
    # Analogy: putting the saved notes back into the filing cabinet
    model.load_state_dict(torch.load("models/dan_model.pt", map_location="cpu"))

    # Set to eval mode — disables dropout for consistent predictions
    # Analogy: switching from practice mode to exam mode
    model.eval()

    print("DAN model loaded")
    return model, word2idx


def evaluate_tfidf(model, vectorizer, test_df):
    # This function's job is to run the test tweets through the TF-IDF model
    # and return the accuracy
    # Analogy: contestant 1 sits the exam

    print("\n--- Evaluating TF-IDF + Logistic Regression ---")

    # Vectorize the test tweets using the same vectorizer from training
    X_test_vec = vectorizer.transform(test_df["text"])
    y_test = test_df["sentiment"]

    # Generate predictions
    y_pred = model.predict(X_test_vec)

    # Calculate accuracy
    correct = (y_pred == y_test).sum()
    total = len(y_test)
    accuracy = correct / total

    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    return accuracy


def encode_tweets(texts, word2idx, max_length=50):
    # Same encode_tweets function from Script 04
    # Converts tweet text into fixed-length lists of integer IDs
    # Analogy: converting words into the same secret code we used during training

    encoded = []
    for text in texts:
        ids = [word2idx.get(word, 1) for word in str(text).split()]
        if len(ids) > max_length:
            ids = ids[:max_length]
        else:
            ids = ids + [0] * (max_length - len(ids))
        encoded.append(ids)
    return encoded


def evaluate_dan(model, word2idx, test_df):
    # This function's job is to run the test tweets through the DAN model
    # and return the accuracy
    # Analogy: contestant 2 sits the same exam

    print("\n--- Evaluating Deep Averaging Network ---")

    X_test = encode_tweets(test_df["text"], word2idx)
    y_test = test_df["sentiment"].tolist()

    dataset = TweetDataset(X_test, y_test)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            predictions = model(X_batch)
            predicted_labels = (predictions >= 0.5).float()
            correct += (predicted_labels == y_batch).sum().item()
            total += y_batch.size(0)

    accuracy = correct / total
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    return accuracy


def plot_comparison(tfidf_accuracy, dan_accuracy):
    # This function's job is to create a bar chart comparing both model accuracies
    # Analogy: the scoreboard at the end of a competition

    print("\nGenerating comparison chart...")

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Define the models and their accuracies
    models = ["TF-IDF +\nLogistic Regression", "Deep Averaging\nNetwork (DAN)"]
    accuracies = [tfidf_accuracy * 100, dan_accuracy * 100]

    # Define colors for each bar
    colors = ["#2196F3", "#4CAF50"]

    # Create the bars
    bars = ax.bar(models, accuracies, color=colors, width=0.5)

    # Add the accuracy number on top of each bar
    # ha="center" centers the text horizontally above the bar
    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{acc:.2f}%",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold"
        )

    # Add a horizontal dashed line showing the random baseline (50%)
    # This is the "do nothing" baseline — a model that just guesses randomly
    ax.axhline(y=50, color="red", linestyle="--", linewidth=1.5, label="Random baseline (50%)")

    # Label the axes and title
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Model Comparison: TF-IDF vs DAN\nTwitter Sentiment Analysis", fontsize=14)

    # Set the y-axis range so differences are visible but not exaggerated
    ax.set_ylim(40, 90)

    ax.legend()
    plt.tight_layout()

    # Save the chart to the models/ folder
    os.makedirs("models", exist_ok=True)
    plt.savefig("models/comparison_chart.png", dpi=150)
    print("Chart saved to models/comparison_chart.png")
    plt.show()


def predict_examples(tfidf_model, vectorizer, dan_model, word2idx):
    # This function's job is to run a handful of example tweets through both models
    # so we can see where they agree and where they differ
    # Analogy: giving both contestants a few bonus questions after the exam
    # and comparing their answers side by side

    print("\n--- Example Predictions ---")

    example_tweets = [
        "I love this so much it made my day",
        "this is the worst thing that has ever happened to me",
        "not bad actually kind of enjoyed it",
        "so tired and stressed out today",
        "just had the most amazing coffee ever",
        "my flight got cancelled and I missed the concert",
        "feeling okay I guess nothing special",
        "absolutely devastated right now"
    ]

    print(f"\n{'Tweet':<45} {'TF-IDF':>10} {'DAN':>10}")
    print("-" * 67)

    for tweet in example_tweets:
        # TF-IDF prediction
        # We have to clean the tweet the same way Script 02 cleaned the training data
        tweet_vec = vectorizer.transform([tweet])
        tfidf_pred = tfidf_model.predict(tweet_vec)[0]
        tfidf_label = "Positive" if tfidf_pred == 1 else "Negative"

        # DAN prediction
        encoded = encode_tweets([tweet], word2idx)
        tensor = torch.tensor(np.array(encoded), dtype=torch.long).to(device)
        with torch.no_grad():
            dan_pred = dan_model(tensor).item()
        dan_label = "Positive" if dan_pred >= 0.5 else "Negative"

        # Flag tweets where the two models disagree
        flag = "  <-- disagree" if tfidf_label != dan_label else ""

        print(f"{tweet:<45} {tfidf_label:>10} {dan_label:>10}{flag}")


# This block only runs if you execute this file directly (python 05_compare.py)
# If another script imports this file, this block is skipped
if __name__ == "__main__":
    test_df = load_test_data()

    tfidf_model, vectorizer = load_tfidf_model()
    dan_model, word2idx = load_dan_model()

    tfidf_accuracy = evaluate_tfidf(tfidf_model, vectorizer, test_df)
    dan_accuracy = evaluate_dan(dan_model, word2idx, test_df)

    plot_comparison(tfidf_accuracy, dan_accuracy)
    predict_examples(tfidf_model, vectorizer, dan_model, word2idx)