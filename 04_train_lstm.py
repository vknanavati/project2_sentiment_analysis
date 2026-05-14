# pandas loads our cleaned train and test CSV files into DataFrames
# Analogy: pandas is like Excel for Python — rows, columns, and easy filtering
import pandas as pd

# numpy is our numerical computing library
# Analogy: numpy is like a calculator that works on entire lists at once
import numpy as np

# torch is PyTorch — the deep learning framework we use to build the neural network
# Analogy: if scikit-learn is a pre-built car, PyTorch is a garage where you build your own
import torch

# nn is PyTorch's neural network module — all the building blocks for neural networks
# Analogy: nn is like a LEGO set — it gives you all the pieces, you decide how to assemble them
import torch.nn as nn

# DataLoader batches our data and feeds it to the model during training
# Dataset is the base class we inherit from to create our own custom dataset
# Analogy: DataLoader is like a conveyor belt — feeds the model one batch at a time
from torch.utils.data import DataLoader, Dataset

# joblib saves and loads Python objects to disk efficiently
import joblib

# os lets us create folders and build file paths
import os


# This script trains a neural network on cleaned tweets using word embeddings
# Instead of reading tweets word by word like an LSTM, this model:
# 1. Converts each word into a vector of numbers (embedding)
# 2. Averages all the word vectors in a tweet into one single vector
# 3. Passes that vector through layers to predict positive or negative sentiment
# This approach is called a Deep Averaging Network (DAN)
# Analogy: instead of reading a book cover to cover (LSTM),
# this model reads all the words simultaneously and takes the average impression —
# like skimming every page at once and forming an overall feeling


# Force CPU — more stable than MPS (Apple Silicon GPU) for this model
device = torch.device("cpu")
print(f"Using device: {device}")


def load_data():
    # This function's job is to load the cleaned train and test CSV files
    # We sample 200k training tweets for faster training
    # Analogy: studying a representative sample of past exams rather than every exam ever written

    print("Loading cleaned data...")
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")

    # Sample 200k tweets from the training set
    train_df = train_df.sample(n=200000, random_state=42).reset_index(drop=True)

    # Drop any rows with missing values
    train_df = train_df.dropna(subset=["text", "sentiment"])
    test_df = test_df.dropna(subset=["text", "sentiment"])

    # Convert sentiment to integer explicitly — prevents type issues during training
    train_df["sentiment"] = train_df["sentiment"].astype(int)
    test_df["sentiment"] = test_df["sentiment"].astype(int)

    print(f"Train set: {train_df.shape[0]:,} tweets")
    print(f"Test set:  {test_df.shape[0]:,} tweets")
    print(f"Sentiment distribution:\n{train_df['sentiment'].value_counts()}")

    return train_df, test_df


def build_vocabulary(texts, max_vocab=20000):
    # This function builds a vocabulary — a dictionary mapping every unique word to an integer ID
    # The model can't read words, only numbers, so every word needs an ID
    # Analogy: assigning every student a unique ID number so the computer can work with them

    print("Building vocabulary...")

    word_counts = {}
    for text in texts:
        for word in str(text).split():
            word_counts[word] = word_counts.get(word, 0) + 1

    # Sort by frequency and keep only the top max_vocab words
    # Rare words like typos get dropped — they don't help the model learn
    sorted_words = sorted(word_counts, key=word_counts.get, reverse=True)[:max_vocab]

    # Reserve index 0 for padding and index 1 for unknown words
    # Padding fills short tweets to a fixed length
    # Unknown words are words seen at test time that weren't in the training vocabulary
    # Analogy: seat 0 is always empty (padding), seat 1 is for visitors (unknown words)
    word2idx = {word: idx + 2 for idx, word in enumerate(sorted_words)}
    word2idx["<PAD>"] = 0
    word2idx["<UNK>"] = 1

    print(f"Vocabulary size: {len(word2idx):,} words")
    return word2idx


def encode_tweets(texts, word2idx, max_length=50):
    # This function converts each tweet from words into a fixed-length list of integer IDs
    # Every tweet must be the same length so the model can process them in batches
    # Analogy: a form with exactly 50 blank boxes —
    # long answers get cut off, short ones get padded with zeros

    encoded = []
    for text in texts:
        # Convert each word to its ID — unknown words get index 1 (<UNK>)
        ids = [word2idx.get(word, 1) for word in str(text).split()]

        # Truncate if too long, pad with zeros if too short
        if len(ids) > max_length:
            ids = ids[:max_length]
        else:
            ids = ids + [0] * (max_length - len(ids))

        encoded.append(ids)

    return encoded


class TweetDataset(Dataset):
    # This class wraps our encoded tweets and labels into a format PyTorch can work with
    # Analogy: putting ingredients into standardized containers for the factory conveyor belt

    def __init__(self, X, y):
        # Convert lists to PyTorch tensors
        # A tensor is PyTorch's version of a numpy array —
        # it's a grid of numbers that can be processed efficiently
        # Think of a tensor like this:
        # a single number is 0D, a list of numbers is 1D (a vector),
        # a table of numbers is 2D (a matrix), and a tensor can be any of these
        # or even higher dimensions — it's just a container for numbers
        # Analogy: a vector is a single row of a spreadsheet,
        # a matrix is the whole spreadsheet,
        # a tensor is a whole filing cabinet of spreadsheets
        self.X = torch.tensor(np.array(X), dtype=torch.long)
        self.y = torch.tensor(np.array(y), dtype=torch.float32)

    def __len__(self):
        # Returns the total number of tweets — DataLoader calls this automatically
        return len(self.X)

    def __getitem__(self, idx):
        # Returns one tweet and its label at a given position
        # Analogy: a librarian retrieving any book by its shelf number
        return self.X[idx], self.y[idx]


class DANClassifier(nn.Module):
    # This class defines our Deep Averaging Network (DAN)
    # It inherits from nn.Module — PyTorch's base class for all neural networks
    # Analogy: the architect's blueprint — describes what the network looks like

    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(DANClassifier, self).__init__()

        # Embedding layer — converts each word ID into a dense vector of numbers
        # vocab_size + 2 accounts for padding (0) and unknown (1) tokens
        # Analogy: converting a student ID number into a rich profile of that student
        self.embedding = nn.Embedding(vocab_size + 2, embed_dim, padding_idx=0)

        # The network has three fully connected layers
        # Each layer transforms the data into a more useful representation
        # Analogy: like a series of filters — each one refines the signal further

        # Layer 1 — takes the averaged word embeddings and transforms them
        self.fc1 = nn.Linear(embed_dim, hidden_dim)

        # Layer 2 — takes the output of layer 1 and refines it further
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)

        # Layer 3 — final output layer, maps to a single number (positive/negative)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)

        # ReLU activation function — applied after layers 1 and 2
        # It replaces any negative number with 0, keeping positive numbers as is
        # This adds non-linearity — without it the layers would collapse into one
        # Analogy: like a filter that lets positive signals through and blocks negative ones
        self.relu = nn.ReLU()

        # Dropout — randomly turns off 30% of neurons during training
        # Forces the model to learn redundant patterns, prevents memorization
        # Analogy: studying without your notes sometimes forces deeper understanding
        self.dropout = nn.Dropout(0.3)

        # Sigmoid — squashes the final output to between 0 and 1
        # Values close to 1 = positive, close to 0 = negative
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # This defines how data flows through the network
        # Analogy: describing how water flows through the pipes of a building

        # Step 1 — convert word IDs to embeddings
        # Shape goes from (batch, max_length) to (batch, max_length, embed_dim)
        embedded = self.embedding(x)

        # Step 2 — average all word embeddings in each tweet into one vector
        # This collapses the sequence dimension — instead of 50 word vectors
        # we now have one averaged vector representing the whole tweet
        # dim=1 means we average across the word dimension
        # Analogy: averaging all the scores on a report card into one GPA
        averaged = embedded.mean(dim=1)

        # Step 3 — pass through the three layers with ReLU and dropout in between
        out = self.relu(self.fc1(averaged))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)

        # Step 4 — squash to between 0 and 1
        return self.sigmoid(out).squeeze()


def train_model(train_df, word2idx):
    # This function builds the DAN model and trains it over multiple epochs
    # Analogy: a student studying the textbook multiple times, improving each pass

    X_train = encode_tweets(train_df["text"], word2idx)
    y_train = train_df["sentiment"].tolist()

    dataset = TweetDataset(X_train, y_train)

    # batch_size=256 — feed 256 tweets at a time
    # shuffle=True — randomize order each epoch
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    model = DANClassifier(
        vocab_size=20000,
        embed_dim=128,      # each word represented by 128 numbers
        hidden_dim=256      # hidden layers have 256 numbers
    ).to(device)

    # BCELoss measures how wrong the predictions are for binary classification
    # Analogy: a teacher marking how far off each answer was, not just right or wrong
    criterion = nn.BCELoss()

    # Adam optimizer adjusts model weights after each batch to reduce loss
    # lr=0.001 controls how big each adjustment step is
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 5
    print(f"Training DAN model for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Zero out gradients from the previous batch
            # Analogy: erasing the whiteboard before solving the next problem
            optimizer.zero_grad()

            # Forward pass — run the batch through the model
            predictions = model(X_batch)

            # Calculate how wrong the predictions were
            loss = criterion(predictions, y_batch)

            # Backward pass — calculate how to adjust each weight
            loss.backward()

            # Gradient clipping — prevents any weight update from being too large
            # Analogy: a speed limiter — the model can still learn but can't make wild swings
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update weights
            optimizer.step()

            total_loss += loss.item()

            # Track training accuracy each epoch
            predicted_labels = (predictions >= 0.5).float()
            correct += (predicted_labels == y_batch).sum().item()
            total += y_batch.size(0)

        avg_loss = total_loss / len(loader)
        train_accuracy = correct / total
        print(f"Epoch {epoch + 1}/{epochs} — Loss: {avg_loss:.4f} — Train Accuracy: {train_accuracy * 100:.2f}%")

    return model


def evaluate_model(model, test_df, word2idx):
    # This function tests the trained model on tweets it has never seen
    # Analogy: the final exam — brand new questions, no peeking at the textbook

    print("\n--- DAN Model Evaluation ---")

    X_test = encode_tweets(test_df["text"], word2idx)
    y_test = test_df["sentiment"].tolist()

    dataset = TweetDataset(X_test, y_test)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    # eval mode — disables dropout and gradient tracking
    # Analogy: switching from practice mode to exam mode
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            predictions = model(X_batch)

            # Above 0.5 = positive, below 0.5 = negative
            predicted_labels = (predictions >= 0.5).float()
            correct += (predicted_labels == y_batch).sum().item()
            total += y_batch.size(0)

    accuracy = correct / total
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    return accuracy


def save_model(model, word2idx):
    # This function saves the trained model and vocabulary to disk
    # so Script 05 can load them for comparison without retraining
    # Analogy: filing away finished notes so you can refer back without re-reading the textbook

    os.makedirs("models", exist_ok=True)

    # torch.save() saves only the model's learned weights
    # Analogy: saving the student's notes, not the entire textbook
    torch.save(model.state_dict(), "models/dan_model.pt")

    # Save the vocabulary so Script 05 can encode new tweets the same way
    joblib.dump(word2idx, "models/dan_vocab.joblib")

    print("\nModel saved to models/dan_model.pt")
    print("Vocabulary saved to models/dan_vocab.joblib")


# This block only runs if you execute this file directly (python 04_train_lstm.py)
# If another script imports this file, this block is skipped
if __name__ == "__main__":
    train_df, test_df = load_data()
    word2idx = build_vocabulary(train_df["text"])
    model = train_model(train_df, word2idx)
    evaluate_model(model, test_df, word2idx)
    save_model(model, word2idx)