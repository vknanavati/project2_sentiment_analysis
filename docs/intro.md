# Project 2: Twitter Sentiment Analysis (NLP)

## The Problem This Project Solves

Every day, millions of people post their opinions on Twitter — about products, movies, politics, restaurants, and everything in between. Companies and researchers want to know: **is the public feeling positive or negative about something?**

Reading millions of tweets by hand is impossible. This project builds a machine that can automatically read a tweet and decide whether it carries a **positive** or **negative** sentiment. That's called **sentiment analysis**, and it's one of the most widely used applications of Natural Language Processing (NLP) in the real world.

---

## What We're Building

A full NLP pipeline that:
1. Downloads and cleans 1.6 million real tweets
2. Trains **two different models** on the same data
3. Compares them side-by-side to understand the tradeoffs

The two models:
- **TF-IDF + Logistic Regression** — a fast, traditional ML approach
- **LSTM Neural Network** — a more powerful deep learning approach that understands word order

---

## Project Structure

```
project2_sentiment_analysis/
│
├── venv/                   # Virtual environment (not committed to git)
├── data/                   # Raw and processed data (not committed to git)
├── models/                 # Saved trained models (not committed to git)
│
├── 01_download_data.py     # Download the Sentiment140 dataset
├── 02_preprocess.py        # Clean and prepare the tweets
├── 03_train_tfidf.py       # Train the TF-IDF + Logistic Regression model
├── 04_train_lstm.py        # Train the LSTM neural network
├── 05_compare.py           # Compare both models side-by-side
│
├── .gitignore
└── README.md
```

---

## Libraries We'll Use

### `pandas`
**What it is:** A library for loading and manipulating data in table format (like a spreadsheet in Python).
**Why we need it:** The Sentiment140 dataset is a CSV file with millions of rows. Pandas lets us load it, filter it, inspect it, and pass it to our models cleanly.

### `numpy`
**What it is:** A library for fast numerical computing — working with arrays and matrices of numbers.
**Why we need it:** Under the hood, almost every ML operation turns data into grids of numbers. NumPy is the foundation that makes that fast.

### `scikit-learn`
**What it is:** The go-to Python library for traditional machine learning.
**Why we need it:** We'll use it for the TF-IDF vectorizer (turns text into numbers), the Logistic Regression model, and evaluation tools like accuracy scores and confusion matrices.

### `torch` (PyTorch)
**What it is:** A deep learning framework developed by Meta, widely used in research and industry.
**Why we need it:** We'll use it to build and train the LSTM neural network — defining layers, running training loops, and saving the model.

### `matplotlib`
**What it is:** A library for creating charts and visualizations.
**Why we need it:** We'll plot training accuracy over time and create a side-by-side comparison chart of both models.

### `requests`
**What it is:** A library for making HTTP requests — downloading things from the internet.
**Why we need it:** To programmatically download the Sentiment140 dataset without leaving the terminal.

### `kagglehub`
**What it is:** A library for accessing datasets from Kaggle directly in Python.
**Why we need it:** Sentiment140 is hosted on Kaggle. KaggleHub lets us download it cleanly with one function call.

---

## Concepts We'll Learn

### Tokenization
Breaking a sentence into individual words or pieces. "I love this movie" becomes `["I", "love", "this", "movie"]`. This is the first step in almost all NLP — you can't analyze a sentence as a whole blob of text.

### Text Preprocessing / Cleaning
Real tweets are messy — full of URLs, @mentions, emojis, and slang. Before training, we strip out the noise so the model focuses on the meaningful words. Garbage in, garbage out.

### TF-IDF (Term Frequency–Inverse Document Frequency)
A way to turn words into numbers. It scores each word based on how often it appears in a tweet *and* how rare it is across all tweets. Common words like "the" get low scores; distinctive words like "terrible" get high scores. This produces a vector (a list of numbers) representing each tweet.

### Logistic Regression
A classic ML algorithm that takes a vector of numbers as input and predicts a category (positive or negative). Despite the name, it's a classification algorithm. It's fast, interpretable, and often surprisingly strong.

### Word Embeddings
Instead of treating words as isolated tokens, embeddings represent each word as a dense vector of numbers that captures *meaning*. Words with similar meanings end up with similar vectors. The LSTM uses these as input.

### LSTM (Long Short-Term Memory)
A type of neural network designed to process *sequences* — things where order matters. Unlike TF-IDF (which treats a tweet as a bag of words with no order), an LSTM reads a tweet word-by-word and maintains a "memory" of what it has seen. This lets it understand that "not bad" is positive, even though "bad" alone is negative.

### Confusion Matrix
A grid that shows not just how accurate a model is, but *how* it makes mistakes. It breaks down true positives, true negatives, false positives, and false negatives — giving a much richer picture than a single accuracy number.

### Overfitting
When a model memorizes the training data instead of learning general patterns. It scores very high on training data but poorly on new, unseen tweets. We'll watch for this with the LSTM.

---

## Script-by-Script Breakdown

### `01_download_data.py`
**Purpose:** Download the Sentiment140 dataset from Kaggle and save it to the `data/` folder.
Sentiment140 contains 1.6 million tweets labeled 0 (negative) or 4 (positive), collected in 2009. We'll relabel 4 → 1 so we're working with clean binary labels: 0 = negative, 1 = positive.

### `02_preprocess.py`
**Purpose:** Clean the raw tweets and prepare them for both models.
Raw tweets contain URLs (`https://...`), @mentions, hashtag symbols, punctuation, and inconsistent casing. This script strips all of that out, lowercases everything, and saves a clean version of the dataset. It also splits the data into training and test sets.

### `03_train_tfidf.py`
**Purpose:** Train the first model — TF-IDF vectorizer + Logistic Regression.
This script converts the cleaned tweets into TF-IDF vectors, trains a Logistic Regression classifier, evaluates it on the test set, and saves the trained model. It also shows which words most strongly predicted positive vs. negative sentiment.

### `04_train_lstm.py`
**Purpose:** Train the second model — an LSTM neural network.
This script builds a vocabulary from the training tweets, converts words to integer IDs, trains a PyTorch LSTM model with word embeddings, and evaluates it on the test set. Training happens over multiple epochs and we track accuracy over time.

### `05_compare.py`
**Purpose:** Load both trained models and compare them directly.
This script runs the same test set through both models and produces: a side-by-side accuracy comparison, a bar chart, and live predictions on a handful of example tweets so you can see where each model agrees and disagrees.

---

## What "Good" Looks Like

Since this is a binary classification task (positive vs. negative), a model that just guessed randomly would be right 50% of the time. A model that always guessed "positive" (the majority class) might hit ~55-60%. We're aiming for:

- **TF-IDF + Logistic Regression:** ~80–83% accuracy
- **LSTM:** ~82–86% accuracy

The gap may be smaller than you expect — that's one of the lessons of this project.