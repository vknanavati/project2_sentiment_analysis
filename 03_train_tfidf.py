# pandas loads our cleaned train and test CSV files into DataFrames
# Analogy: pandas is like Excel for Python — rows, columns, and easy filtering
import pandas as pd

# TfidfVectorizer converts raw text into a matrix of TF-IDF numbers
# It turns each tweet into a vector of numbers the model can learn from
# Analogy: TfidfVectorizer is like a translator that converts human language
# into a language of numbers that the model can understand
from sklearn.feature_extraction.text import TfidfVectorizer

# LogisticRegression is our first model — it takes the TF-IDF vectors and learns
# to classify each tweet as positive or negative
# Analogy: LogisticRegression is like a judge who looks at the evidence (the numbers)
# and makes a decision (positive or negative)
from sklearn.linear_model import LogisticRegression

# accuracy_score measures what percentage of predictions the model got right
# classification_report gives a detailed breakdown of precision, recall, and F1 score
# Analogy: accuracy_score is the final exam grade,
# classification_report is the detailed report card
from sklearn.metrics import accuracy_score, classification_report

# joblib saves and loads Python objects to disk efficiently
# We use it to save the trained model and vectorizer so Script 05 can load them later
# Analogy: joblib is like a filing cabinet — it stores the trained model so we
# don't have to retrain it every time we want to use it
import joblib

# os lets us create folders and build file paths
import os


# This script is the first model trainer of the project
# It loads the cleaned tweets, converts them into TF-IDF number vectors,
# trains a Logistic Regression classifier, evaluates it on the test set,
# and saves the trained model to disk
# Analogy: think of this script as a student who reads a textbook (training data),
# takes notes in a numerical shorthand (TF-IDF), learns patterns from those notes
# (Logistic Regression), then takes a final exam (test set) to prove what they learned


def load_data():
    # This function's job is to load the cleaned train and test CSV files
    # that Script 02 saved to disk
    # Analogy: like a chef picking up the prepped ingredients the prep cook left ready

    print("Loading cleaned data...")

    # Read the training set — this is what the model will learn from
    train_df = pd.read_csv("data/train.csv")

    # Read the test set — this is what we'll use to evaluate the model
    test_df = pd.read_csv("data/test.csv")

    print(f"Train set: {train_df.shape[0]:,} tweets")
    print(f"Test set:  {test_df.shape[0]:,} tweets")

    return train_df, test_df


def get_features_and_labels(train_df, test_df):
    # This function's job is to separate each DataFrame into:
    # X = the tweet text (the input the model learns from)
    # y = the sentiment label (the answer the model is trying to predict)
    # Analogy: like separating an exam paper into questions (X) and answer key (y)
    # The model only sees the questions during training — never the answer key

    # X_train is the tweet text from the training set
    # y_train is the sentiment label from the training set (0 or 1)
    X_train = train_df["text"]
    y_train = train_df["sentiment"]

    # X_test is the tweet text from the test set
    # y_test is the sentiment label from the test set (0 or 1)
    X_test = test_df["text"]
    y_test = test_df["sentiment"]

    return X_train, y_train, X_test, y_test


def vectorize(X_train, X_test):
    # This function's job is to convert raw tweet text into TF-IDF number vectors
    # The model cannot read words — it can only read numbers
    # TF-IDF scores each word based on how often it appears in a tweet
    # AND how rare it is across all tweets
    # Common words like "the" get low scores — they appear everywhere and mean nothing
    # Distinctive words like "terrible" get high scores — they signal something meaningful
    # Analogy: like a highlighter that ignores filler words and emphasizes the words
    # that actually tell you something about the tweet's sentiment

    print("Vectorizing tweets with TF-IDF...")

    # max_features=50000 means we only keep the 50,000 most important words
    # and ignore the rest — this keeps the model fast without losing much signal
    # Analogy: instead of memorizing every word ever written, you study only
    # the 50,000 most useful ones
    vectorizer = TfidfVectorizer(max_features=50000)

    # fit_transform() does two things at once on the training data:
    # fit = learns which 50,000 words matter and what their scores are
    # transform = converts every training tweet into a vector of those scores
    # Analogy: the vectorizer reads the entire textbook (fit) then rewrites
    # every tweet as a row of numbers (transform)
    X_train_vec = vectorizer.fit_transform(X_train)

    # transform() only — we do NOT fit on the test data
    # The vectorizer already learned the vocabulary from the training set
    # We just apply those same rules to the test tweets
    # Analogy: the exam uses the same scoring rules the student studied —
    # we don't invent new rules for the test
    X_test_vec = vectorizer.transform(X_test)

    print(f"Each tweet is now a vector of {X_train_vec.shape[1]:,} numbers")

    return vectorizer, X_train_vec, X_test_vec


def train_model(X_train_vec, y_train):
    # This function's job is to train the Logistic Regression model
    # It looks at all 1.27 million training tweets as TF-IDF vectors
    # and learns which word patterns predict positive vs negative sentiment
    # Analogy: like a judge reading thousands of past cases to learn
    # which patterns of evidence tend to lead to guilty vs not guilty verdicts

    print("Training Logistic Regression model...")

    # max_iter=1000 gives the model up to 1000 attempts to find the best solution
    # The default is 100 which sometimes isn't enough for large datasets like this
    # Analogy: like giving a student 1000 practice problems instead of 100
    # so they have enough chances to really learn the patterns
    model = LogisticRegression(max_iter=1000)

    # fit() is where the actual learning happens
    # The model adjusts its internal weights until it can reliably separate
    # positive tweets from negative ones based on their TF-IDF vectors
    # Analogy: the judge reviews all the cases and builds up an internal rulebook
    model.fit(X_train_vec, y_train)

    print("Training complete")

    return model


def evaluate_model(model, vectorizer, X_test_vec, y_test):
    # This function's job is to test the trained model on tweets it has never seen before
    # and report how well it performed
    # Analogy: like sitting the student down for a final exam using brand new questions
    # that weren't in the textbook — this is the true test of whether they really learned

    print("\n--- Model Evaluation ---")

    # model.predict() runs every test tweet through the trained model
    # and returns a prediction of 0 (negative) or 1 (positive) for each one
    # Analogy: the judge reads each new case and delivers a verdict
    y_pred = model.predict(X_test_vec)

    # accuracy_score compares our predictions to the real labels
    # and returns the percentage we got right
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    # classification_report gives a deeper breakdown per class:
    # precision = of all tweets we called positive, how many actually were?
    # recall    = of all actually positive tweets, how many did we catch?
    # f1-score  = a single number that balances precision and recall
    # Analogy: precision is "how often are you right when you make a call?"
    # recall is "how many of the right answers did you find?"
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

    # Show the top 10 words that most strongly predict positive sentiment
    # and the top 10 words that most strongly predict negative sentiment
    # feature_names_out() gives us the list of all 50,000 words the vectorizer learned
    # coef_[0] gives us the weight the model assigned to each word
    # High positive weight = strongly predicts positive sentiment
    # High negative weight = strongly predicts negative sentiment
    # Analogy: like finding out which words on a resume most impress a hiring manager
    # and which words most put them off
    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_[0]

    top_positive = sorted(zip(coefs, feature_names), reverse=True)[:10]
    top_negative = sorted(zip(coefs, feature_names))[:10]

    print("\nTop 10 words predicting POSITIVE sentiment:")
    for coef, word in top_positive:
        print(f"  {word}: {coef:.4f}")

    print("\nTop 10 words predicting NEGATIVE sentiment:")
    for coef, word in top_negative:
        print(f"  {word}: {coef:.4f}")


def save_model(model, vectorizer):
    # This function's job is to save the trained model and vectorizer to disk
    # so Script 05 can load them later for comparison without retraining
    # Analogy: like a student writing up their notes and filing them away
    # so they can refer back to them later without re-reading the whole textbook

    # Create the models/ folder if it doesn't exist yet
    os.makedirs("models", exist_ok=True)

    # joblib.dump() serializes the object and writes it to a file
    # Serializing means converting a Python object into a format that can be saved to disk
    # Analogy: like taking a 3D object and flattening it into a blueprint
    # that can be stored in a filing cabinet and rebuilt later
    joblib.dump(model, "models/tfidf_model.joblib")
    joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")

    print("\nModel saved to models/tfidf_model.joblib")
    print("Vectorizer saved to models/tfidf_vectorizer.joblib")


# This block only runs if you execute this file directly (python 03_train_tfidf.py)
# If another script imports this file, this block is skipped
if __name__ == "__main__":
    train_df, test_df = load_data()
    X_train, y_train, X_test, y_test = get_features_and_labels(train_df, test_df)
    vectorizer, X_train_vec, X_test_vec = vectorize(X_train, X_test)
    model = train_model(X_train_vec, y_train)
    evaluate_model(model, vectorizer, X_test_vec, y_test)
    save_model(model, vectorizer)