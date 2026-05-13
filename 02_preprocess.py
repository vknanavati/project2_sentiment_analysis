# pandas loads and manipulates our data in table format
# Analogy: pandas is like Excel for Python — rows, columns, and easy filtering
import pandas as pd

# re is Python's built-in regular expressions library
# It lets us find and replace patterns in text — like "delete anything that starts with http"
# Analogy: re is like a very powerful Find & Replace in a text editor
import re

# sklearn's train_test_split divides our data into a training set and a test set
# The model learns from the training set and gets evaluated on the test set
# Analogy: the training set is the textbook, the test set is the final exam
from sklearn.model_selection import train_test_split


# This script is the data janitor of the project
# The raw tweets are messy — full of URLs, @mentions, symbols, and inconsistent casing
# This script loads the raw data, scrubs every tweet clean, and saves two tidy files
# (train.csv and test.csv) that the model scripts can learn from directly
# Analogy: think of this script as a kitchen prep cook — it doesn't do the cooking,
# but it washes, peels, and chops everything so the chef (the model) can work cleanly


def load_raw_data():
    # This function's only job is to open the raw CSV file and load it into a DataFrame
    # The Sentiment140 CSV has no header row — the columns have no names yet
    # so we name them ourselves with the 'names' parameter
    # Analogy: like receiving an unmarked spreadsheet and writing the column headers yourself
    # before you can make sense of what's in it

    # The six columns in Sentiment140 are:
    # sentiment  = 0 (negative) or 4 (positive)
    # id         = unique tweet ID
    # date       = when the tweet was posted
    # query      = search query used to find it (mostly "NO_QUERY")
    # user       = the Twitter username
    # text       = the actual tweet content
    print("Loading raw data...")
    df = pd.read_csv(
        "data/training.1600000.processed.noemoticon.csv",
        encoding="latin-1",
        names=["sentiment", "id", "date", "query", "user", "text"]
    )

    print(f"Loaded {df.shape[0]:,} tweets")
    return df


def clean_data(df):
    # This function's job is to simplify the labels and throw away columns we don't need
    # It remaps sentiment from 0/4 to 0/1 and keeps only the sentiment and text columns
    # Analogy: like receiving a form with 6 fields but only needing 2 of them —
    # you circle the ones you care about and shred the rest

    # Sentiment140 uses 0 for negative and 4 for positive
    # We remap 4 → 1 so our labels are clean binary: 0 = negative, 1 = positive
    # This makes it easier for the model and for us to read results
    # Analogy: changing a grading scale from 0/4 to 0/1 — same meaning, cleaner numbers
    df["sentiment"] = df["sentiment"].map({0: 0, 4: 1})

    # We only need the tweet text and its label — drop everything else
    # id, date, query, and user don't help the model learn sentiment
    df = df[["sentiment", "text"]]

    print(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")
    return df


def clean_tweet(text):
    # This function's job is to take one raw tweet and scrub it clean
    # It removes all the noise — URLs, @mentions, symbols, punctuation, numbers
    # and returns a lowercase plain-English version of the tweet
    # Analogy: like a translator who strips out all the slang, symbols, and shorthand
    # before handing the message to someone who only understands plain English

    # Remove URLs — anything starting with http or https
    # re.sub(pattern, replacement, text) finds all matches and replaces them with ""
    # r"http\S+" means: the word "http" followed by any non-space characters
    # Analogy: crossing out website addresses in a printed article before reading it
    text = re.sub(r"http\S+", "", text)

    # Remove @mentions like @username — they don't carry sentiment meaning
    # r"@\S+" means: the @ symbol followed by any non-space characters
    text = re.sub(r"@\S+", "", text)

    # Remove hashtag symbols but keep the word itself
    # "I love #python" becomes "I love python"
    # We only remove the # character, not the word that follows
    text = re.sub(r"#", "", text)

    # Remove any character that isn't a letter or a space
    # r"[^a-zA-Z\s]" means: anything that is NOT a-z, A-Z, or whitespace
    # This removes punctuation, numbers, emojis, and special characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Convert everything to lowercase so "Great" and "great" are treated as the same word
    # Analogy: like making everyone wear the same uniform — the model sees words,
    # not capitalization styles
    text = text.lower()

    # strip() removes any extra whitespace from the beginning and end of the tweet
    text = text.strip()

    return text


def apply_cleaning(df):
    # This function's job is to run clean_tweet() on every single row in the dataset
    # It applies the cleaning function to all 1.6 million tweets automatically
    # then drops any tweets that ended up empty after cleaning
    # Analogy: like sending an entire warehouse of dirty packages through a conveyor belt
    # car wash — every package gets cleaned, and anything that comes out empty gets tossed

    print("Cleaning tweets...")

    # df["text"].apply(clean_tweet) calls clean_tweet() once per tweet — all 1.6 million
    df["text"] = df["text"].apply(clean_tweet)

    # Drop any rows where cleaning left us with an empty string
    # Some tweets are nothing but URLs or mentions — after cleaning they become blank
    # A blank tweet is useless for training
    df = df[df["text"].str.strip() != ""]

    print(f"Tweets remaining after cleaning: {df.shape[0]:,}")
    return df


def split_and_save(df):
    # This function's job is to divide the cleaned data into two groups and save them to disk
    # The training set is what the model learns from
    # The test set is held back and used to evaluate how well the model learned
    # We save both as CSV files so Scripts 03 and 04 can load them directly
    # without having to re-clean everything from scratch
    # Analogy: like a teacher splitting a question bank into practice questions (training)
    # and exam questions (test) — the student never sees the exam questions until test day

    # train_test_split() shuffles the data and splits it into two groups
    # test_size=0.2 means 20% goes to the test set, 80% goes to training
    # random_state=42 locks in the shuffle so we get the same split every time we run this
    # Analogy: marking a deck of cards before shuffling so the shuffle is always identical
    print("Splitting into train and test sets...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Save both sets as CSV files in the data/ folder
    # index=False means don't write the row numbers into the file
    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)

    print(f"Train set: {train_df.shape[0]:,} tweets")
    print(f"Test set:  {test_df.shape[0]:,} tweets")
    print("Saved to data/train.csv and data/test.csv")


# This block only runs if you execute this file directly (python 02_preprocess.py)
# If another script imports this file, this block is skipped
if __name__ == "__main__":
    df = load_raw_data()
    df = clean_data(df)
    df = apply_cleaning(df)
    split_and_save(df)