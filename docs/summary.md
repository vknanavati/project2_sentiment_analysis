## Project 2 Complete — Final Summary

### What we built
A full NLP sentiment analysis pipeline that takes raw tweets, cleans them, trains two different models, and compares them side by side.

### Final results

| Model | Accuracy | Baseline |
|---|---|---|
| Random guessing | 50.00% | — |
| TF-IDF + Logistic Regression | 79.76% | +29.76% |
| Deep Averaging Network (DAN) | 78.94% | +28.94% |

### What each script did

**`01_download_data.py`** — Downloaded 1.6 million real tweets from Kaggle using kagglehub and saved them to the `data/` folder

**`02_preprocess.py`** — Cleaned every tweet by removing URLs, @mentions, hashtags, punctuation, and inconsistent casing. Split the data into 1,277,069 training tweets and 319,268 test tweets

**`03_train_tfidf.py`** — Converted tweets into TF-IDF vectors and trained a Logistic Regression classifier. Achieved 79.76% accuracy and revealed the most positive and negative words in the dataset

**`04_train_lstm.py`** — Built and trained a Deep Averaging Network using PyTorch word embeddings. Achieved 78.94% accuracy across 5 training epochs

**`05_compare.py`** — Loaded both models, ran them on the same test set, generated a comparison bar chart, and ran example predictions side by side

---

### Key lessons learned

**1. Simple models are surprisingly powerful**
TF-IDF + Logistic Regression and the DAN finished less than 1% apart in accuracy. For short text like tweets, knowing which words appear is almost as powerful as having rich learned representations of those words.

**2. More complex doesn't always mean better**
The LSTM was theoretically more powerful than the DAN but couldn't learn anything on this setup. The DAN was simpler, more stable, and achieved nearly identical results.

**3. Data cleaning matters enormously**
3,663 tweets were dropped after cleaning because they contained nothing but URLs or @mentions. Garbage in, garbage out — the cleaner the data, the better the model.

**4. Word order matters less for short text**
Both models ignored word order and still hit ~79% accuracy. For longer, more complex text the gap between sequential models (LSTM) and averaging models (DAN/TF-IDF) would be much larger.

**5. Hardware matters**
PyTorch's MPS backend on Apple Silicon caused the LSTM to get completely stuck at 0.6932 loss — a real world reminder that the same code can behave differently on different hardware.

---

### New concepts learned this project
- Tokenization and text preprocessing
- TF-IDF vectorization
- Word embeddings
- Deep Averaging Networks
- PyTorch tensors, datasets, and dataloaders
- Training loops, loss functions, and gradient clipping
- The `if __name__ == "__main__"` guard and why it exists
- Epoch, loss, and what a stuck model looks like vs a learning one