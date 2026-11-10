# Twitter Sentiment Analysis

A machine learning web app that sorts a tweet into one of four classes: **Positive**, **Negative**, **Neutral** or **Irrelevant**. The model uses TF-IDF features with a Linear Support Vector Classifier. The app is built with Streamlit.

**Live demo:** _add your Streamlit Cloud link here_

---

## Table of Contents
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Method](#method)
4. [Data Integrity: Leakage Found and Fixed](#data-integrity-leakage-found-and-fixed)
5. [Results](#results)
6. [Project Structure](#project-structure)
7. [Run Locally](#run-locally)
8. [Deployment](#deployment)
9. [Limitations and Possible Solutions](#limitations-and-possible-solutions)
10. [Future Work](#future-work)

---

## Overview
This project covers the full machine learning workflow:
- cleaning and preparing noisy social media text
- comparing three classic text models with hyperparameter tuning
- checking the data for leakage between the train and test sets
- packaging the best model as a scikit-learn pipeline
- serving the model through an interactive web app

## Dataset
**Source:** [Twitter Entity Sentiment Analysis (Kaggle)](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)

The data has four columns: tweet ID, entity (the brand or game the tweet is about), sentiment label, and tweet text. The files have no header row, so column names were added manually.

| Split | Raw rows | Rows after cleaning |
|---|---|---|
| Train (`twitter_training.csv`) | 74,681 | 61,443 |
| Test (`twitter_validation.csv`) | 999 | 999 |

Class balance of the final train set:

| Class | Tweets | Share |
|---|---|---|
| Negative | 18,617 | 30.3% |
| Positive | 16,378 | 26.7% |
| Neutral | 15,566 | 25.3% |
| Irrelevant | 10,882 | 17.7% |

The classes are reasonably balanced, so no resampling was applied.

## Method

### 1. Text preprocessing
Every step is applied in the same order during training (notebook) and inference (`cleaning.py`):

| Step | Example |
|---|---|
| Lowercase | `Great Game` → `great game` |
| Emojis to text | `😍` → `:smiling_face_with_heart-eyes:` |
| Remove HTML tags | `<br>` → (removed) |
| Remove URLs | `https://...` → (removed) |
| Remove punctuation | `wow!!!` → `wow` |
| Expand contractions | `i've` → `i have` |
| Remove stopwords (NLTK English list) | `this is a game` → `game` |
| Lemmatize (WordNet) | `studies` → `study` |

Lemmatization was chosen over stemming because it returns real dictionary words, which keeps the meaning of each word.

The `id` and `entity` columns were dropped. The model only uses tweet text, because app users will only type a tweet.

### 2. Feature extraction
TF-IDF with unigrams and bigrams (`ngram_range=(1, 2)`). Bigrams let the model use word pairs such as "highly recommend".

### 3. Model selection
Three models were tuned with `GridSearchCV` (3-fold cross-validation, accuracy scoring). Each model was wrapped in a `Pipeline` with the TF-IDF step:

| Model | Hyperparameter searched | Best value |
|---|---|---|
| Logistic Regression | `C` ∈ {1, 5, 10} | 1 |
| Linear SVC | `C` ∈ {1, 5, 10} | 1 |
| Multinomial Naive Bayes | `alpha` ∈ {0.1, 0.5, 1.0} | 1.0 |

The best pipeline (TF-IDF + Linear SVC) was saved with `pickle` as `sentiment.pkl`, so the app can pass cleaned text straight to the model.

## Data Integrity: Leakage Found and Fixed
The first version of the model reached **98.5%** test accuracy, which is unusually high for tweet sentiment. An audit found two problems:

| Problem | Count |
|---|---|
| Exact duplicate tweets in the train set | 11,951 |
| Test tweets that also appeared in the train set | 604 of 999 (60%) |

So the model was mostly being tested on tweets it had already seen, and the score measured memory rather than real skill.

**Fix:**
1. Removed exact duplicates from the train set.
2. Removed every train tweet that also appears in the test set. The test set was kept at its full 999 rows.
3. Checked that both counts are now zero, then retrained all models.

## Results
Results on the 999-tweet test set, after the leakage fix:

| Model | Test accuracy |
|---|---|
| **Linear SVC** | **97.9%** |
| Logistic Regression | 94.5% |
| Multinomial Naive Bayes | 91.9% |

Linear SVC per-class results:

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Irrelevant | 0.99 | 0.98 | 0.99 | 171 |
| Negative | 0.98 | 0.98 | 0.98 | 266 |
| Neutral | 0.99 | 0.97 | 0.98 | 285 |
| Positive | 0.96 | 0.98 | 0.97 | 277 |
| **Macro average** | **0.98** | **0.98** | **0.98** | 999 |

> **Note:** these scores are still higher than the model's real performance on new tweets. See [Limitations](#limitations-and-possible-solutions), point 1.

## Project Structure
```
twitter-sentiment-analysis/
├── app.py                                   # Streamlit web app
├── cleaning.py                              # Text cleaning (same steps as training)
├── sentiment.pkl                            # Trained pipeline: TF-IDF + Linear SVC
├── Project_Twitter_Sentiment_Analysis.ipynb # Training and evaluation notebook
├── requirements.txt                         # Python dependencies
└── README.md
```

## Run Locally
Requires Python 3.12 or newer.

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/twitter-sentiment-analysis.git
cd twitter-sentiment-analysis

# 2. Create and activate a virtual environment
python -m venv myenv
myenv\Scripts\activate        # Windows
# source myenv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the app
streamlit run app.py
```

The app opens at `http://localhost:8501`. NLTK data (stopwords, WordNet) is downloaded automatically on the first run.

> `scikit-learn` is pinned to **1.6.1**, the version used to train the model. A different version may fail to load `sentiment.pkl`.

## Deployment
The app is deployed on **Streamlit Community Cloud**:
1. Push this repository to GitHub.
2. Sign in to [share.streamlit.io](https://share.streamlit.io) with GitHub.
3. Select **Create app**, choose this repository, branch `main`, and main file `app.py`.
4. Under **Advanced settings**, choose Python 3.12, then select **Deploy**.

## Limitations and Possible Solutions

### 1. Near-duplicate tweets still leak between train and test
**Problem:** the dataset contains many lightly reworded copies of the same tweet. For example:
- "i am coming to the borders and i will kill you all"
- "im getting on borderlands and i will kill you all"

Exact-match removal does not catch these, so the 97.9% score is still optimistic.

**Possible solutions:**
- Remove near-duplicates using similarity matching (for example, TF-IDF cosine similarity or MinHash) before splitting.
- Split the data by tweet ID or by entity (`GroupShuffleSplit`), so all versions of a tweet end up on the same side.
- Test on a fresh, hand-labelled set of real tweets.

### 2. Negation is lost during cleaning
**Problem:** the NLTK stopword list contains negation words such as "not", "no" and "nor". Removing them flips the meaning: "This phone is not good" becomes "phone good" and is predicted as **Positive**.

**Possible solution:** remove negation words from the stopword list before cleaning, in both the notebook and `cleaning.py`, then retrain:
```python
keep = {"not", "no", "nor", "don't", "isn't", "wasn't", "aren't",
        "didn't", "doesn't", "won't", "can't", "couldn't", "shouldn't"}
stop_words = set(stopwords.words("english")) - keep
```

### 3. Weak on Neutral and Irrelevant for new tweets
**Problem:** on new, unseen tweets, the model often predicts Positive or Negative for factual or off-topic text. For example, "Microsoft will announce the new Xbox update tomorrow" is predicted as Positive. There are two reasons:
- In this dataset, **Irrelevant** means "not about the given entity". The entity column is not used, so this cannot be learned from text alone.
- Neutral tweets have no strong opinion words, so TF-IDF has few useful signals.

**Possible solutions:**
- Merge Irrelevant into Neutral for a cleaner three-class problem.
- Add the entity as an input feature, if the use case always has a target brand.
- Use `class_weight="balanced"` and tune for macro F1 instead of accuracy.
- Use a transformer model pre-trained on tweets (see Future Work).

### 4. Bag-of-words cannot understand context or sarcasm
**Problem:** TF-IDF only counts words and word pairs. It does not understand word order beyond bigrams, sarcasm ("great, my phone died again"), or long-range context.

**Possible solution:** fine-tune a contextual model such as DistilBERT or Twitter-RoBERTa.

### 5. Dataset domain is narrow
**Problem:** the tweets are mostly about a small set of gaming and tech brands. The model may not generalize to other topics such as politics, health or sports.

**Possible solution:** train on a broader dataset (for example, TweetEval or Sentiment140), or combine several sources.

### 6. Pickle portability and security
**Problem:** `sentiment.pkl` only loads reliably with the same scikit-learn version, and pickle files can run arbitrary code when loaded.

**Possible solutions:** keep the version pinned in `requirements.txt`, only load model files from trusted sources, and consider `skops` or ONNX for safer model sharing.

## Future Work
- Add [Twitter-RoBERTa](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) as a second model in the app and compare it with the classic model.
- Show prediction confidence for each class in the app.
- Add a batch mode: upload a CSV of tweets and download the predictions.
- Add unit tests for `cleaning.py` to make sure training and inference stay consistent.

## Tech Stack
Python · pandas · scikit-learn · NLTK · emoji · contractions · Streamlit

## Author
**Muhammad Ahmed Memon**
