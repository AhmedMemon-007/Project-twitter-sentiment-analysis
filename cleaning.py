# cleaning.py
# The same cleaning steps as the notebook, in the SAME ORDER.
# If you change a step here, you must also change it in the notebook and retrain the model.

import re
import string

import nltk
import emoji
import contractions
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data the first time (needed on the server too)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# Same stopword list as the notebook (the full NLTK English list)
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
exclude = string.punctuation


def remove_html_tags(text):
    pattern = re.compile("<.*?>")
    return pattern.sub(r"", text)


def remove_url(text):
    pattern = re.compile(r"https?://\S+|www\.\S+")
    return pattern.sub(r"", text)


def remove_punc(text):
    return text.translate(str.maketrans("", "", exclude))


def remove_stopwords(text):
    words = text.split()
    filtered = [x for x in words if x.lower() not in stop_words]
    return " ".join(filtered)


def lemmatize_text(text):
    return " ".join(lemmatizer.lemmatize(x) for x in text.split())


def clean_text(text):
    # Order is the same as the notebook (cells 17, 22, 27, 29)
    text = text.lower()
    text = emoji.demojize(text)
    text = remove_html_tags(text)
    text = remove_url(text)
    text = remove_punc(text)
    text = contractions.fix(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    return text