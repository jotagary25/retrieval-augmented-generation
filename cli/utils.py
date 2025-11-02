import string
from nltk.stem import PorterStemmer

_STEMMER = PorterStemmer()

def stemming(text: str, stopwords_path: str) -> list[str]:
    tokens = without_stopwords(text, stopwords_path)
    stemmed_tokens = [_STEMMER.stem(token) for token in tokens]
    return stemmed_tokens

def without_stopwords(text: str, stopwords_file: str) -> list[str]:
    tokens = tokenize(text)

    with open(stopwords_file, "r", encoding="utf-8") as file:
        stopwords = {line.strip().lower() for line in file if line.strip()}

    filtered_tokens = [t for t in tokens if t not in stopwords]
    return filtered_tokens

def tokenize(text: str) -> list[str]:
    cleaned = clean_text(text)
    return cleaned.split()

def clean_text(text: str) -> str:
    return text.translate(str.maketrans("", "", string.punctuation)).lower()
