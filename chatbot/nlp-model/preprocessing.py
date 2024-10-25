import nltk
import spacy

nltk.download('punkt')
nlp = spacy.load('en_core_web_sm')


def preprocess_text(text):
    # Tokenization, lemmatization, etc.
    doc = nlp(text)
    return [token.lemma_ for token in doc]
