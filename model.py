from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf

def create_traditional_model(model_type="logistic_regression"):
    # Set up TF-IDF with bi- and tri-grams
    if model_type == "logistic_regression":
        classifier = LogisticRegression(max_iter=1000)
    elif model_type == "naive_bayes":
        classifier = MultinomialNB()
    elif model_type == "svm":
        classifier = SVC()

    model = Pipeline([
        # includes unigrams, bigrams, and trigrams
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 3))),
        ('classifier', classifier)
    ])
    return model
