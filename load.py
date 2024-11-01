import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
import nltk
nltk.download('stopwords')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Custom stopwords
    custom_stopwords = set(stopwords.words("english"))
    custom_stopwords.update(["lyx", "cool", "also", "tweet", "please", "like", "just"])
    words = text.split()
    words = [word for word in words if word not in custom_stopwords]
    
    # Stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)

def load_and_preprocess(filepath):
    columns = ["textID", "text", "selected_text", "sentiment", "Time of Tweet", "Age of User", "Country", "Population -2020", "Land Area (Km²)", "Density (P/Km²)"]
    df = pd.read_csv(filepath, usecols=["text", "sentiment"], encoding="ISO-8859-1", names=columns)
    
    # Map sentiment to labels
    sentiment_mapping = {"negative": 0, "neutral": 1, "positive": 2}
    df["sentiment"] = df["sentiment"].map(sentiment_mapping)

    df["text"] = df["text"].apply(clean_text)
    X_train, X_test, y_train, y_test = train_test_split(df["text"], df["sentiment"], test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
