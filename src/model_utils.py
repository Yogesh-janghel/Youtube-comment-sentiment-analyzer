import os
import re
import pickle
import warnings
import numpy as np
import nltk

# Initialize NLTK data path at the very beginning
nltk_data_path = os.getenv('NLTK_DATA', '/usr/local/nltk_data')
if nltk_data_path not in nltk.data.path:
    nltk.data.path.insert(0, nltk_data_path)

# Download resources before importing sub-packages
def ensure_nltk_resources():
    # 'punkt' is essential for word_tokenize
    # 'punkt_tab' is used in newer NLTK versions
    resources = ['stopwords', 'wordnet', 'punkt', 'punkt_tab', 'omw-1.4']
    for res in resources:
        try:
            # Simple check if data exists
            if res == 'stopwords':
                nltk.data.find('corpora/stopwords')
            elif res == 'wordnet':
                nltk.data.find('corpora/wordnet')
            elif res == 'punkt':
                nltk.data.find('tokenizers/punkt')
            elif res == 'punkt_tab':
                nltk.data.find('tokenizers/punkt_tab')
            elif res == 'omw-1.4':
                nltk.data.find('corpora/omw-1.4')
        except LookupError:
            print(f"NLTK Resource {res} not found, downloading...")
            nltk.download(res, download_dir=nltk_data_path)

ensure_nltk_resources()

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from langdetect import detect
from googletrans import Translator
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Suppress BeautifulSoup warning
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

# Global variables for lazy loading
_analyzer = None
_translator = None
_lemmatizer = None
_stop_words = None

def get_analyzer():
    """Lazy load VADER analyzer"""
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentIntensityAnalyzer()
    return _analyzer

def get_utilities():
    """Lazy load utilities on first use"""
    global _translator, _lemmatizer, _stop_words
    if _translator is None:
        _translator = Translator()
    if _lemmatizer is None:
        _lemmatizer = WordNetLemmatizer()
    if _stop_words is None:
        _stop_words = set(stopwords.words('english'))
    return _translator, _lemmatizer, _stop_words

def detect_hinglish(comment):
    try:
        lang = detect(comment)
        # Simple heuristic to identify Hinglish
        if lang == 'en' and any(word in comment.lower() for word in ['hai', 'kya', 'nahi', 'kaise', 'toh', 'bhi', 'kuch', 'phir']):
            return True
        return False
    except:
        return False

def translate_to_english(comment):
    try:
        translator, _, _ = get_utilities()
        translation = translator.translate(comment, src='hi', dest='en')
        return translation.text
    except Exception as e:
        print(f"Translation error: {e}")
    return comment

def clean_and_preprocess_comments(comment):
    if detect_hinglish(comment):
        comment = translate_to_english(comment)
    
    # Convert to lowercase
    comment = comment.lower()
    # Remove URLs
    comment = re.sub(r'http\S+|www\S+|https\S+', '', comment, flags=re.MULTILINE)
    # Remove mentions
    comment = re.sub(r'@\w+', '', comment)
    # Remove punctuation
    comment = re.sub(r'[^\w\s]', '', comment)
    
    # Tokenize
    tokens = word_tokenize(comment)
    # Remove stopwords and lemmatize
    _, lemmatizer, stop_words = get_utilities()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def get_sentiment_vader(text):
    analyzer = get_analyzer()
    score = analyzer.polarity_scores(text)
    compound = score['compound']
    
    # Map compound score to 0 (Neg), 1 (Neu), 2 (Pos)
    if compound >= 0.05:
        return 2, compound
    elif compound <= -0.05:
        return 0, compound
    else:
        return 1, compound

def perform_sentiment_analysis(df, max_seq_length=None):
    # Clean and preprocess comments
    df['cleaned_text'] = df['text'].apply(clean_and_preprocess_comments)
    
    sentiments = []
    compounds = []
    for text in df['cleaned_text']:
        label, score = get_sentiment_vader(text)
        sentiments.append(label)
        compounds.append(score)
    
    df['sentiment'] = sentiments
    df['sentiment_score'] = compounds
    return df

def calculate_overall_sentiment(df):
    sentiment_counts = df['sentiment'].value_counts()
    if sentiment_counts.empty:
        return "Neutral", sentiment_counts
        
    overall_sentiment = sentiment_counts.idxmax()
    sentiment_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return sentiment_labels.get(overall_sentiment, "Neutral"), sentiment_counts

def prepare_top_comments(df):
    # Filter top positive comments
    top_positive_comments = df[df['sentiment'] == 2].nlargest(5, 'like_count')
    # Filter top negative comments
    top_negative_comments = df[df['sentiment'] == 0].nlargest(5, 'like_count')

    def clean_text(text):
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text()
        return re.sub(r'\s+', ' ', text).strip()

    pos_list = [{'text': clean_text(c['text']), 'like_count': c['like_count']} for _, c in top_positive_comments.iterrows()]
    neg_list = [{'text': clean_text(c['text']), 'like_count': c['like_count']} for _, c in top_negative_comments.iterrows()]

    return pos_list, neg_list
