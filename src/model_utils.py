import os
import re
import pickle
import warnings
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from langdetect import detect
from googletrans import Translator
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning

# Suppress BeautifulSoup warning
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

# Ensure NLTK data is available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Paths
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'lstm_sentiment_model.h5')
TOKENIZER_PATH = os.path.join(MODELS_DIR, 'tokenizer.pickle')

# Global variables for lazy loading
_model_lstm = None
_tokenizer = None
_translator = None
_lemmatizer = None
_stop_words = None

def get_model_and_tokenizer():
    """Lazy load the LSTM model and tokenizer on first use"""
    global _model_lstm, _tokenizer
    if _model_lstm is None:
        _model_lstm = load_model(MODEL_PATH)
    if _tokenizer is None:
        with open(TOKENIZER_PATH, 'rb') as handle:
            _tokenizer = pickle.load(handle)
    return _model_lstm, _tokenizer

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

def perform_sentiment_analysis(df, max_seq_length=100):
    model_lstm, tokenizer = get_model_and_tokenizer()
    
    # Clean and preprocess comments
    df['cleaned_text'] = df['text'].apply(clean_and_preprocess_comments)
    
    sentiments = []
    # Vectorized approach is better, but keeping logic consistent for now
    for text in df['cleaned_text']:
        new_sequence = tokenizer.texts_to_sequences([text])
        new_padded = pad_sequences(new_sequence, maxlen=max_seq_length)
        lstm_pred = model_lstm.predict(new_padded, verbose=0)
        sentiment_score = np.argmax(lstm_pred)
        sentiments.append(sentiment_score)
    
    df['sentiment'] = sentiments
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
