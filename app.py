import os
# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import numpy as np
from flask import Flask, request, render_template
from dotenv import load_dotenv
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Import modularized logic
from src.model_utils import (
    get_model_and_tokenizer, 
    clean_and_preprocess_comments, 
    perform_sentiment_analysis, 
    calculate_overall_sentiment, 
    prepare_top_comments
)
from src.youtube_utils import (
    get_youtube_client, 
    extract_video_id, 
    fetch_video_details, 
    fetch_comments
)
from src.plot_utils import create_plots, generate_single_wordcloud

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configuration
DEVELOPER_KEY = os.getenv("YOUTUBE_API_KEY")
MAX_SEQ_LENGTH = 100

# Ensure required directories exist
for path in ['static/images', 'static/last_fetched']:
    if not os.path.exists(path):
        os.makedirs(path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/how_it_works')
def how_it_works():
    return render_template('how_it_works.html')

@app.route('/submit_comment', methods=['POST'])
def submit_comment():
    comment = request.form.get('comment')
    if not comment:
        return "No comment provided", 400
        
    cleaned_comment = clean_and_preprocess_comments(comment)
    model_lstm, tokenizer = get_model_and_tokenizer()
    
    # Predict sentiment
    new_sequence = tokenizer.texts_to_sequences([cleaned_comment])
    new_padded = pad_sequences(new_sequence, maxlen=MAX_SEQ_LENGTH)
    lstm_pred = model_lstm.predict(new_padded, verbose=0)
    
    sentiment_score = np.argmax(lstm_pred)
    confidence = float(lstm_pred[0][sentiment_score])
    
    sentiment_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
    sentiment_label = sentiment_labels.get(sentiment_score, "Neutral")
    
    # Generate word cloud for the single comment
    wordcloud_path = 'static/images/comment_wordcloud.png'
    generate_single_wordcloud(cleaned_comment, wordcloud_path)
    
    return render_template(
        'comment.html', 
        sentiment_label=sentiment_label, 
        sentiment_score=f"{confidence:.2%}", 
        original_comment=comment, 
        sentiment_distribution=lstm_pred[0], 
        wordcloud_image='/' + wordcloud_path
    )

@app.route('/submit_url', methods=['POST'])
def submit_url():
    youtube_url = request.form.get('youtube_url')
    num_comments = int(request.form.get('num_comments', 100))
    min_comment_length = int(request.form.get('min_comment_length', 10))

    if not youtube_url:
        return "No URL provided", 400

    if not DEVELOPER_KEY:
        return "YouTube API Key not configured. Please check your .env file.", 500

    try:
        video_id = extract_video_id(youtube_url)
        if not video_id:
            return "Invalid YouTube URL", 400

        youtube = get_youtube_client(DEVELOPER_KEY)
        video_title = fetch_video_details(youtube, video_id)
        df = fetch_comments(youtube, video_id, max_comments=num_comments, min_comment_length=min_comment_length)

        if df.empty:
            return "No comments found for this video.", 404

        # Perform sentiment analysis
        df = perform_sentiment_analysis(df, MAX_SEQ_LENGTH)
        
        # Calculate overall sentiment
        sentiment, _ = calculate_overall_sentiment(df)
        top_pos, top_neg = prepare_top_comments(df)

        # Generate all plots
        create_plots(df, save_dir='static/images')

        # Save data for "Last Fetched" functionality
        data_to_save = {
            'sentiment': sentiment,
            'video_title': video_title,
            'video_id': video_id,
            'top_positive_comments': top_pos,
            'top_negative_comments': top_neg
        }
        with open('static/last_fetched/last_viewed_data_new.json', 'w') as f:
            json.dump(data_to_save, f)

        return render_template(
            'youtube.html', 
            sentiment=sentiment, 
            video_title=video_title,
            video_id=video_id,
            top_positive_comments=top_pos,
            top_negative_comments=top_neg,
            # Image paths
            like_dist_image='/static/images/like_distribution.png',
            comment_corr_image='/static/images/comment_length_vs_likes.png',
            comment_activity_image='/static/images/comment_activity_over_time.png',
            top_authors_image='/static/images/top_authors.png',
            comment_length_dist_image='/static/images/comment_length_distribution.png',
            comment_activity_by_hour_image='/static/images/comment_activity_by_hour.png',
            wordcloud_image='/static/images/wordcloud.png',
            comment_activity_heatmap_image='/static/images/comment_activity_heatmap.png',
            likes_over_time_image='/static/images/likes_over_time.png',
            sentiment_dist_image='/static/images/sentiment_distribution.png'
        )
    except Exception as e:
        return f"An error occurred: {str(e)}", 500

@app.route('/last_fetched', methods=['POST'])
def last_fetch_fucn():
    # Attempt to load the "old" data as per original logic
    try:
        path = 'static/last_fetched/last_viewed_data_old.json'
        if not os.path.exists(path):
            # Fallback to "new" if "old" doesn't exist
            path = 'static/last_fetched/last_viewed_data_new.json'
            
        with open(path, 'r') as f:
            data = json.load(f)
    except Exception:
        return "No previously fetched data available.", 400

    return render_template(
        'last_fetch.html', 
        sentiment=data['sentiment'], 
        video_title=data['video_title'],
        video_id=data['video_id'],
        top_positive_comments=data['top_positive_comments'],
        top_negative_comments=data['top_negative_comments'],
        # Use paths as defined in original template
        like_dist_image='/static/last_fetched/like_distribution.png',
        comment_corr_image='/static/last_fetched/comment_length_vs_likes.png',
        comment_activity_image='/static/last_fetched/comment_activity_over_time.png',
        top_authors_image='/static/last_fetched/top_authors.png',
        comment_length_dist_image='/static/last_fetched/comment_length_distribution.png',
        comment_activity_by_hour_image='/static/last_fetched/comment_activity_by_hour.png',
        wordcloud_image='/static/last_fetched/wordcloud.png',
        comment_activity_heatmap_image='/static/last_fetched/comment_activity_heatmap.png',
        likes_over_time_image='/static/last_fetched/likes_over_time.png',
        sentiment_dist_image='/static/last_fetched/sentiment_distribution.png'
    )

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
