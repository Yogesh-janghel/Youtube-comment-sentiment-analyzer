import os
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def create_plots(df, save_dir='static/images'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Like distribution
    fig = px.histogram(df, x='like_count', nbins=30, title='Distribution of Comment Likes')
    fig.update_layout(xaxis_title='Like Count', yaxis_title='Number of Comments')
    fig.write_image(os.path.join(save_dir, 'like_distribution.png'))

    # Comment length vs likes correlation
    fig = px.scatter(df, x='comment_length', y='like_count', opacity=0.5, title='Correlation Between Comment Length and Likes')
    fig.update_layout(xaxis_title='Comment Length', yaxis_title='Like Count')
    fig.write_image(os.path.join(save_dir, 'comment_length_vs_likes.png'))

    # Comment activity over time
    df['published_at'] = pd.to_datetime(df['published_at'])
    comment_activity = df.set_index('published_at').resample('D').size().reset_index(name='count')
    fig = px.line(comment_activity, x='published_at', y='count', title='Comment Activity Over Time')
    fig.update_layout(xaxis_title='Date', yaxis_title='Number of Comments')
    fig.write_image(os.path.join(save_dir, 'comment_activity_over_time.png'))

    # Top 10 most active authors
    top_authors = df['author'].value_counts().head(10).reset_index()
    top_authors.columns = ['author', 'count']
    fig = px.bar(top_authors, x='author', y='count', title='Top 10 Most Active Authors')
    fig.update_layout(xaxis_title='Author', yaxis_title='Number of Comments')
    fig.write_image(os.path.join(save_dir, 'top_authors.png'))

    # Comment length distribution
    fig = px.histogram(df, x='comment_length', nbins=30, title='Distribution of Comment Lengths')
    fig.update_layout(xaxis_title='Comment Length', yaxis_title='Number of Comments')
    fig.write_image(os.path.join(save_dir, 'comment_length_distribution.png'))

    # Comment activity by hour
    df['hour'] = df['published_at'].dt.hour
    comment_hours = df.groupby('hour').size().reset_index(name='count')
    fig = px.density_heatmap(comment_hours, x='hour', y='count', title='Comment Activity by Hour')
    fig.update_layout(xaxis_title='Hour of the Day', yaxis_title='Number of Comments')
    fig.write_image(os.path.join(save_dir, 'comment_activity_by_hour.png'))

    # Heatmap of Comment Activity by Day and Hour
    df['day_of_week'] = df['published_at'].dt.day_name()
    heatmap_data = df.groupby(['day_of_week', 'hour']).size().reset_index(name='count')
    fig = px.density_heatmap(heatmap_data, x='hour', y='day_of_week', z='count', title='Comment Activity by Day and Hour')
    fig.update_layout(xaxis_title='Hour of the Day', yaxis_title='Day of the Week')
    fig.write_image(os.path.join(save_dir, 'comment_activity_heatmap.png'))

    # Time Series Analysis of Likes
    likes_over_time = df.groupby(df['published_at'].dt.date)['like_count'].sum().reset_index()
    fig = px.line(likes_over_time, x='published_at', y='like_count', title='Likes Over Time')
    fig.update_layout(xaxis_title='Date', yaxis_title='Total Likes')
    fig.write_image(os.path.join(save_dir, 'likes_over_time.png'))

    # Pie Chart of Sentiment Distribution
    sentiment_counts = df['sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['sentiment', 'count']
    sentiment_labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    sentiment_counts['sentiment'] = sentiment_counts['sentiment'].map(sentiment_labels)
    fig = px.pie(sentiment_counts, values='count', names='sentiment', title='Sentiment Distribution of Comments')
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.write_image(os.path.join(save_dir, 'sentiment_distribution.png'))

    # Word cloud
    text = ' '.join(df['text'].tolist())
    if text.strip():
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Most Common Words in Comments')
        plt.savefig(os.path.join(save_dir, 'wordcloud.png'))
        plt.close()

def generate_single_wordcloud(text, save_path):
    if not text.strip():
        return
    wordcloud = WordCloud(width=400, height=200, background_color='white').generate(text)
    wordcloud.to_file(save_path)
