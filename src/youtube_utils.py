import os
from urllib.parse import urlparse, parse_qs
import googleapiclient.discovery
import googleapiclient.errors
import pandas as pd

def get_youtube_client(developer_key):
    api_service_name = "youtube"
    api_version = "v3"
    return googleapiclient.discovery.build(api_service_name, api_version, developerKey=developer_key)

def extract_video_id(url):
    parsed_url = urlparse(url)
    if parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
        if parsed_url.path == '/watch':
            query_params = parse_qs(parsed_url.query)
            return query_params.get('v', [None])[0]
    if parsed_url.hostname == 'youtu.be':
        return parsed_url.path[1:]
    return None

def fetch_video_details(youtube, video_id):
    video_request = youtube.videos().list(
        part="snippet",
        id=video_id
    )
    video_response = video_request.execute()
    if not video_response['items']:
        return "Unknown Video"
    video_title = video_response['items'][0]['snippet']['title']
    return video_title

def fetch_comments(youtube, video_id, max_comments=100, min_comment_length=10):
    comments = []
    next_page_token = None
    while True:
        yt_request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            pageToken=next_page_token
        )
        response = yt_request.execute()

        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']
            comment_text = comment['textDisplay']
            if len(comment_text) >= min_comment_length:
                comments.append([
                    comment['authorDisplayName'],
                    comment['publishedAt'],
                    comment['updatedAt'],
                    comment['likeCount'],
                    comment_text,
                    len(comment_text)
                ])

        if len(comments) >= max_comments or not response.get('nextPageToken'):
            break
        next_page_token = response.get('nextPageToken')

    df = pd.DataFrame(comments, columns=['author', 'published_at', 'updated_at', 'like_count', 'text', 'comment_length'])
    return df
