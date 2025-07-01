import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
from googleapiclient.discovery import build
from datetime import datetime, timedelta
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from tqdm import tqdm

# Download NLTK data
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')

# ---- API Setup ----
load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")
youtube = build('youtube', 'v3', developerKey=API_KEY)

# ---- Helper Functions ----
def extract_channel_id(channel_url):
    if "channel/" in channel_url:
        return channel_url.split("channel/")[1].split("/")[0]
    elif "user/" in channel_url or "@" in channel_url:
        username = channel_url.split("/")[-1].replace("@", "")
        res = youtube.search().list(part='snippet', q=username, type='channel', maxResults=1).execute()
        return res['items'][0]['snippet']['channelId']
    else:
        raise ValueError("Invalid channel URL format")

def get_uploads_playlist_id(channel_id):
    res = youtube.channels().list(part='contentDetails', id=channel_id).execute()
    return res['items'][0]['contentDetails']['relatedPlaylists']['uploads']

def get_video_stats(video_id):
    try:
        res = youtube.videos().list(part='statistics', id=video_id).execute()
        stats = res['items'][0]['statistics']
        return {
            'likes': int(stats.get('likeCount', 0)),
            'views': int(stats.get('viewCount', 0))
        }
    except:
        return {'likes': 0, 'views': 0}

def get_recent_videos(playlist_id):
    videos = []
    cutoff = datetime.utcnow() - timedelta(days=90)
    next_page = None

    while True:
        res = youtube.playlistItems().list(
            part='snippet',
            playlistId=playlist_id,
            maxResults=50,
            pageToken=next_page
        ).execute()

        for item in res['items']:
            pub = item['snippet']['publishedAt']
            pub_date = datetime.strptime(pub, "%Y-%m-%dT%H:%M:%SZ")
            if pub_date >= cutoff:
                vid = item['snippet']['resourceId']['videoId']
                title = item['snippet']['title']
                stats = get_video_stats(vid)
                videos.append({
                    'video_id': vid,
                    'title': title,
                    'likes': stats['likes'],
                    'views': stats['views']
                })

        next_page = res.get('nextPageToken')
        if not next_page:
            break

    return videos

def get_comments(video_id):
    comments = []
    next_page_token = None

    while True:
        try:
            res = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=next_page_token,
                textFormat="plainText"
            ).execute()

            for item in res['items']:
                text = item['snippet']['topLevelComment']['snippet']['textDisplay']
                comments.append(text)

            next_page_token = res.get('nextPageToken')
            if not next_page_token:
                break
        except:
            break
    return comments

def analyze_sentiments(comments):
    sia = SentimentIntensityAnalyzer()
    pos, neg, neu = [], [], []

    for comment in comments:
        score = sia.polarity_scores(comment)['compound']
        if score >= 0.05:
            pos.append(comment)
        elif score <= -0.05:
            neg.append(comment)
        else:
            neu.append(comment)

    return pos, neg, neu

def extract_keywords(text, num_keywords=10):
    if not text.strip():
        return "N/A"
    try:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
        tfidf = vectorizer.fit_transform([text])
        words = vectorizer.get_feature_names_out()
        scores = tfidf.toarray()[0]
        ranked = sorted(zip(scores, words), reverse=True)[:num_keywords]
        return ", ".join([w for _, w in ranked]) if ranked else "N/A"
    except ValueError:
        return "N/A"

def generate_summary(university_name, channel_url):
    channel_id = extract_channel_id(channel_url)
    playlist_id = get_uploads_playlist_id(channel_id)
    videos = get_recent_videos(playlist_id)

    all_video_ids, all_titles = [], []
    total_likes, total_views = 0, 0
    all_comments = []

    for video in videos:
        all_video_ids.append(video['video_id'])
        all_titles.append(video['title'])
        total_likes += video['likes']
        total_views += video['views']
        comments = get_comments(video['video_id'])
        all_comments.extend(comments)

    pos, neg, neu = analyze_sentiments(all_comments)
    full_text = " ".join(all_comments)
    keywords = extract_keywords(full_text)

    return {
        "university_name": university_name,
        "video_ids": ", ".join(all_video_ids),
        "titles": ", ".join(all_titles),
        "total_videos": len(videos),
        "total_comments": len(all_comments),
        "All_comments": " ||| ".join(all_comments),
        "positive_comments": " ||| ".join(pos),
        "negative_comments": " ||| ".join(neg),
        "neutral_comments": " ||| ".join(neu),
        "sentiment_overall": f"{len(pos)}/{len(neg)}/{len(neu)}",
        "keywords": keywords,
        "total_likes": total_likes,
        "total_views": total_views
    }

# ---- Streamlit UI ----

st.set_page_config(page_title="YouTube University Analyzer", layout="wide")

st.title("YouTube Analyzer")
st.markdown("Analyze recent YouTube videos and comments.")

with st.form("input_form"):
    university = st.text_input("Enter University Name")
    channel_url = st.text_input("Enter YouTube Channel URL")
    submitted = st.form_submit_button("Analyze")

if submitted:
    try:
        with st.spinner("Processing..."):
            result = generate_summary(university, channel_url)
            df = pd.DataFrame([result])
            st.success("✅ Analysis complete!")
            st.dataframe(df.drop(columns=["All_comments", "positive_comments", "negative_comments", "neutral_comments"]))

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download CSV", csv, "university_youtube_summary.csv", "text/csv")

            with st.expander("🔍 See All Comments"):
                st.write(result["All_comments"])

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
