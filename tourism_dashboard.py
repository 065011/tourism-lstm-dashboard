import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import feedparser
import re

st.set_page_config(page_title="Tourism Analytics Dashboard", layout="wide")

st.title("🌍 Tourism Forecast and Sentiment Analytics Dashboard")

st.write("This dashboard analyzes tourism demand trends and tourist sentiment using machine learning and NLP techniques.")

analyzer = SentimentIntensityAnalyzer()

# ------------------------------------------------
# TOURISM DEMAND DATA
# ------------------------------------------------

st.header("📈 Tourism Demand Forecast")

data = pd.read_csv(
"https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
)

fig = px.line(
data,
x="Month",
y="Passengers",
title="Tourism Demand Trend"
)

st.plotly_chart(fig)

# KPI METRICS

col1, col2, col3 = st.columns(3)

col1.metric("Total Records", len(data))

col2.metric("Average Tourists", int(data["Passengers"].mean()))

col3.metric("Max Tourists", int(data["Passengers"].max()))

# ------------------------------------------------
# SENTIMENT ANALYSIS
# ------------------------------------------------

st.divider()

st.header("🧳 Tourist Review Sentiment Analysis")

review = st.text_area("Enter Tourist Review")

if review != "":

    clean = re.sub(r'[^a-zA-Z\s]','',review.lower())

    score = analyzer.polarity_scores(clean)

    compound = score["compound"]

    pos = score["pos"]
    neg = score["neg"]
    neu = score["neu"]

    if compound >= 0.05:
        sentiment = "Positive"
        st.success("😊 Positive Sentiment")

    elif compound <= -0.05:
        sentiment = "Negative"
        st.error("😡 Negative Sentiment")

    else:
        sentiment = "Neutral"
        st.warning("😐 Neutral Sentiment")

    st.write("Sentiment Score:", compound)

    # KPI
    st.subheader("Sentiment Probability")

    col1,col2,col3 = st.columns(3)

    col1.metric("Positive",round(pos,3))
    col2.metric("Neutral",round(neu,3))
    col3.metric("Negative",round(neg,3))

    # Bar Chart

    fig2,ax = plt.subplots()

    labels = ["Negative","Neutral","Positive"]
    values = [neg,neu,pos]

    ax.bar(labels,values)

    ax.set_ylabel("Score")

    st.pyplot(fig2)

# ------------------------------------------------
# WORD FREQUENCY
# ------------------------------------------------

    st.subheader("Word Frequency Analysis")

    words = clean.split()

    freq = Counter(words)

    common = freq.most_common(10)

    w = [i[0] for i in common]
    c = [i[1] for i in common]

    fig3,ax = plt.subplots()

    ax.bar(w,c)

    ax.set_title("Top Words")

    st.pyplot(fig3)

# ------------------------------------------------
# WORD CLOUD
# ------------------------------------------------

    st.subheader("Word Cloud")

    wc = WordCloud(width=800,height=400,background_color="white").generate(clean)

    fig4,ax = plt.subplots()

    ax.imshow(wc)

    ax.axis("off")

    st.pyplot(fig4)

# ------------------------------------------------
# SAMPLE TOURISM DATASET
# ------------------------------------------------

st.divider()

st.header("📊 Sample Tourism Sentiment Dataset")

sample_reviews = [

"Beautiful beaches and wonderful hospitality",

"Terrible hotel service and dirty rooms",

"Amazing sightseeing experience",

"Overcrowded tourist attractions",

"Great food and friendly locals",

"Bad weather ruined the trip",

"Fantastic cultural heritage sites",

"Transportation was slow",

"Excellent travel experience",

"Average tourist services"

]

df = pd.DataFrame(sample_reviews,columns=["review"])

scores=[]
sentiments=[]

for text in df["review"]:

    s = analyzer.polarity_scores(text)["compound"]

    scores.append(s)

    if s>=0.05:

        sentiments.append("Positive")

    elif s<=-0.05:

        sentiments.append("Negative")

    else:

        sentiments.append("Neutral")

df["sentiment_score"]=scores
df["sentiment"]=sentiments

st.dataframe(df)

# PIE CHART

st.subheader("Tourist Sentiment Distribution")

counts = df["sentiment"].value_counts()

fig5,ax = plt.subplots()

ax.pie(counts,labels=counts.index,autopct="%1.1f%%")

st.pyplot(fig5)

# ------------------------------------------------
# SENTIMENT TREND
# ------------------------------------------------

st.subheader("📉 Sentiment Trend Analysis")

trend_df = pd.DataFrame({

"Index":range(len(scores)),
"Sentiment Score":scores

})

fig6,ax = plt.subplots()

ax.plot(trend_df["Index"],trend_df["Sentiment Score"],marker="o")

ax.set_xlabel("Review Index")
ax.set_ylabel("Sentiment Score")

st.pyplot(fig6)

# ------------------------------------------------
# TOURISM NEWS SENTIMENT
# ------------------------------------------------

st.divider()

st.header("📰 Tourism Industry News")

rss_url = "https://news.google.com/rss/search?q=tourism"

feed = feedparser.parse(rss_url)

articles = feed.entries[:5]

for article in articles:

    title = article.title

    link = article.link

    score = analyzer.polarity_scores(title)["compound"]

    if score>=0.05:

        label="Positive"

    elif score<=-0.05:

        label="Negative"

    else:

        label="Neutral"

    st.write("###",title)

    st.write("Sentiment:",label)

    st.write(link)

    st.write("---")

st.write("Dashboard developed using Streamlit for Tourism Analytics.")