import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import torch
import torch.nn as nn

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from wordcloud import WordCloud
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import feedparser
import re

st.set_page_config(page_title="Tourism Analytics Dashboard", layout="wide")

st.title("🌍 Tourism Forecast and Sentiment Analytics Dashboard")

st.write("This dashboard analyzes tourism demand trends and tourist sentiment using Machine Learning and NLP.")

# ------------------------------------------------
# LOAD TOURISM DATASET
# ------------------------------------------------

data = pd.read_csv(
"https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
)

values = data["Passengers"].values.reshape(-1,1)

# ------------------------------------------------
# TOURISM DEMAND VISUALIZATION
# ------------------------------------------------

st.header("📈 Tourism Demand Trend")

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
col3.metric("Maximum Tourists", int(data["Passengers"].max()))

# ------------------------------------------------
# LSTM TOURISM FORECAST MODEL
# ------------------------------------------------

st.subheader("🔮 Tourism Demand Forecast using LSTM")

scaler = MinMaxScaler()

scaled = scaler.fit_transform(values)

def create_dataset(data, window):

    X=[]
    y=[]

    for i in range(len(data)-window):

        X.append(data[i:i+window])
        y.append(data[i+window])

    return np.array(X),np.array(y)

window = 10

X,y = create_dataset(scaled,window)

train_size=int(len(X)*0.8)

X_train=X[:train_size]
X_test=X[train_size:]

y_train=y[:train_size]
y_test=y[train_size:]

X_train=torch.tensor(X_train,dtype=torch.float32)
X_test=torch.tensor(X_test,dtype=torch.float32)

y_train=torch.tensor(y_train,dtype=torch.float32)
y_test=torch.tensor(y_test,dtype=torch.float32)

class TourismLSTM(nn.Module):

    def __init__(self):

        super(TourismLSTM,self).__init__()

        self.lstm=nn.LSTM(1,50,batch_first=True)

        self.fc=nn.Linear(50,1)

    def forward(self,x):

        out,_=self.lstm(x)

        out=out[:,-1,:]

        out=self.fc(out)

        return out

model=TourismLSTM()

criterion=nn.MSELoss()

optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

for epoch in range(20):

    output=model(X_train)

    loss=criterion(output,y_train)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

# ------------------------------------------------
# FORECAST PREDICTION
# ------------------------------------------------

with torch.no_grad():

    pred=model(X_test).numpy()

pred=scaler.inverse_transform(pred)

actual=scaler.inverse_transform(y_test.numpy())

rmse=np.sqrt(mean_squared_error(actual,pred))

st.metric("Forecast RMSE", round(rmse,2))

# ------------------------------------------------
# FORECAST GRAPH
# ------------------------------------------------

forecast_plot=np.empty_like(values,dtype=float)

forecast_plot[:]=np.nan

forecast_plot[train_size+window:]=pred

fig2,ax = plt.subplots(figsize=(10,5))

ax.plot(values,label="Actual Tourism Demand")
ax.plot(forecast_plot,label="Forecast")

ax.set_title("Tourism Demand Forecast")

ax.legend()

st.pyplot(fig2)

# ------------------------------------------------
# SENTIMENT ANALYSIS
# ------------------------------------------------

st.divider()

st.header("🧳 Tourist Review Sentiment Analysis")

analyzer = SentimentIntensityAnalyzer()

review = st.text_area("Enter Tourist Review")

if review != "":

    clean = re.sub(r'[^a-zA-Z\s]','',review.lower())

    score = analyzer.polarity_scores(clean)

    compound = score["compound"]
    pos = score["pos"]
    neg = score["neg"]
    neu = score["neu"]

    if compound >= 0.05:

        st.success("😊 Positive Sentiment")

    elif compound <= -0.05:

        st.error("😡 Negative Sentiment")

    else:

        st.warning("😐 Neutral Sentiment")

    st.write("Sentiment Score:", compound)

    st.subheader("Sentiment Probability")

    col1,col2,col3 = st.columns(3)

    col1.metric("Positive",round(pos,3))
    col2.metric("Neutral",round(neu,3))
    col3.metric("Negative",round(neg,3))

    fig3,ax = plt.subplots()

    labels = ["Negative","Neutral","Positive"]
    values_sent = [neg,neu,pos]

    ax.bar(labels,values_sent)

    ax.set_ylabel("Score")

    st.pyplot(fig3)

# ------------------------------------------------
# WORD FREQUENCY
# ------------------------------------------------

    st.subheader("Word Frequency")

    words = clean.split()

    freq = Counter(words)

    common = freq.most_common(10)

    w = [i[0] for i in common]
    c = [i[1] for i in common]

    fig4,ax = plt.subplots()

    ax.bar(w,c)

    ax.set_title("Top Words")

    st.pyplot(fig4)

# ------------------------------------------------
# WORD CLOUD
# ------------------------------------------------

    st.subheader("Word Cloud")

    wc = WordCloud(width=800,height=400,background_color="white").generate(clean)

    fig5,ax = plt.subplots()

    ax.imshow(wc)

    ax.axis("off")

    st.pyplot(fig5)

# ------------------------------------------------
# SAMPLE DATASET SENTIMENT
# ------------------------------------------------

st.divider()

st.header("📊 Sample Tourism Review Dataset")

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

st.subheader("Sentiment Distribution")

counts = df["sentiment"].value_counts()

fig6,ax = plt.subplots()

ax.pie(counts,labels=counts.index,autopct="%1.1f%%")

st.pyplot(fig6)

# ------------------------------------------------
# SENTIMENT TREND
# ------------------------------------------------

st.subheader("Sentiment Trend")

trend_df = pd.DataFrame({

"Index":range(len(scores)),
"Sentiment Score":scores

})

fig7,ax = plt.subplots()

ax.plot(trend_df["Index"],trend_df["Sentiment Score"],marker="o")

ax.set_xlabel("Review Index")
ax.set_ylabel("Sentiment Score")

st.pyplot(fig7)

# ------------------------------------------------
# TOURISM NEWS SENTIMENT
# ------------------------------------------------

st.divider()

st.header("📰 Tourism Industry News Sentiment")

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