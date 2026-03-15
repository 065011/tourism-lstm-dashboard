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

st.write("Tourism demand forecasting and sentiment analytics using AI and NLP.")

# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------

@st.cache_data
def load_data():
    data = pd.read_csv(
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
    )
    return data

data = load_data()
values = data["Passengers"].values.reshape(-1,1)

# ------------------------------------------------
# TOURISM TREND
# ------------------------------------------------

st.header("📈 Tourism Demand Trend")

fig = px.line(data, x="Month", y="Passengers")
st.plotly_chart(fig)

col1,col2,col3 = st.columns(3)

col1.metric("Total Records", len(data))
col2.metric("Average Tourists", int(data["Passengers"].mean()))
col3.metric("Maximum Tourists", int(data["Passengers"].max()))

# ------------------------------------------------
# LSTM FORECAST MODEL
# ------------------------------------------------

st.subheader("🔮 Tourism Demand Forecast (LSTM)")

@st.cache_resource
def train_model(values):

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

    with torch.no_grad():
        pred=model(X_test).numpy()

    pred=scaler.inverse_transform(pred)
    actual=scaler.inverse_transform(y_test.numpy())

    rmse=np.sqrt(mean_squared_error(actual,pred))

    return pred,actual,rmse,train_size,window

pred,actual,rmse,train_size,window = train_model(values)

st.metric("Forecast RMSE",round(rmse,2))

forecast_plot=np.empty_like(values,dtype=float)
forecast_plot[:]=np.nan
forecast_plot[train_size+window:]=pred

fig2,ax=plt.subplots()

ax.plot(values,label="Actual")
ax.plot(forecast_plot,label="Forecast")

ax.legend()
ax.set_title("Tourism Demand Forecast")

st.pyplot(fig2)

# ------------------------------------------------
# SENTIMENT ANALYSIS
# ------------------------------------------------

st.divider()

st.header("🧳 Tourist Review Sentiment")

analyzer = SentimentIntensityAnalyzer()

review = st.text_area("Enter Review")

if review:

    clean = re.sub(r'[^a-zA-Z\s]','',review.lower())

    score = analyzer.polarity_scores(clean)

    compound=score["compound"]

    if compound>=0.05:
        st.success("Positive Sentiment")
    elif compound<=-0.05:
        st.error("Negative Sentiment")
    else:
        st.warning("Neutral Sentiment")

    st.write("Score:",compound)

    pos,neu,neg = score["pos"],score["neu"],score["neg"]

    fig3,ax=plt.subplots()

    ax.bar(["Negative","Neutral","Positive"],[neg,neu,pos])

    st.pyplot(fig3)

    # Word Frequency

    words = clean.split()

    freq = Counter(words).most_common(10)

    w=[i[0] for i in freq]
    c=[i[1] for i in freq]

    fig4,ax=plt.subplots()

    ax.bar(w,c)
    st.pyplot(fig4)

    # Word Cloud

    wc = WordCloud(width=800,height=400).generate(clean)

    fig5,ax=plt.subplots()

    ax.imshow(wc)
    ax.axis("off")

    st.pyplot(fig5)

# ------------------------------------------------
# NEWS SENTIMENT
# ------------------------------------------------

st.divider()

st.header("📰 Tourism News")

rss_url="https://news.google.com/rss/search?q=tourism"

feed=feedparser.parse(rss_url)

for article in feed.entries[:5]:

    title=article.title
    link=article.link

    score=analyzer.polarity_scores(title)["compound"]

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