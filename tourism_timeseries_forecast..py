import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


print("Loading tourism demand dataset")

data = pd.read_csv(
"https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
)

print(data.head())

values = data["Passengers"].values.reshape(-1,1)

scaler = MinMaxScaler()

scaled = scaler.fit_transform(values)


def create_dataset(data,window):

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


print("Training tourism forecast model")

for epoch in range(20):

    output=model(X_train)

    loss=criterion(output,y_train)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if (epoch+1)%5==0:

        print("Epoch",epoch+1,"Loss",loss.item())


with torch.no_grad():

    pred=model(X_test).numpy()

pred=scaler.inverse_transform(pred)

actual=scaler.inverse_transform(y_test.numpy())

rmse=np.sqrt(mean_squared_error(actual,pred))

print("RMSE:",rmse)


plt.figure(figsize=(10,5))

plt.plot(values,label="Actual Tourism Demand")

forecast_plot=np.empty_like(values,dtype=float)
forecast_plot[:]=np.nan

forecast_plot[train_size+window:]=pred

plt.plot(forecast_plot,label="Forecast")

plt.legend()

plt.title("Tourism Demand Forecast")

plt.show()