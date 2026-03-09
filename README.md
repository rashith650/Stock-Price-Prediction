# Stock-Price-Prediction


## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset

Stock price prediction is an important task in financial analysis. The goal is to predict future stock prices based on historical data. In this experiment, historical stock price data is used to train a Recurrent Neural Network model. The dataset contains stock information such as the closing price over time. The model learns patterns from past prices and predicts future stock prices.

## Design Steps

### Step 1:

Load the stock price dataset and select the closing price values. Normalize the data using MinMaxScaler to scale the values between 0 and 1 for better training performance.

### Step 2:

Convert the time-series data into sequences so that the RNN can learn temporal patterns. Each sequence contains a fixed number of previous time steps used to predict the next stock price.

### Step 3:

Define the RNN model using PyTorch, train the model using the training dataset, and evaluate its performance by predicting stock prices on the test dataset. Finally, compare the actual stock prices and predicted prices using a plot.

#### Name: Mohamed Rashith S
#### Register Number: 212223243003
Include your code here
```Python 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

df_train = pd.read_csv('/content/trainset.csv')
df_test = pd.read_csv('/content/testset.csv')

train_prices = df_train['Close'].values.reshape(-1, 1)
test_prices = df_test['Close'].values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))

scaled_train = scaler.fit_transform(train_prices)
scaled_test = scaler.transform(test_prices)

def create_sequences(data, seq_length):
    x = []
    y = []

    for i in range(len(data) - seq_length):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length])

    return np.array(x), np.array(y)


seq_length = 60

x_train, y_train = create_sequences(scaled_train, seq_length)
x_test, y_test = create_sequences(scaled_test, seq_length)

x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(x_train_tensor, y_train_tensor)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = RNNModel().to(device)

print("Using device:", device)

import torch
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=1,
            hidden_size=64,
            num_layers=2,
            batch_first=True
        )
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 64)
        out, _ = self.rnn(x, h0)
        out = out[:, -1, :]
        return self.fc(out)

model = RNNModel()

print("Model input_size:", model.rnn.input_size)
!pip install torchinfo
from torchinfo import summary
!pip install torchinfo
from torchinfo import summary
from torchsummary import summary

model = RNNModel(input_size=10, hidden_size=32, num_layers=2, output_size=1)

summary(model, input_size=(5, 10))
criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)
print("Input tensor shape:", x_train_tensor.shape)
print("Model expects input_size:", model.rnn.input_size)
num_epochs = 30
train_losses = []

for epoch in range(num_epochs):

    model.train()
    running_loss = 0.0

    for inputs, targets in train_loader:

        inputs = inputs.to(device)           # shape: (batch, 60, 1)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)              # forward pass

        loss = criterion(outputs, targets)   # MSE loss

        loss.backward()                      # backprop

        optimizer.step()                     # update weights

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.6f}")
print('Name: Mohamed Rashith S ')
print('Register Number: 212223243003')

plt.figure(figsize=(8,5))
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.grid()
plt.show()

model.eval()

with torch.no_grad():

    predicted = model(x_test_tensor.to(device))
    predicted = predicted.cpu().numpy()

    actual = y_test_tensor.cpu().numpy()
model.eval()

with torch.no_grad():
    predicted = model(x_test_tensor.to(device)).cpu().numpy()
    actual = y_test_tensor.cpu().numpy()

predicted_prices = scaler.inverse_transform(predicted)
actual_prices = scaler.inverse_transform(actual)
print('Name: Mohamed Rashith S ')
print('Register Number: 212223243003')

plt.figure(figsize=(10,6))

plt.plot(actual_prices, label='Actual Price')
plt.plot(predicted_prices, label='Predicted Price')

plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Stock Price Prediction using RNN')

plt.legend()
plt.grid()

plt.show()
print(f"Last Predicted Price: {predicted_prices[-1][0]:.2f}")
print(f"Last Actual Price:    {actual_prices[-1][0]:.2f}")
```

## Output

### True Stock Price, Predicted Stock Price vs time

<img width="629" height="243" alt="image" src="https://github.com/user-attachments/assets/9e09e2b7-5eb0-4c84-bf76-e41b6b845fcf" />


<img width="810" height="480" alt="image" src="https://github.com/user-attachments/assets/367cfc0d-7fed-40d0-85a5-05c06cd6b331" />



<img width="1034" height="529" alt="image" src="https://github.com/user-attachments/assets/f8431de8-3fcb-436f-8401-347f811fb969" />




### Predictions 

<img width="466" height="40" alt="image" src="https://github.com/user-attachments/assets/865c5d7d-1690-4994-a97e-443be7119be9" />



## Result

Thus, a Recurrent Neural Network (RNN) model was successfully developed for stock price prediction. The model was trained using historical stock price data and the predicted prices were compared with the actual prices using a graph, demonstrating the model’s ability to learn patterns in time-series data.
