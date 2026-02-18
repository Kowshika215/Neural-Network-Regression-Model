# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

This code builds and trains a feedforward neural network in PyTorch for a regression task. The model takes a single input feature, passes it through two hidden layers with ReLU activation, and predicts one continuous output. It uses MSE loss and RMSProp optimizer to minimize the error between predictions and actual values over training epochs.

## Neural Network Model

<img width="999" height="648" alt="Screenshot 2026-02-18 135035" src="https://github.com/user-attachments/assets/b1d5d2e9-dd7e-4006-a7bb-9602ca67fb32" />


## DESIGN STEPS

### STEP 1:

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot.

### STEP 7:

Evaluate the model with the testing data.

### STEP 8:
Use the trained model to predict for a new input value .

## PROGRAM
### Name: Kowshika R
### Register Number: 212224220049
```

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

dataset1 = pd.read_csv('sample.csv')
X = dataset1[['Input']].values
y = dataset1[['Output']].values

dataset1.head(5)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=33)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Name: KOWSHIKA R
# Register Number: 212224220049
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1,8)
        self.fc2 = nn.Linear(8,10)
        self.fc3 = nn.Linear(10,1)
        self.relu = nn.ReLU()
        self.history = {'loss': []}

  def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the Model, Loss Function, and Optimizer
kowshi = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(kowshi.parameters(), lr=0.001)

# Name: KOWSHIKA R
# Register Number: 212224230106
def train_model(kowshi, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(kowshi(X_train), y_train)
        loss.backward()
        optimizer.step()


        # Append loss inside the loop
        kowshi.history['loss'].append(loss.item())

        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')


train_model(kowshi, X_train_tensor, y_train_tensor, criterion, optimizer)

with torch.no_grad():
    test_loss = criterion(kowshi(X_test_tensor), y_test_tensor)
    print(f'Test Loss: {test_loss.item():.6f}')

loss_df = pd.DataFrame(kowshi.history)

import matplotlib.pyplot as plt
print("\nName:KOWSHIKA R")
print("Register Number:212224220049")
loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()

X_n1_1 = torch.tensor([[9]], dtype=torch.float32)
prediction = kowshi(torch.tensor(scaler.transform(X_n1_1), dtype=torch.float32)).item()
print("Name:KOWSHIKA R")
print("Register Number:212224220049")
print(f'Prediction: {prediction}')


```
## Dataset Information

<img width="482" height="384" alt="Screenshot 2026-02-18 134102" src="https://github.com/user-attachments/assets/aa43d8c0-3dd9-4165-94e1-bc3f77b94bf0" />

## OUTPUT

 <img width="446" height="302" alt="Screenshot 2026-02-18 134629" src="https://github.com/user-attachments/assets/d4a4d464-bb87-478c-bc20-b10675fd9092" />



### Training Loss Vs Iteration Plot

<img width="739" height="620" alt="Screenshot 2026-02-18 134656" src="https://github.com/user-attachments/assets/96058a1d-4b5b-4c14-903c-9bfa4b82d0ee" />


### New Sample Data Prediction

<img width="379" height="72" alt="Screenshot 2026-02-18 134826" src="https://github.com/user-attachments/assets/0bd66ca1-4c6e-44a3-8e07-a2dd5532228a" />



## RESULT

Thus, a neural network regression model was successfully developed and trained using PyTorch.
