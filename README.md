# Developing a Neural Network Regression Model

## AIM


To develop a neural network regression model for the given dataset.

## THEORY


Developing a neural network regression model involves designing a feedforward network with fully connected layers to predict continuous values. The model is trained using a loss function like Mean Squared Error (MSE)
and optimized with algorithms like RMSprop or Adam. 

## Neural Network Model


![image](https://github.com/user-attachments/assets/aba5dc49-7078-4d84-84bd-0ced47c7c1e4)


## DESIGN STEPS


### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.


## PROGRAM

### Name: Kowsalya M
### Register Number: 212222230069

```
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

df1=pd.read_csv("/content/nn-dl-exp.csv")
X = df1[['input']].values
y = df1[['output']].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=33)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test =  scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(1,10)
        self.fc2=nn.Linear(10,18)
        self.fc3=nn.Linear(18,1)
        self.relu=nn.ReLU()
        self.history={'loss':[]}

    def forward(self,x):
        x=self.relu(self.fc1(x))
        x=self.relu(self.fc2(x))
        x=self.fc3(x)
        return x

# Initialize the Model, Loss Function, and Optimizer
ai_brain=NeuralNet()
criterion=nn.MSELoss()
optimizer=optim.RMSprop(ai_brain.parameters(),lr=0.001)


def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
    optimizer.zero_grad()
    loss=criterion(ai_brain(X_train),y_train)
    loss.backward()
    optimizer.step()

    ai_brain.history['loss'].append(loss.item())
    if epoch%200==0:
      print(f'Epoch [{epoch}/{epochs}], Loss:{loss.item():.6f}')

with torch.no_grad():
    test_loss = criterion(ai_brain(X_test_tensor), y_test_tensor)
    print(f'Test Loss: {test_loss.item():.6f}')

loss_df = pd.DataFrame(ai_brain.history)

import matplotlib.pyplot as plt
loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()


X_n1_1 = torch.tensor([[3]], dtype=torch.float32)
prediction = ai_brain(torch.tensor(scaler.transform(X_n1_1), dtype=torch.float32)).item()
print(f'Prediction: {prediction}')
     

```
## Dataset Information

![data-info-dl](https://github.com/user-attachments/assets/b8bb69d8-ea10-44f2-8e1b-3115c10b7ce1)


## OUTPUT


### Training Loss Vs Iteration Plot


![image](https://github.com/user-attachments/assets/54eff130-8d03-4ae7-8e1f-f21e9465fcab)



### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/d6d202d2-37d1-4d7f-9f91-312f20a9dc7e)


## RESULT


The neural network regression model successfully learns the mapping between input and output, reducing the loss over training epochs. The model demonstrated strong predictive performance on unseen data, with a low error rate.
