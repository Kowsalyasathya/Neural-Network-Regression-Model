# Developing a Neural Network Regression Model

## AIM


To develop a neural network regression model for the given dataset.

## THEORY


Developing a neural network regression model involves designing a feedforward network with fully connected layers to predict continuous values. The model is trained using a loss function like Mean Squared Error (MSE)
and optimized with algorithms like RMSprop or Adam. 

## Neural Network Model


![modell-dl](https://github.com/user-attachments/assets/6493433a-6e70-4e13-ae7f-6eb931b7309c)

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
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(1,4)
        self.fc2=nn.Linear(4,8)
        self.fc3=nn.Linear(8,1)
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
    if epochs%200==0:
      print(f'Epoch [{epoch}/{epochs}], Loss:{loss.item():.6f}')

```

## Dataset Information

![data-info-dl](https://github.com/user-attachments/assets/b8bb69d8-ea10-44f2-8e1b-3115c10b7ce1)



## OUTPUT


### Training Loss Vs Iteration Plot


![image](https://github.com/user-attachments/assets/6629cff9-aa76-4c56-a367-83c810186352)

### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/dd7d644f-8c1d-46ea-b8be-fab0f7782a18)


![image](https://github.com/user-attachments/assets/ef4ae5eb-b672-48f9-989f-b217209e40fe)

## RESULT


The neural network regression model successfully learns the mapping between input and output, reducing the loss over training epochs. The model demonstrated strong predictive performance on unseen data, with a low error rate.
