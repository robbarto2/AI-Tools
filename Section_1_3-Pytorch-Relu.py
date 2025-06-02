import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Generate training data with 2000 samples
np.random.seed(0)
torch.manual_seed(0)
x_train = torch.linspace(-5, 5, 2000).unsqueeze(1)
y_train = torch.sin(x_train) - 0.01 * x_train**2

# Define the neural network architecture
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(1, 4)  # Input to hidden layer
        self.layer2 = nn.Linear(4, 1)  # Hidden to output layer

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# Implement the training loop
def train_model(model, x_train, y_train, num_epochs=2000, learning_rate=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

# Create the model and train it
model = NeuralNetwork()
train_model(model, x_train, y_train)

# Plot the results
x_test = torch.linspace(-5, 5, 2000).unsqueeze(1)
y_test = torch.sin(x_test) - 0.01 * x_test**2
y_pred = model(x_test)

plt.figure(figsize=(8, 6))
plt.plot(x_test, y_test, label='True function', color='blue')
plt.plot(x_test, y_pred.detach().numpy(), label='Neural Network', color='red')
plt.scatter(x_train, y_train, label='Training data', color='green', s=10)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Neural Network Fit to sin(x) - 0.01x^2')
plt.legend()
plt.show()
