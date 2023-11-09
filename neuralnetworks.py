import torch
import torch.nn as nn
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 3)
        self.fc2 = nn.Linear(3, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Let's create some dummy data for demonstration
X_train = torch.Tensor([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y_train = torch.Tensor([[0], [0], [1], [1], [1]])  # Update the target shape

# Create an instance of Net
model = Net()

# Define a loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the network
for epoch in range(100000):
    # Forward pass
    outputs = model(X_train)
    # Compute loss
    loss = criterion(outputs, y_train)
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Make a prediction
with torch.no_grad():
    print("Prediction for [5, 6]:", model(torch.Tensor([[5, 6]])))
