import torch
import torch.nn as nn
import torch.optim as optim

# Study hours
X = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])

# Exam scores
y = torch.tensor([[50.0], [60.0], [70.0], [80.0], [90.0]])

# Neural network with 1 input and 1 output
model = nn.Linear(1, 1)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(1000):
    predictions = model(X)
    loss = criterion(predictions, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Test the model
new_hours = torch.tensor([[6.0]])
predicted_score = model(new_hours)

print(f"Predicted score for 6 hours of study: {predicted_score.item():.2f}")
