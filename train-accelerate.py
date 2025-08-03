import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pytorch_accelerated import Trainer
import time
import numpy as np

# Simple Neural Network
class SimpleNN(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def create_synthetic_data(num_samples=10000, input_size=784, num_classes=10):
    """Create synthetic dataset for fast training"""
    X = torch.randn(num_samples, input_size)
    y = torch.randint(0, num_classes, (num_samples,))
    return TensorDataset(X, y)

def train_model():
    # Hyperparameters
    learning_rate = 0.001
    num_epochs = 1
    input_size = 784
    num_classes = 10
    
    # Create synthetic datasets
    train_dataset = create_synthetic_data(num_samples=100000*10)
    val_dataset = create_synthetic_data(num_samples=5000*10)
    test_dataset = create_synthetic_data(num_samples=5000*10)
    
    # Initialize model, optimizer, and loss function
    model = SimpleNN(input_size=input_size, num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()
    
    # Create trainer
    trainer = Trainer(
        model,
        loss_func=loss_func,
        optimizer=optimizer,
    )
    
    # Train the model
    trainer.train(
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        num_epochs=num_epochs,
        per_device_batch_size=32,
    )
    
    # Evaluate on test dataset
    trainer.evaluate(
        dataset=test_dataset,
        per_device_batch_size=32,
    )

if __name__ == "__main__":
    train_model()
