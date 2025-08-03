import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pytorch_accelerated import Trainer
import time
import numpy as np
import argparse

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

def train_vanilla_pytorch(train_dataset, val_dataset, test_dataset, model, optimizer, loss_func, num_epochs=1, batch_size=32):
    """Vanilla PyTorch training loop"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Training on device: {device}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        start_time = time.time()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            output = model(data)
            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 20 == 0:
                print(f'Epoch: {epoch+1}/{num_epochs}, '
                      f'Batch: {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}')
        
        train_acc = 100. * correct / total
        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - start_time
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += loss_func(output, target).item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
        
        val_acc = 100. * val_correct / val_total
        val_loss /= len(val_loader)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'  Time: {epoch_time:.2f}s')
        print('-' * 50)
    
    # Test evaluation
    model.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_func(output, target).item()
            _, predicted = output.max(1)
            test_total += target.size(0)
            test_correct += predicted.eq(target).sum().item()
    
    test_acc = 100. * test_correct / test_total
    test_loss /= len(test_loader)
    
    print(f'Test Results:')
    print(f'  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')

def train_model(use_accelerated=True):
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
    
    if use_accelerated:
        print("Using pytorch-accelerated training...")
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
    else:
        print("Using vanilla PyTorch training...")
        train_vanilla_pytorch(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            model=model,
            optimizer=optimizer,
            loss_func=loss_func,
            num_epochs=num_epochs,
            batch_size=32
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train SimpleNN with different backends')
    parser.add_argument('--backend', choices=['accelerated', 'vanilla'], default='accelerated',
                        help='Training backend to use (default: accelerated)')
    args = parser.parse_args()
    
    use_accelerated = args.backend == 'accelerated'
    train_model(use_accelerated=use_accelerated)
