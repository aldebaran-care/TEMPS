import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from accelerate import Accelerator
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
    # Initialize accelerator
    accelerator = Accelerator()
    
    # Hyperparameters
    batch_size = 128
    learning_rate = 0.001
    num_epochs = 5
    input_size = 784
    num_classes = 10
    
    # Create synthetic dataset
    train_dataset = create_synthetic_data(num_samples=8000)
    val_dataset = create_synthetic_data(num_samples=2000)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model, optimizer, and loss function
    model = SimpleNN(input_size=input_size, num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Prepare everything with accelerator
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    
    # Training loop
    accelerator.print(f"Starting training on {accelerator.device}")
    accelerator.print(f"Number of processes: {accelerator.num_processes}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        start_time = time.time()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass with accelerator
            accelerator.backward(loss)
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 20 == 0:
                accelerator.print(f'Epoch: {epoch+1}/{num_epochs}, '
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
                output = model(data)
                val_loss += criterion(output, target).item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
        
        val_acc = 100. * val_correct / val_total
        val_loss /= len(val_loader)
        
        accelerator.print(f'Epoch {epoch+1}/{num_epochs}:')
        accelerator.print(f'  Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%')
        accelerator.print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        accelerator.print(f'  Time: {epoch_time:.2f}s')
        accelerator.print('-' * 50)
    
    # Save model (only on main process)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save(unwrapped_model.state_dict(), 'simple_nn_model.pth')
        accelerator.print("Model saved successfully!")

if __name__ == "__main__":
    train_model()
