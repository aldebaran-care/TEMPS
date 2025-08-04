import argparse
import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
from torch.optim import Adam
import numpy as np

class SimpleNN(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(SimpleNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)

def create_dummy_dataset(size=1000, input_size=784, num_classes=10):
    """Create dummy dataset for demonstration"""
    X = torch.randn(size, input_size)
    y = torch.randint(0, num_classes, (size,))
    return TensorDataset(X, y)

def train_single_gpu(args):
    """Training function for single GPU"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on single GPU: {device}")
    
    # Create model, dataset, and dataloader
    model = SimpleNN().to(device)
    dataset = create_dummy_dataset(args.dataset_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Optimizer and loss function
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch} completed. Average Loss: {avg_loss:.4f}')

def setup_ddp(rank, world_size):
    """Setup distributed data parallel"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    """Cleanup distributed data parallel"""
    dist.destroy_process_group()

def train_ddp_worker(rank, world_size, args):
    """Worker function for DDP training"""
    setup_ddp(rank, world_size)
    
    # Create model and move to GPU
    model = SimpleNN().cuda(rank)
    model = DDP(model, device_ids=[rank])
    
    # Create dataset with distributed sampler
    dataset = create_dummy_dataset(args.dataset_size)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)
    
    # Optimizer and loss function
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    model.train()
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.cuda(rank), target.cuda(rank)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0 and rank == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        if rank == 0:
            avg_loss = total_loss / len(dataloader)
            print(f'Epoch {epoch} completed. Average Loss: {avg_loss:.4f}')
    
    cleanup_ddp()

def train_ddp(args):
    """Setup and launch DDP training"""
    world_size = torch.cuda.device_count()
    print(f"Training with DDP on {world_size} GPUs")
    
    if world_size < 2:
        print("Warning: DDP requires at least 2 GPUs. Falling back to single GPU training.")
        train_single_gpu(args)
        return
    
    mp.spawn(train_ddp_worker, args=(world_size, args), nprocs=world_size, join=True)

def main():
    parser = argparse.ArgumentParser(description='PyTorch Neural Network Training')
    parser.add_argument('--mode', type=str, choices=['single', 'ddp'], default='single',
                        help='Training mode: single GPU or DDP (default: single)')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs (default: 5)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--dataset-size', type=int, default=1000,
                        help='Size of dummy dataset (default: 1000)')
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA not available. Training on CPU.")
        args.mode = 'single'
    
    print(f"Training configuration:")
    print(f"  Mode: {args.mode}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Dataset size: {args.dataset_size}")
    
    if args.mode == 'single':
        train_single_gpu(args)
    elif args.mode == 'ddp':
        train_ddp(args)

if __name__ == '__main__':
    main()
