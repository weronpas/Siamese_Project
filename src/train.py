import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import time
import copy 
import json
import os

from .dataset import LFWTripletDataset
from .model import SiameseNetwork

def run_training(data_dir, batch_size, epochs, lr, margin, patience, device):
    print("--- Initializing Augmented Data Pipeline ---")
    
    # Advanced Data Augmentation for robust feature learning
    train_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.ToTensor(), 
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15), value=0), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = LFWTripletDataset(root_dir=data_dir, transform=train_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = SiameseNetwork(embedding_dim=128).to(device)
    criterion = torch.nn.TripletMarginLoss(margin=margin, p=2)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # NEW: Initialize the ReduceLROnPlateau Scheduler
    # It will cut the LR in half (factor=0.5) if the loss doesn't improve for 2 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=2
    )

    # --- EARLY STOPPING & HISTORY VARIABLES ---
    best_loss = float('inf')
    epochs_no_improve = 0
    best_model_weights = copy.deepcopy(model.state_dict())
    
    training_history = {
        'train_loss': [],
        'learning_rates': [],  # NEW: Track LR changes over epochs
        'best_epoch': 0
    }

    print("\n--- Starting Training (with Augmentation, Scheduler and Early Stopping) ---")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        
        # Capture the current active learning rate before the epoch starts
        current_lr = optimizer.param_groups[0]['lr']
        training_history['learning_rates'].append(current_lr)
        
        for batch_idx, (anchor, positive, negative) in enumerate(dataloader):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            
            optimizer.zero_grad()
            
            emb_anchor = model(anchor)
            emb_positive = model(positive)
            emb_negative = model(negative)
            
            loss = criterion(emb_anchor, emb_positive, emb_negative)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
                
        epoch_time = time.time() - start_time
        avg_loss = running_loss / len(dataloader)
        training_history['train_loss'].append(avg_loss)
        
        # NEW: Step the scheduler based on the average epoch loss
        scheduler.step(avg_loss)
        
        print(f"=== Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f}s | Loss: {avg_loss:.4f} | LR: {current_lr:.6f} ===")
        
        # --- EARLY STOPPING LOGIC ---
        if avg_loss < best_loss:
            print(f"[*] New best model! Loss decreased from {best_loss:.4f} to {avg_loss:.4f}. Saving weights...")
            best_loss = avg_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            training_history['best_epoch'] = epoch + 1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"[!] No improvement for {epochs_no_improve} epoch(s). (Patience: {patience})")
            
            if epochs_no_improve >= patience:
                print(f"\n[!!!] EARLY STOPPING TRIGGERED [!!!]")
                print(f"Training stopped at epoch {epoch+1}. Loss stopped decreasing.")
                break 

    print("\n--- Finishing ---")
    model.load_state_dict(best_model_weights)
    
    # Save Model Weights
    save_path = "best_siamese_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Best weights saved to: {save_path} (Loss: {best_loss:.4f})")
    
    # Save Training History for Jupyter Notebook
    history_path = "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=4)
    print(f"Metrics history saved to: {history_path}")