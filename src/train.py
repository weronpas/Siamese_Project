import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import json
import time
import copy 

from .dataset import LFWTripletDataset
from .model import SiameseNetwork

def run_training(data_dir, batch_size, epochs, lr, margin, patience, device):
    print("--- Initializing Augmented Data Pipeline ---")
    
    # Advanced Data Augmentation for robust feature learning
    train_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(p=0.5), # Robustness to face orientation
        transforms.ColorJitter(
            brightness=0.3,   # Robustness to shadows and bright spots
            contrast=0.3,     # Robustness to underexposed/overexposed lighting
            saturation=0.2,   # Robustness to camera sensor variations
            hue=0.1
        ),
        transforms.ToTensor(),
        transforms.RandomErasing(
            p=0.2, 
            scale=(0.02, 0.15), 
            value='random'
        ), # Simulates occlusions like sunglasses, hair, or masks
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # We pass the heavily augmented transform into our training dataset
    dataset = LFWTripletDataset(root_dir=data_dir, transform=train_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = SiameseNetwork(embedding_dim=128).to(device)
    criterion = torch.nn.TripletMarginLoss(margin=margin, p=2)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # --- EARLY STOPPING VARIABLES ---
    best_loss = float('inf')
    epochs_no_improve = 0
    best_model_weights = copy.deepcopy(model.state_dict())

    training_history = {
        'train_loss': [],
        'best_epoch': 0
    }

    print("\n--- Starting Training (with Augmentation and Early Stopping) ---")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        
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
        
        print(f"=== Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f}s | Average Loss: {avg_loss:.4f} ===")
        
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

    save_path = "best_siamese_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Best weights saved to: {save_path} (Loss: {best_loss:.4f})")

    history_path = "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=4)
    print(f"Metrics history saved to: {history_path}")