import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import time
import copy 

from .dataset import LFWTripletDataset
from .model import SiameseNetwork

def run_training(data_dir, batch_size, epochs, lr, margin, patience, device):
    print("--- Initializing Data Pipeline ---")
    
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = LFWTripletDataset(root_dir=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = SiameseNetwork(embedding_dim=128).to(device)
    criterion = torch.nn.TripletMarginLoss(margin=margin, p=2)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # --- EARLY STOPPING VARIABLES ---
    best_loss = float('inf')
    epochs_no_improve = 0
    best_model_weights = copy.deepcopy(model.state_dict())

    print("\n--- Starting Training (with Early Stopping) ---")
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
        
        print(f"=== Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f}s | Average Loss: {avg_loss:.4f} ===")
        
        # --- EARLY STOPPING LOGIC ---
        if avg_loss < best_loss:
            print(f"New best model! Loss decreased from {best_loss:.4f} to {avg_loss:.4f}. Saving weights...")
            best_loss = avg_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s). (Patience: {patience})")
            
            if epochs_no_improve >= patience:
                print(f"\n[!!!] EARLY STOPPING TRIGGERED [!!!]")
                print(f"Training stopped at epoch {epoch+1}. Loss stopped decreasing.")
                break 

    print("\n--- Finishing ---")
    model.load_state_dict(best_model_weights)
    save_path = "best_siamese_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Best weights saved to: {save_path} (Loss: {best_loss:.4f})")