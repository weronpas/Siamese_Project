import torch
from model import SiameseNetwork

if __name__ == "__main__":
    print("Initializing the Siamese Network...")
    
    model = SiameseNetwork(embedding_dim=128)
    device = torch.device("mps")
    model = model.to(device)

    dummy_input = torch.randn(16, 3, 112, 112).to(device)
    
    print(f"Feeding dummy input of shape: {dummy_input.shape}")

    with torch.no_grad():
        output = model(dummy_input)
        
    print(f"Output shape: {output.shape}")
    
    if output.shape == (16, 128):
        print("Test Passed: The model correctly outputs 128-dimensional embeddings for the batch!")
    else:
        print("Test Failed: Unexpected output shape.")