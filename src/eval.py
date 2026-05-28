import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

from .model import SiameseNetwork

def evaluate_pair(model_path, img1_path, img2_path, threshold, device):
    print("\n--- Starting Evaluation ---")
    print(f"Loading weights from: {model_path}")

    # 1. Initialize the model and load the trained weights
    model = SiameseNetwork(embedding_dim=128).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"[ERROR] Weights file not found at {model_path}. Train the model first.")
        return
        
    # IMPORTANT: Set model to evaluation mode (disables Dropout, fixes BatchNorm)
    model.eval()

    # 2. Define the exact same transformations used during training
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. Load and preprocess images
    try:
        img1_pil = Image.open(img1_path).convert('RGB')
        img2_pil = Image.open(img2_path).convert('RGB')
    except FileNotFoundError as e:
        print(f"[ERROR] Could not find image file: {e}")
        return

    # Add batch dimension: [Channels, Height, Width] -> [1, Channels, Height, Width]
    img1_tensor = transform(img1_pil).unsqueeze(0).to(device)
    img2_tensor = transform(img2_pil).unsqueeze(0).to(device)

    # 4. Forward pass
    with torch.no_grad():
        emb1 = model(img1_tensor)
        emb2 = model(img2_tensor)

    # 5. Calculate Euclidean distance
    distance = F.pairwise_distance(emb1, emb2).item()
    is_same_person = distance < threshold

    # 6. Console Output
    print("\n[RESULT]")
    print(f"Distance between embeddings: {distance:.4f}")
    print(f"Decision Threshold:          {threshold:.4f}")
    
    if is_same_person:
        print("Final Verdict: SAME PERSON (Access Granted)")
    else:
        print("Final Verdict: DIFFERENT PEOPLE (Access Denied)")

    # 7. Visualization
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(img1_pil)
    axes[0].set_title("Image A")
    axes[0].axis('off')

    axes[1].imshow(img2_pil)
    axes[1].set_title("Image B")
    axes[1].axis('off')

    result_text = f"Distance: {distance:.4f} | Same Person: {is_same_person}"
    color = "green" if is_same_person else "red"
    plt.suptitle(result_text, color=color, fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.show()