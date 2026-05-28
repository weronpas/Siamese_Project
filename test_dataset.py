import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from dataset import LFWTripletDataset

def show_triplet(anchor, positive, negative):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    anchor_np = anchor.permute(1, 2, 0).numpy()
    positive_np = positive.permute(1, 2, 0).numpy()
    negative_np = negative.permute(1, 2, 0).numpy()
    
    axes[0].imshow(anchor_np)
    axes[0].set_title("Anchor")
    axes[0].axis("off")
    
    axes[1].imshow(positive_np)
    axes[1].set_title("Positive")
    axes[1].axis("off")
    
    axes[2].imshow(negative_np)
    axes[2].set_title("Negative")
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Initializing transformations...")
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor()
    ])
    
    print("Loading dataset...")
    dataset = LFWTripletDataset(root_dir='data/lfw', transform=transform)
    
    print(f"Dataset successfully loaded! Total people with at least 2 images: {len(dataset)}")
    
    if len(dataset) > 0:
        anchor, positive, negative = dataset[0]
        
        print("Displaying the first triplet...")
        print(f"Tensor shape: {anchor.shape}") # Should be [3, 112, 112]
        
        show_triplet(anchor, positive, negative)
    else:
        print("Error: No valid images found. Check your 'data/lfw' path.")