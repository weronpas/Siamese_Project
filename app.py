import gradio as gr
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import yaml
from PIL import Image

from src.model import SiameseNetwork

def load_config(config_path):
    """Loads configuration from YAML."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# ==========================================
# 1. GLOBAL SETUP & MODEL LOADING
# ==========================================
CONFIG = load_config('configs/baseline.yaml')
MODEL_PATH = CONFIG['paths']['save_path']
THRESHOLD = 0.4
DEVICE = torch.device("mps")

model = SiameseNetwork(embedding_dim=128).to(DEVICE)

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()  # Crucial: set to evaluation mode
    print(f"[INFO] Successfully loaded model weights from {MODEL_PATH}")
except FileNotFoundError:
    print(f"[WARNING] Weights file not found at {MODEL_PATH}. App will fail to predict until trained.")

# Define image transformations (must match training exactly)
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 2. INFERENCE FUNCTION
# ==========================================
def verify_faces(image1_path, image2_path):
    """
    Function called by Gradio when the user clicks 'Evaluate'.
    Inputs are now file paths (strings) to ensure compatibility with modern Gradio.
    """
    if image1_path is None or image2_path is None:
        return "Error: Please upload both images.", 0.0

    # Open images from paths and ensure they are in RGB format
    img1_pil = Image.open(image1_path).convert('RGB')
    img2_pil = Image.open(image2_path).convert('RGB')

    # Apply transforms and add batch dimension: [1, Channels, Height, Width]
    img1_tensor = transform(img1_pil).unsqueeze(0).to(DEVICE)
    img2_tensor = transform(img2_pil).unsqueeze(0).to(DEVICE)

    # Forward pass through the network
    with torch.no_grad():
        emb1 = model(img1_tensor)
        emb2 = model(img2_tensor)

    # Calculate Euclidean distance
    distance = F.pairwise_distance(emb1, emb2).item()
    
    # Determine verdict based on the threshold from config
    is_same_person = distance < THRESHOLD
    
    if is_same_person:
        verdict = "✅ SAME PERSON (Access Granted)"
    else:
        verdict = "❌ DIFFERENT PEOPLE (Access Denied)"
        
    return verdict, round(distance, 4)

# ==========================================
# 3. GRADIO WEB INTERFACE DESIGN
# ==========================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🕵️‍♀️ Face Verification System (Zero-Shot Learning)")
    gr.Markdown(
        "Upload two images to check if they belong to the same person. "
        "The underlying model is a custom Siamese Network using a ResNet-18 backbone."
    )
    
    with gr.Row():
        with gr.Column():
            # Zmiana: type="filepath" jest domyślny i najbardziej stabilny w nowym Gradio
            img_input1 = gr.Image(type="filepath", label="Image A (Anchor)")
        with gr.Column():
            img_input2 = gr.Image(type="filepath", label="Image B (Subject)")
            
    verify_btn = gr.Button("Evaluate Pair", variant="primary")
    
    with gr.Row():
        # Zmiana: Układ 'scale' przeniesiony do gr.Column dla poprawnego renderowania UI
        with gr.Column(scale=2):
            verdict_output = gr.Textbox(label="System Verdict")
        with gr.Column(scale=1):
            distance_output = gr.Number(label="Euclidean Distance Score")
        
    gr.Markdown(f"*Current Decision Threshold: {THRESHOLD}*")
    
    # Connect the button to the Python inference function
    verify_btn.click(
        fn=verify_faces,
        inputs=[img_input1, img_input2],
        outputs=[verdict_output, distance_output]
    )

if __name__ == "__main__":
    print("[INFO] Starting Web Application...")
    # Launch the server
    demo.launch(share=False, inbrowser=True)