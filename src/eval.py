import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import random
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from .model import SiameseNetwork

def evaluate_accuracy(model_path, data_dir, device, num_pairs=200, threshold=0.8):
    """
    Evaluates the model using advanced binary classification metrics:
    Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
    """
    print("\n--- Running Global Model Evaluation (Advanced Metrics) ---")
    model = SiameseNetwork(embedding_dim=128).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    except FileNotFoundError:
        print(f"[ERROR] Weights file not found at {model_path}.")
        return

    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Gather all available people folders
    all_people = [p for p in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, p))]
    people_with_multiple_images = [p for p in all_people if len(os.listdir(os.path.join(data_dir, p))) >= 2]

    y_true = []
    y_pred = []

    print(f"Evaluating {num_pairs} random pairs (50% matching, 50% non-matching)...")

    with torch.no_grad():
        for i in range(num_pairs):
            is_positive_pair = (i % 2 == 0)

            if is_positive_pair and len(people_with_multiple_images) > 0:
                # Same person (Label: 1)
                person = random.choice(people_with_multiple_images)
                person_dir = os.path.join(data_dir, person)
                img1_name, img2_name = random.sample(os.listdir(person_dir), 2)
                img1_path = os.path.join(person_dir, img1_name)
                img2_path = os.path.join(person_dir, img2_name)
                expected_label = 1
            else:
                # Different people (Label: 0)
                person1, person2 = random.sample(all_people, 2)
                img1_name = random.choice(os.listdir(os.path.join(data_dir, person1)))
                img2_name = random.choice(os.listdir(os.path.join(data_dir, person2)))
                img1_path = os.path.join(data_dir, person1, img1_name)
                img2_path = os.path.join(data_dir, person2, img2_name)
                expected_label = 0

            try:
                img1 = transform(Image.open(img1_path).convert('RGB')).unsqueeze(0).to(device)
                img2 = transform(Image.open(img2_path).convert('RGB')).unsqueeze(0).to(device)
                
                emb1 = model(img1)
                emb2 = model(img2)
                
                distance = F.pairwise_distance(emb1, emb2).item()
                predicted_label = 1 if distance < threshold else 0

                y_true.append(expected_label)
                y_pred.append(predicted_label)
            except Exception:
                continue

    # Convert lists to numpy arrays for sklearn metrics
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate advanced metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    report = classification_report(y_true, y_pred, target_names=['Different People', 'Same Person'])

    print("\n=======================================")
    print("         DETAILED EVALUATION REPORT     ")
    print("=======================================")
    print(report)
    print("---------------------------------------")
    print("CONFUSION MATRIX:")
    print(f" True Negatives (Correct Denied):  {tn}")
    print(f" False Positives (Security Leaks): {fp} <-- Lower is better!")
    print(f" False Negatives (False Alarms):  {fn}")
    print(f" True Positives (Correct Allowed): {tp}")
    print("=======================================\n")
    
    return report