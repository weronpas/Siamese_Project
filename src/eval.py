import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import random
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
import matplotlib.pyplot as plt

from .model import SiameseNetwork


def evaluate_accuracy(model_path, data_dir, device, num_pairs=200, threshold=None):
    print("\n--- Running Global Model Evaluation (Dynamic Threshold) ---")
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

    all_people = [p for p in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, p))]
    people_with_multiple_images = [p for p in all_people if len(os.listdir(os.path.join(data_dir, p))) >= 2]

    y_true = []
    distances = []

    print(f"Evaluating {num_pairs} random pairs (50% matching, 50% non-matching)...")

    with torch.no_grad():
        for i in range(num_pairs):
            is_positive_pair = (i % 2 == 0)

            if is_positive_pair and len(people_with_multiple_images) > 0:
                person = random.choice(people_with_multiple_images)
                person_dir = os.path.join(data_dir, person)
                img1_name, img2_name = random.sample(os.listdir(person_dir), 2)
                img1_path = os.path.join(person_dir, img1_name)
                img2_path = os.path.join(person_dir, img2_name)
                expected_label = 1
            else:
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
                
                # Zapisujemy dokładny dystans, zamiast od razu wydawać wyrok
                dist = F.pairwise_distance(emb1, emb2).item()
                
                y_true.append(expected_label)
                distances.append(dist)
            except Exception:
                continue

    y_true = np.array(y_true)
    distances = np.array(distances)

    # --- ROC CURVE: OPTIMAL THRESHOLD CALCULATION ---
    # We use negative distances because ROC expects higher scores for the positive class
    fpr, tpr, roc_thresholds = roc_curve(y_true, -distances)
    
    # Youden's J statistic to find the best balance between Sensitivity and Specificity
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = -roc_thresholds[optimal_idx]

    print(f"\n[AI ANALYSIS] The hardcoded threshold was: {threshold if threshold else 0.8}")
    print(f"[AI ANALYSIS] The dynamically calculated OPTIMAL threshold is: {optimal_threshold:.4f}")

    # ==========================================
    # NEW: MATPLOTLIB ROC CURVE VISUALIZATION
    # ==========================================
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC Curve')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.scatter([fpr[optimal_idx]], [tpr[optimal_idx]], color='red', s=100, zorder=5, 
                label=f'Optimal Threshold ({optimal_threshold:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (Security Leaks)')
    plt.ylabel('True Positive Rate (Correct Access)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    # ==========================================
    
    # Use the optimal threshold to calculate final metrics
    y_pred = (distances < optimal_threshold).astype(int)

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