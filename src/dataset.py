import os
import random
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from facenet_pytorch import MTCNN  # NEW: Advanced Face Detection & Alignment

class LFWTripletDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Custom Dataset for LFW that generates triplets (Anchor, Positive, Negative)
        with on-the-fly MTCNN Face Detection and Alignment.
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # NEW: Initialize MTCNN detector
        # image_size=112 matches our network input size. 
        # margin=20 adds a small safety padding around the face bounding box.
        # select_largest=True ensures we crop the main face if backgrounds are busy.
        self.mtcnn = MTCNN(
            image_size=112, 
            margin=20, 
            select_largest=True, 
            post_process=False # We keep raw pixels so torchvision transforms can execute later
        )

        # Gather all unique people folders
        self.all_persons = [p for p in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, p))]
        
        # Anchors must have at least 2 images to form an (Anchor, Positive) pair
        self.anchor_persons = [p for p in self.all_persons if len(os.listdir(os.path.join(root_dir, p))) >= 2]

    def _load_and_align_face(self, img_path):
        """
        Helper method to load an image and pass it through the MTCNN face detector.
        Falls back to standard image cropping if no face is detected.
        """
        try:
            img = Image.open(img_path).convert('RGB')
            
            # Extract face tensor using MTCNN
            # This automatically detects landmarks, crops, and resizes to 112x112
            face_tensor = self.mtcnn(img)
            
            if face_tensor is not None:
                # MTCNN outputs a torch tensor [C, H, W]. 
                # We convert it back to a PIL Image so that our Data Augmentation pipeline 
                # in train.py (ColorJitter, Flips, etc.) can process it normally.
                face_img = transforms.ToPILImage()(face_tensor.byte())
                return face_img
        except Exception:
            pass
            
        # Robust Fallback: If MTCNN fails, load raw image and let transforms handle it
        return Image.open(img_path).convert('RGB')

    def __len__(self):
        # We define the epoch length based on the total number of valid anchor individuals
        return len(self.anchor_persons)

    def __getitem__(self, idx):
        # 1. Pick an Anchor person
        anchor_person = self.anchor_persons[idx]
        anchor_dir = os.path.join(self.root_dir, anchor_person)
        
        # 2. Pick two distinct images of the Anchor person (Anchor & Positive)
        anchor_img_name, positive_img_name = random.sample(os.listdir(anchor_dir), 2)
        
        # 3. Pick a Negative person (different identity)
        negative_person = random.choice(self.all_persons)
        while negative_person == anchor_person:
            negative_person = random.choice(self.all_persons)
            
        negative_dir = os.path.join(self.root_dir, negative_person)
        negative_img_name = random.choice(os.listdir(negative_dir))

        # 4. Load and Align all three faces using MTCNN
        anchor_img = self._load_and_align_face(os.path.join(anchor_dir, anchor_img_name))
        positive_img = self._load_and_align_face(os.path.join(anchor_dir, positive_img_name))
        negative_img = self._load_and_align_face(os.path.join(negative_dir, negative_img_name))

        # 5. Apply training augmentations (ColorJitter, Erasing, Normalization)
        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return anchor_img, positive_img, negative_img