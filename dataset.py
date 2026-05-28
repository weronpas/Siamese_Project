import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset

class LFWTripletDataset(Dataset):
    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir
        self.transform = transform
        
        self.person_to_images = {}
        
        for person_name in os.listdir(root_dir):
            person_path = os.path.join(root_dir, person_name)

            if os.path.isdir(person_path):
                images = [img for img in os.listdir(person_path) if img.endswith('.jpg')]
                
                if len(images) > 1:
                    self.person_to_images[person_name] = images
        
        self.persons = list(self.person_to_images.keys())
        
    def __len__(self):
        return len(self.persons)

    def __getitem__(self, idx):
        anchor_person = self.persons[idx]
        
        anchor_img_name, positive_img_name = random.sample(self.person_to_images[anchor_person], 2)
        
        negative_person = random.choice(self.persons)
        while negative_person == anchor_person:
            negative_person = random.choice(self.persons)
            
        negative_img_name = random.choice(self.person_to_images[negative_person])
        
        anchor_path = os.path.join(self.root_dir, anchor_person, anchor_img_name)
        positive_path = os.path.join(self.root_dir, anchor_person, positive_img_name)
        negative_path = os.path.join(self.root_dir, negative_person, negative_img_name)
        
        anchor_img = Image.open(anchor_path).convert('RGB')
        positive_img = Image.open(positive_path).convert('RGB')
        negative_img = Image.open(negative_path).convert('RGB')
        
        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)
            
        return anchor_img, positive_img, negative_img