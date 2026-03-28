import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset
import cv2
import os

# Loading Data Section
class BaseballVideoLoader(Dataset):
    def __init__(self, video_dir, transform=None):
        self.video_dir = video_dir
        # Get all .mov files 
        self.video_files = [f for f in os.listdir(video_dir) if f.endswith('.mov')]
        self.transform = transform

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = os.path.join(self.video_dir, self.video_files[idx])
        cap = cv2.VideoCapture(video_path)
        
        #Attempting to grab the middle frame
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return torch.zeros(3, 224, 224), 0
        
        #standardize for the model
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        return frame_tensor, 0

# Neural Network section
def get_baseball_model():
    # Using ResNet-18 as our foundational architecture (hope this works)
    model = models.resnet18(weights=None)
    # Adjust for 2 classes ->>>(Baseball vs No Baseball)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model
