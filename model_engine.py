import torch
import torch.nn as nn
from torchvision import models

class SilentSightModel(nn.Module):
    def __init__(self, num_classes=5):
        super(SilentSightModel, self).__init__()
        # Load pre-trained ResNet-50
        self.resnet = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        
        # Replace the final fully connected layer
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)

def load_model(path):
    model = SilentSightModel()
    # If you have trained weights, uncomment the line below
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model