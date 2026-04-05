import cv2
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def ben_graham_preprocessing(image):
    """Enhances blood vessels and removes noise"""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    # Subtract local mean color to enhance features
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0,0), 10), -4, 128)
    return image

def generate_heatmap(model, input_tensor, original_img):
    """Generates XAI heatmap using Grad-CAM"""
    target_layers = [model.resnet.layer4[-1]]  # Target last convolutional layer
    cam = GradCAM(model=model, target_layers=target_layers)
    
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)
    grayscale_cam = grayscale_cam[0, :]
    
    # Normalize original image for visualization
    rgb_img = np.float32(original_img) / 255
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    return visualization