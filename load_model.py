import cv2
import torch
from facenet_pytorch import MTCNN
from torchvision import transforms
from model.xmodel import XMT
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw
import torch.nn.functional as F


def load_model_xmt(): 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mtcnn = MTCNN(select_largest=False, keep_all=True, post_process=False, device=device)
    # Define the XMT model
    model = XMT(image_size=224, patch_size=7, num_classes=2, channels=1024, dim=1024, depth=6, heads=8, mlp_dim=2048, gru_hidden_size=1024)
    model.to(device)

    # Load the pre-trained weights for the XMT model
    checkpoint = torch.load('/Users/lap01743/Downloads/WorkSpace/capstone_wed/XMT_Model/xmodel_deepfake.pth', map_location=torch.device('cpu'))
    filtered_state_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model.state_dict()}
    model.load_state_dict(filtered_state_dict)

    # Put the model in evaluation mode
    model.eval()
    print("load model successfully")
    return mtcnn, model, device