import torch
from facenet_pytorch import MTCNN
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw
import torch.nn.functional as F
from load_model import load_model_xmt
from PIL import ImageFont, ImageDraw


mtcnn, model, device = load_model_xmt()

# Image preprocessing transformation
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

def process_and_save_image(image_path):
    image = Image.open(image_path)
    image = image.convert('RGB')
    predictions_list = []

    # Detect faces
    boxes, _ = mtcnn.detect(image)
    if boxes is not None:
        for box in boxes:
            face = image.crop(box)
            face = np.array(face)
            face = normalize_transform(face).unsqueeze(0).to(device)

            prediction = model(face)
            _, predicted_class = torch.max(prediction, 1)  # Xác định lớp dựa trên logits
            pred_label = predicted_class.item()

            # Giả định lớp 1 là "Real", lớp 0 là "Fake"
            label = "Real" if pred_label == 1 else "Fake"
            draw_box_and_label(image, box, label)
            predictions_list.append(pred_label)
    return image

def draw_box_and_label(image, box, label):
    draw = ImageDraw.Draw(image)

    # Ensure the box has the correct format (x1, y1, x2, y2)
    box = [int(coordinate) for coordinate in box]
    box_tuple = (box[0], box[1], box[2], box[3])
    font = ImageFont.load_default(30)

    draw.rectangle(box_tuple, outline="red", width=2)
    text_position = (box[0], box[1] - 10) if box[1] - 10 > 0 else (box[0], box[1])
    draw.text(text_position, label, fill="red", font=font)

# image_path = '/Users/lap01743/Downloads/WorkSpace/capstone_wed/test_image/1_QOMN.jpg'
# # output_path = 'XMT-Model/Output'


# predictions = process_and_save_image(image_path)
# print("predictions image is ", predictions)
