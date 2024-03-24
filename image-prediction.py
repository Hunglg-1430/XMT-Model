import cv2
import torch
from facenet_pytorch import MTCNN
from torchvision import transforms
from xmodel import XMT
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw
import torch.nn.functional as F

# Device setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize face recognition model
mtcnn = MTCNN(select_largest=False, keep_all=True, post_process=False, device=device)

# Define the CViT model
model = XMT(image_size=224, patch_size=7, num_classes=2, channels=1024, dim=1024, depth=6, heads=8, mlp_dim=2048, gru_hidden_size=1024)
model.to(device)

# Load the pre-trained weights for the CViT model
checkpoint = torch.load('weight/xmodel_deepfake_sample_1.pth', map_location=torch.device('cpu'))
filtered_state_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model.state_dict()}
model.load_state_dict(filtered_state_dict)

# Put the model in evaluation mode
model.eval()

# Image preprocessing transformation
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

def process_and_save_image(image_path, output_path):
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

    plt.imshow(image)
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight')

    return predictions_list

def draw_box_and_label(image, box, label):
    draw = ImageDraw.Draw(image)

    # Ensure the box has the correct format (x1, y1, x2, y2)
    box = [int(coordinate) for coordinate in box]
    box_tuple = (box[0], box[1], box[2], box[3])

    draw.rectangle(box_tuple, outline="red", width=2)
    text_position = (box[0], box[1] - 10) if box[1] - 10 > 0 else (box[0], box[1])
    draw.text(text_position, label, fill="red")

folder_path = 'data/sample_train_data/val/real'

# List all files in the folder
image_files = os.listdir(folder_path)

# Create a folder to store the output images
output_folder = 'XMT-Model/Output'
os.makedirs(output_folder, exist_ok=True)

correct_predictions = 0
total_predictions = 0
for file_name in image_files:
    if file_name.endswith('.jpg') or file_name.endswith('.png'):
        image_path = os.path.join(folder_path, file_name)
        output_image_name = file_name.split('.')[0] + '_processed.png'
        output_path = os.path.join(output_folder, output_image_name)

        predictions = process_and_save_image(image_path, output_path)
        expected_label = 1 if 'real' in folder_path.lower() else 0  # Assuming 'real' in folder path implies real images

        # Count correct predictions
        correct_predictions += sum(pred == expected_label for pred in predictions)
        total_predictions += len(predictions)
        print(f"Processed and saved image: {output_path}")

# Calculate accuracy
if total_predictions > 0:
    accuracy = (correct_predictions / total_predictions) * 100
    print(f"Prediction accuracy: {accuracy}%")
else:
    print("No faces detected in any image.")

# Inform the user where the output images are saved
print(f"Output images are saved in the folder: {output_folder}")
