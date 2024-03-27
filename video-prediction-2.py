import cv2
import torch
from facenet_pytorch import MTCNN
from torchvision import transforms
from PIL import ImageDraw
from xmodel import XMT
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw
import torch.nn.functional as F
import argparse

# Device setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize face recognition model
mtcnn = MTCNN(select_largest=False, keep_all=True, post_process=False, device=device)

# Define the XMT model
model = XMT(image_size=224, patch_size=7, num_classes=2, channels=1024, dim=1024, depth=6, heads=8, mlp_dim=2048, gru_hidden_size=1024)
model.to(device)

parser = argparse.ArgumentParser(description="Process and display video frames")
parser.add_argument("--video_path", type=str, required=True, help="Path to the input video file")
parser.add_argument("--output_path", type=str, help="Path to the output video file")
parser.add_argument("--save", action="store_true", help="Save the processed video")
parser.add_argument("--display", action="store_true", help="Display the video in real-time")
args = parser.parse_args()

# Load the pre-trained weights for the XMT model
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
    accuracy_list = []

    # Detect faces
    boxes, _ = mtcnn.detect(image)
    if boxes is not None:
        for box in boxes:
            face = image.crop(box)
            face = np.array(face)
            face = normalize_transform(face).unsqueeze(0).to(device)

            prediction = model(face)
            prediction = torch.softmax(prediction, dim=1)
            _, predicted_class = torch.max(prediction, 1)
            pred_label = predicted_class.item()

            pred_accuracy = torch.max(prediction).item() * 100

            # label = "Fake" if pred_label == 1 else "Real"
            label = "Real" if pred_label == 1 else "Fake"
            label_with_accuracy = f"{label} ({pred_accuracy:.2f}%)"
            draw_box_and_label(image, box, label_with_accuracy)
            predictions_list.append(pred_label)
            accuracy_list.append(pred_accuracy)

            print(f"Processed {image_path}: {label_with_accuracy}")  # In ra dự đoán và độ chính xác

    plt.imshow(image)
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight')

    return predictions_list, accuracy_list

def draw_box_and_label(image, box, label):
    draw = ImageDraw.Draw(image)

    # Determine color based on the label
    color = "green" if label.startswith("Real") else "red"

    # Ensure the box has the correct format (x1, y1, x2, y2)
    box = [int(coordinate) for coordinate in box]
    box_tuple = (box[0], box[1], box[2], box[3])

    # Draw rectangle and text with the determined color
    draw.rectangle(box_tuple, outline=color, width=2)
    text_position = (box[0], box[1] - 10) if box[1] - 10 > 0 else (box[0], box[1])
    draw.text(text_position, label, fill=color)

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cap.release()

def process_frame(frame):
    image = Image.fromarray(frame)
    boxes, _ = mtcnn.detect(image)
    if boxes is not None:
        for box in boxes:
            face = image.crop(box)
            face = np.array(face)
            face = normalize_transform(face).unsqueeze(0).to(device)

            prediction = model(face)
            prediction = torch.softmax(prediction, dim=1)
            _, predicted_class = torch.max(prediction, 1)
            pred_label = predicted_class.item()

            pred_accuracy = torch.max(prediction).item() * 100
            label = "Real" if pred_label == 1 else "Fake"
            label_with_accuracy = f"{label} ({pred_accuracy:.2f}%)"
            draw_box_and_label(image, box, label_with_accuracy)

    return np.array(image)

def save_video(frames, output_path, fps=20.0, resolution=(1280, 720)):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, resolution)
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()

def resize_frame(frame, target_size=(1920, 1080)):
    """
    Resize the frame to the target size while maintaining aspect ratio.
    Adds padding if necessary to fit the target size without distortion.
    """
    h, w = frame.shape[:2]
    desired_w, desired_h = target_size

    # Calculate ratios and determine scaling direction
    ratio_w = desired_w / w
    ratio_h = desired_h / h
    new_w, new_h = w, h

    if ratio_w < ratio_h:
        # Scale width and add padding to height
        new_w = desired_w
        new_h = round(h * ratio_w)
        frame = cv2.resize(frame, (new_w, new_h))
        pad_top = (desired_h - new_h) // 2
        pad_bottom = desired_h - new_h - pad_top
        frame = cv2.copyMakeBorder(frame, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT)
    else:
        # Scale height and add padding to width
        new_h = desired_h
        new_w = round(w * ratio_h)
        frame = cv2.resize(frame, (new_w, new_h))
        pad_left = (desired_w - new_w) // 2
        pad_right = desired_w - new_w - pad_left
        frame = cv2.copyMakeBorder(frame, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT)

    return frame

# To Save the Processed Video:
# python your_script_name.py --video_path "path_to_video.mp4" --output_path "path_to_output.avi" --save
# To Display the Video in Real-Time:
# python your_script_name.py --video_path "path_to_video.mp4" --display
# To Save and Display the Video:
# python your_script_name.py --video_path "path_to_video.mp4" --output_path "path_to_output.avi" --save --display
# To Only Process (Without Saving or Displaying):
# python your_script_name.py --video_path "path_to_video.mp4"
# "q" to quit the display window

def main():
    video_path = args.video_path
    output_path = args.output_path if args.output_path else "output.mp4"

    processed_frames = (process_frame(frame) for frame in extract_frames(video_path))

    if args.save:
        # Convert frames to BGR before saving
        bgr_frames = (cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in processed_frames)
        save_video(bgr_frames, output_path, resolution=(1920, 1080))
        print(f"Saved processed video to {output_path}")
    elif args.display:
        for frame in processed_frames:
            # Convert frames to BGR before displaying
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow('Frame', frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
    else:
        print("No action specified. Please use --save to save the video or --display to view it in real-time.")

if __name__ == "__main__":
    main()
