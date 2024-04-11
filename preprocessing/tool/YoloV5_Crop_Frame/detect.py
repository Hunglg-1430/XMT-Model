import os
import cv2
import torch
import random
from pathlib import Path
from tqdm import tqdm
from models.experimental import attempt_load
from utils.torch_utils import select_device
from utils.general import non_max_suppression
from torchvision import transforms

def scale_coords(img_shape, coords, actual_shape):
    gain = min(img_shape[0] / actual_shape[0], img_shape[1] / actual_shape[1])
    pad_x = (img_shape[1] - actual_shape[1] * gain) / 2  # horizontal padding
    pad_y = (img_shape[0] - actual_shape[0] * gain) / 2  # vertical padding
    coords[:, [0, 2]] -= pad_x
    coords[:, [1, 3]] -= pad_y
    coords[:, :4] /= gain
    coords[:, :4] = coords[:, :4].clamp(min=0)
    return coords

def get_frame_count(video_path):
    # Open video file
    cap = cv2.VideoCapture(video_path)
    # Get total number of frames
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Release video capture
    cap.release()
    return frame_count

def select_random_frames(total_frames, num_frames=100):
    # Select 100 random frame indices
    return random.sample(range(total_frames), num_frames)

# Load YOLOv5 model
device = select_device('')
model = attempt_load('face_detection_yolov5s.pt')
stride = int(model.stride.max())

# Input and output directories
input_folder = r'D:\Youtube\frame_1'
output_folder = r'D:\Youtube\frame_2'
Path(output_folder).mkdir(parents=True, exist_ok=True)

# Transformation to resize images to 224x224
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Iterate through all video files in the input folder
for video_file in os.listdir(input_folder):
    if video_file.endswith(".mp4"):
        video_path = os.path.join(input_folder, video_file)
        total_frames = get_frame_count(video_path)
        selected_frames = select_random_frames(total_frames)

        # Open video file
        cap = cv2.VideoCapture(video_path)

        for frame_num in tqdm(selected_frames):
            # Set the frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                break

            # Resize the input image to match the expected dimensions of the YOLOv5 model
            img = torch.from_numpy(frame).to(device)
            img = img.float() / 255.0

            # Convert to RGB (3 channels)
            img = img.permute(2, 0, 1).unsqueeze(0)

            # Pad the image to have a size divisible by the model's stride
            img = torch.nn.functional.pad(img, (0, stride - img.shape[-1] % stride, 0, stride - img.shape[-2] % stride))

            # Perform face detection on the frame
            pred = model(img)[0]
            pred = non_max_suppression(pred, conf_thres=0.8, iou_thres=0.8)[0]

            if pred is not None and len(pred) > 0:
                pred = scale_coords(img.shape[2:], pred[:, :4], frame.shape).round()
                for i, det in enumerate(pred):
                    x1, y1, x2, y2 = map(int, det)
                    face = frame[y1:y2, x1:x2]

                    # Resize the cropped face to 224x224
                    resized_face = transform(face)

                    output_filename = f'{output_folder}/{video_file.split(".")[0]}_frame_{frame_num}_crop_{i}.jpg'
                    cv2.imwrite(output_filename, resized_face.numpy().transpose(1, 2, 0) * 255)

        # Release video capture
        cap.release()
