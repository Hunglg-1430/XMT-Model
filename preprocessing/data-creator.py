import os
import shutil
from PIL import Image
import random

def prepare_photo_folders(input_folders, output_path, subfolder_name):
    # Create train, test, val folders if they don't exist
    for folder in ['train', 'test', 'val']:
        os.makedirs(os.path.join(output_path, folder, subfolder_name), exist_ok=True)

    # Retrieve all photos
    all_photos = []
    for folder in input_folders:
        for file in os.listdir(folder):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                all_photos.append(os.path.join(folder, file))

    # Shuffle and divide photos
    random.shuffle(all_photos)
    total_photos = len(all_photos)
    train_split = int(total_photos * 0.7)
    test_split = int(total_photos * 0.15)

    train_photos = all_photos[:train_split]
    test_photos = all_photos[train_split:train_split + test_split]
    val_photos = all_photos[train_split + test_split:]

    # Function to process and move photos
    def process_and_move(photos, folder):
        count = 0
        for photo in photos:
            try:
                with Image.open(photo) as img:
                    # Convert to JPG
                    if img.format != 'JPEG':
                        photo = os.path.splitext(photo)[0] + '.jpg'
                        img = img.convert('RGB')

                    # Resize if not 224x224
                    if img.size != (224, 224):
                        img = img.resize((224, 224), Image.Resampling.LANCZOS)

                    # Save to new location
                    save_path = os.path.join(output_path, folder, subfolder_name, os.path.basename(photo))
                    img.save(save_path)
                    count += 1
            except Exception as e:
                print(f"Error processing {photo}: {e}")

        return count

    # Process and move photos
    moved_train = process_and_move(train_photos, 'train')
    moved_test = process_and_move(test_photos, 'test')
    moved_val = process_and_move(val_photos, 'val')

    return moved_train, moved_test, moved_val

# Usage
moved_train, moved_test, moved_val = prepare_photo_folders([r'D:\yolo\output'], "D:\Capstone\X-Model-Transformers\data\output", "fake")
