# download_dataset.py

import os
import json
import urllib.request
import zipfile
from tqdm import tqdm
import argparse
import shutil

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    """Download a URL with a progress bar."""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def download_and_extract_dataset(dataset_name, output_dir):
    """Download and extract dataset."""
    os.makedirs(output_dir, exist_ok=True)
    
    if dataset_name == "flickr8k":
        # URLs for Flickr8k dataset
        image_url = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip"
        text_url = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip"
        
        # Download images
        image_zip_path = os.path.join(output_dir, "Flickr8k_Dataset.zip")
        print("Downloading Flickr8k images...")
        download_url(image_url, image_zip_path)
        
        # Download captions
        text_zip_path = os.path.join(output_dir, "Flickr8k_text.zip")
        print("Downloading Flickr8k captions...")
        download_url(text_url, text_zip_path)
        
        # Extract images
        print("Extracting images...")
        with zipfile.ZipFile(image_zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        
        # Extract captions
        print("Extracting captions...")
        with zipfile.ZipFile(text_zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        
        # Process captions to JSON format expected by our code
        print("Processing captions...")
        captions_file = os.path.join(output_dir, "Flickr8k_text", "Flickr8k.token.txt")
        
        # Create directories
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        
        # Move images to right location
        image_dir = os.path.join(output_dir, "Flickr8k_Dataset", "Flicker8k_Dataset")
        for img in os.listdir(image_dir):
            shutil.copy(os.path.join(image_dir, img), os.path.join(output_dir, "images", img))
        
        # Process captions
        annotations = []
        image_ids = set()
        
        with open(captions_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                img_name, caption = line.split('\t')
                img_id = img_name.split('.')[0]  # Remove extension
                
                annotations.append({
                    "image_id": img_id,
                    "caption": caption
                })
                image_ids.add(img_id)
        
        # Create JSON format
        coco_format = {
            "images": [{"id": img_id} for img_id in image_ids],
            "annotations": annotations
        }
        
        # Save as JSON
        captions_json = os.path.join(output_dir, "captions.json")
        with open(captions_json, 'w') as f:
            json.dump(coco_format, f)
        
        print(f"Dataset prepared and saved to {output_dir}")
        print(f"Number of images: {len(image_ids)}")
        print(f"Number of captions: {len(annotations)}")
        
    elif dataset_name == "coco":
        # For MS COCO, we'll use a smaller subset for simplicity
        # In a real project, you might want to download the full dataset
        
        # URLs for COCO 2017 dataset
        train_url = "http://images.cocodataset.org/zips/train2017.zip"
        val_url = "http://images.cocodataset.org/zips/val2017.zip"
        annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        
        # Download training images (large file - 18GB)
        train_zip_path = os.path.join(output_dir, "train2017.zip")
        print("Downloading COCO training images (this is a large file ~18GB)...")
        download_url(train_url, train_zip_path)
        
        # Download validation images
        val_zip_path = os.path.join(output_dir, "val2017.zip")
        print("Downloading COCO validation images...")
        download_url(val_url, val_zip_path)
        
        # Download annotations
        annotations_zip_path = os.path.join(output_dir, "annotations_trainval2017.zip")
        print("Downloading COCO annotations...")
        download_url(annotations_url, annotations_zip_path)
        
        # Extract files
        for zip_path, name in [
            (train_zip_path, "training images"),
            (val_zip_path, "validation images"),
            (annotations_zip_path, "annotations")
        ]:
            print(f"Extracting {name}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
        
        # Create directories
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        
        # Copy a subset of images for simplicity
        print("Copying a subset of images...")
        val_image_dir = os.path.join(output_dir, "val2017")
        for i, img in enumerate(os.listdir(val_image_dir)):
            if i >= 1000:  # Limit to 1000 images
                break
            shutil.copy(
                os.path.join(val_image_dir, img),
                os.path.join(output_dir, "images", f"{img.split('.')[0]}.jpg")
            )
        
        # Process annotations
        captions_file = os.path.join(output_dir, "annotations", "captions_val2017.json")
        
        with open(captions_file, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
        
        # Filter annotations for the subset of images
        image_ids = set()
        for i, img in enumerate(os.listdir(val_image_dir)):
            if i >= 1000:  # Limit to 1000 images
                break
            image_ids.add(int(img.split('.')[0]))
        
        filtered_images = [img for img in coco_data["images"] if img["id"] in image_ids]
        filtered_annotations = [ann for ann in coco_data["annotations"] if ann["image_id"] in image_ids]
        
        # Create filtered JSON
        filtered_coco = {
            "images": filtered_images,
            "annotations": filtered_annotations
        }
        
        # Save as JSON
        captions_json = os.path.join(output_dir, "captions.json")
        with open(captions_json, 'w') as f:
            json.dump(filtered_coco, f)
        
        print(f"Dataset prepared and saved to {output_dir}")
        print(f"Number of images: {len(filtered_images)}")
        print(f"Number of captions: {len(filtered_annotations)}")
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and prepare dataset for image captioning")
    parser.add_argument("--dataset", type=str, default="flickr8k", choices=["flickr8k", "coco"],
                       help="Dataset to download: flickr8k or coco")
    parser.add_argument("--output_dir", type=str, default="./data",
                       help="Output directory for the dataset")
    
    args = parser.parse_args()
    
    download_and_extract_dataset(args.dataset, args.output_dir)