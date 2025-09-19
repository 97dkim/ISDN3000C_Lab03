# classify_batch.py
#
# FOR THE ADVANCED TASK
# This script processes all images in a specified folder, runs inference on each,
# and saves the results to a CSV file.
#
# Usage:
# python classify_batch.py <path_to_image_folder>

import os
import sys
import csv
import torch
from torchvision import models, transforms
from PIL import Image
import json
import urllib.request

LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"

def get_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.eval()
    return model

def get_labels():
    with urllib.request.urlopen(LABELS_URL) as url:
        labels = json.loads(url.read().decode())
    return labels

def process_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert('RGB')
    img_t = preprocess(img)
    batch_t = torch.unsqueeze(img_t, 0)
    return batch_t

def predict(model, image_tensor, labels):
    with torch.no_grad():
        output = model(image_tensor)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top1_prob, top1_cat_id = torch.topk(probabilities, 1)
    category_name = labels[top1_cat_id.item()]
    confidence_score = top1_prob.item()
    return category_name, confidence_score

def main(image_folder):
    model = get_model()
    labels = get_labels()
    results = []

    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(image_folder, filename)
            try:
                image_tensor = process_image(image_path)
                category, confidence = predict(model, image_tensor, labels)
                results.append({
                    'image_name': filename,
                    'detected_class': category,
                    'confidence_level': f"{confidence:.4f}"
                })
                print(f"{filename}: {category} ({confidence:.2%})")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    # Save results to results.csv in the current directory
    csv_path = os.path.join(os.getcwd(), 'results.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['image_name', 'detected_class', 'confidence_level']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"Results saved to {csv_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python classify_batch.py <path_to_image_folder>")
        sys.exit(1)
    main(sys.argv[1])
