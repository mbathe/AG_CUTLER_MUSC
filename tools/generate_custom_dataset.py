#!/usr/bin/env python3
"""
Script pour générer un dataset personnalisé avec des images synthétiques et leurs masques de segmentation.
Ce script crée un dataset au format COCO avec des formes géométriques simples.
"""

import os
import json
import numpy as np
import cv2
from PIL import Image, ImageDraw
import random
from datetime import datetime
import argparse
from typing import List, Dict, Any


def create_synthetic_image_with_masks(width: int = 512, height: int = 512, num_objects: int = None) -> tuple:
    """
    Crée une image synthétique avec des objets géométriques et retourne l'image avec les masques.
    
    Returns:
        tuple: (image_rgb, masks_list, categories_list)
    """
    if num_objects is None:
        num_objects = random.randint(2, 5)
    
    # Créer une image de fond
    background_color = (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255))
    image = Image.new('RGB', (width, height), background_color)
    draw = ImageDraw.Draw(image)
    
    masks = []
    categories = []
    
    shapes = ['circle', 'rectangle', 'triangle']
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128)
    ]
    
    for i in range(num_objects):
        shape = random.choice(shapes)
        color = random.choice(colors)
        
        # Créer un masque pour cet objet
        mask = Image.new('L', (width, height), 0)
        mask_draw = ImageDraw.Draw(mask)
        
        if shape == 'circle':
            # Cercle
            center_x = random.randint(50, width - 50)
            center_y = random.randint(50, height - 50)
            radius = random.randint(20, 60)
            
            bbox = (center_x - radius, center_y - radius, center_x + radius, center_y + radius)
            draw.ellipse(bbox, fill=color)
            mask_draw.ellipse(bbox, fill=255)
            categories.append(0)  # Toutes les formes sont maintenant "defect"
            
        elif shape == 'rectangle':
            # Rectangle
            x1 = random.randint(50, width - 100)
            y1 = random.randint(50, height - 100)
            x2 = x1 + random.randint(40, 80)
            y2 = y1 + random.randint(40, 80)
            
            bbox = (x1, y1, x2, y2)
            draw.rectangle(bbox, fill=color)
            mask_draw.rectangle(bbox, fill=255)
            categories.append(0)  # Toutes les formes sont maintenant "defect"
            
        elif shape == 'triangle':
            # Triangle
            x1 = random.randint(50, width - 100)
            y1 = random.randint(50, height - 100)
            x2 = x1 + random.randint(-30, 30)
            y2 = y1 + random.randint(40, 80)
            x3 = x1 + random.randint(40, 80)
            y3 = y1 + random.randint(40, 80)
            
            points = [(x1, y1), (x2, y2), (x3, y3)]
            draw.polygon(points, fill=color)
            mask_draw.polygon(points, fill=255)
            categories.append(0)  # Toutes les formes sont maintenant "defect"
        
        masks.append(np.array(mask))
    
    return np.array(image), masks, categories


def mask_to_polygon(mask: np.ndarray) -> List[List[float]]:
    """
    Convertit un masque binaire en polygone au format COCO.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    
    for contour in contours:
        if len(contour) >= 3:  # Au moins 3 points pour former un polygone
            polygon = contour.flatten().tolist()
            if len(polygon) >= 6:  # Au moins 3 points (x,y)
                polygons.append(polygon)
    
    return polygons


def create_coco_annotation(image_id: int, annotation_id: int, mask: np.ndarray, category_id: int) -> Dict[str, Any]:
    """
    Crée une annotation au format COCO à partir d'un masque.
    """
    # Convertir le masque en polygone
    polygons = mask_to_polygon(mask)
    
    if not polygons:
        return None
    
    # Calculer la bounding box
    y_indices, x_indices = np.where(mask > 0)
    if len(x_indices) == 0 or len(y_indices) == 0:
        return None
    
    x_min, x_max = int(np.min(x_indices)), int(np.max(x_indices))
    y_min, y_max = int(np.min(y_indices)), int(np.max(y_indices))
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    
    # Calculer l'aire
    area = int(np.sum(mask > 0))
    
    annotation = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "segmentation": polygons,
        "area": area,
        "bbox": [x_min, y_min, width, height],
        "iscrowd": 0
    }
    
    return annotation


def generate_dataset(output_dir: str, num_images: int = 100, split: str = "train"):
    """
    Génère un dataset complet avec images et annotations COCO.
    """
    # Créer les dossiers
    images_dir = os.path.join(output_dir, "images", split)
    annotations_dir = os.path.join(output_dir, "annotations")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)
    
    # Initialiser la structure COCO
    coco_data = {
        "info": {
            "description": "Custom Synthetic Dataset",
            "version": "1.0",
            "year": 2024,
            "contributor": "Custom Dataset Generator",
            "date_created": datetime.now().isoformat()
        },
        "licenses": [
            {
                "id": 1,
                "name": "Custom License",
                "url": ""
            }
        ],
        "categories": [
            {"id": 0, "name": "defect", "supercategory": "thing"}
        ],
        "images": [],
        "annotations": []
    }
    
    annotation_id = 1
    
    print(f"Génération de {num_images} images pour le split '{split}'...")
    
    for image_id in range(1, num_images + 1):
        if image_id % 10 == 0:
            print(f"Généré {image_id}/{num_images} images")
        
        # Générer l'image et les masques
        image, masks, categories = create_synthetic_image_with_masks()
        
        # Sauvegarder l'image
        image_filename = f"{split}_{image_id:06d}.jpg"
        image_path = os.path.join(images_dir, image_filename)
        Image.fromarray(image).save(image_path, "JPEG", quality=95)
        
        # Ajouter les métadonnées de l'image
        image_info = {
            "id": image_id,
            "width": image.shape[1],
            "height": image.shape[0],
            "file_name": image_filename,
            "license": 1,
            "date_captured": datetime.now().isoformat()
        }
        coco_data["images"].append(image_info)
        
        # Créer les annotations pour chaque objet
        for mask, category_id in zip(masks, categories):
            annotation = create_coco_annotation(image_id, annotation_id, mask, category_id)
            if annotation is not None:
                coco_data["annotations"].append(annotation)
                annotation_id += 1
    
    # Sauvegarder les annotations
    annotations_filename = f"instances_{split}.json"
    annotations_path = os.path.join(annotations_dir, annotations_filename)
    
    with open(annotations_path, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"Dataset généré avec succès!")
    print(f"Images: {images_dir}")
    print(f"Annotations: {annotations_path}")
    print(f"Nombre total d'annotations: {len(coco_data['annotations'])}")
    
    return annotations_path


def main():
    parser = argparse.ArgumentParser(description="Génère un dataset personnalisé au format COCO")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Répertoire de sortie pour le dataset")
    parser.add_argument("--num-train", type=int, default=100,
                       help="Nombre d'images d'entraînement")
    parser.add_argument("--num-val", type=int, default=20,
                       help="Nombre d'images de validation")
    
    args = parser.parse_args()
    
    # Générer le dataset d'entraînement
    print("Génération du dataset d'entraînement...")
    generate_dataset(args.output_dir, args.num_train, "train")
    
    # Générer le dataset de validation
    print("\nGénération du dataset de validation...")
    generate_dataset(args.output_dir, args.num_val, "val")
    
    print(f"\nDataset complet généré dans: {args.output_dir}")


if __name__ == "__main__":
    main()
