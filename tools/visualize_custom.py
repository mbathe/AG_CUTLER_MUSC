#!/usr/bin/env python3
"""
Script pour visualiser les résultats de CutLER sur un dataset personnalisé.
"""

import os
import sys
import cv2
import json
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon

# Ajouter le chemin du projet au PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from detectron2.config import get_cfg
    from detectron2.data import MetadataCatalog
    from detectron2.engine import DefaultPredictor
    from detectron2.utils.visualizer import Visualizer, ColorMode
    from cutler.config import add_cutler_config
    from cutler.data.datasets.custom_datasets import register_dataset_from_path, get_custom_metadata
    DETECTRON2_AVAILABLE = True
except ImportError:
    DETECTRON2_AVAILABLE = False
    print("Detectron2 non disponible. Utilisation du mode visualisation simple.")


def load_coco_annotations(annotation_file: str) -> dict:
    """
    Charge les annotations COCO depuis un fichier JSON.
    """
    with open(annotation_file, 'r') as f:
        return json.load(f)


def visualize_ground_truth(images_dir: str, annotations_file: str, output_dir: str, num_images: int = 10):
    """
    Visualise les annotations ground truth du dataset.
    """
    print(f"Visualisation des annotations ground truth...")
    
    # Charger les annotations
    coco_data = load_coco_annotations(annotations_file)
    
    # Créer le mapping image_id -> image_info
    images_dict = {img['id']: img for img in coco_data['images']}
    categories_dict = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # Grouper les annotations par image
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)
    
    # Créer le dossier de sortie
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualiser les premières images
    image_ids = list(annotations_by_image.keys())[:num_images]
    
    for i, image_id in enumerate(image_ids):
        print(f"Visualisation image {i+1}/{len(image_ids)}")
        
        # Charger l'image
        image_info = images_dict[image_id]
        image_path = os.path.join(images_dir, image_info['file_name'])
        
        if not os.path.exists(image_path):
            print(f"Image non trouvée: {image_path}")
            continue
        
        image = Image.open(image_path).convert('RGB')
        
        # Créer la visualisation
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)
        ax.set_title(f"Ground Truth - {image_info['file_name']}")
        
        # Ajouter les annotations
        annotations = annotations_by_image[image_id]
        colors = plt.cm.Set3(np.linspace(0, 1, len(categories_dict)))
        
        for ann in annotations:
            category_name = categories_dict[ann['category_id']]
            color = colors[ann['category_id'] - 1]
            
            # Dessiner la bounding box
            bbox = ann['bbox']  # [x, y, width, height]
            rect = patches.Rectangle(
                (bbox[0], bbox[1]), bbox[2], bbox[3],
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            
            # Ajouter le label
            ax.text(bbox[0], bbox[1] - 5, category_name, 
                   fontsize=10, color=color, weight='bold')
            
            # Dessiner le masque si disponible
            if 'segmentation' in ann and ann['segmentation']:
                for seg in ann['segmentation']:
                    if len(seg) >= 6:  # Au moins 3 points
                        poly_points = np.array(seg).reshape(-1, 2)
                        polygon = Polygon(poly_points, alpha=0.3, facecolor=color)
                        ax.add_patch(polygon)
        
        ax.axis('off')
        
        # Sauvegarder
        output_path = os.path.join(output_dir, f"gt_visualization_{i+1:03d}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Visualisations sauvegardées dans: {output_dir}")


def visualize_predictions(model_config: str, model_weights: str, images_dir: str, 
                         output_dir: str, num_images: int = 10):
    """
    Visualise les prédictions du modèle entraîné.
    """
    if not DETECTRON2_AVAILABLE:
        print("Detectron2 requis pour la visualisation des prédictions.")
        return
    
    print("Visualisation des prédictions du modèle...")
    
    # Configurer le modèle
    cfg = get_cfg()
    add_cutler_config(cfg)
    cfg.merge_from_file(model_config)
    cfg.MODEL.WEIGHTS = model_weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    
    # Créer le prédicteur
    predictor = DefaultPredictor(cfg)
    
    # Obtenir les métadonnées
    metadata = get_custom_metadata()
    
    # Créer le dossier de sortie
    os.makedirs(output_dir, exist_ok=True)
    
    # Lister les images
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files = sorted(image_files)[:num_images]
    
    for i, image_file in enumerate(image_files):
        print(f"Prédiction image {i+1}/{len(image_files)}")
        
        # Charger l'image
        image_path = os.path.join(images_dir, image_file)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Impossible de charger l'image: {image_path}")
            continue
        
        # Faire la prédiction
        outputs = predictor(image)
        
        # Visualiser
        v = Visualizer(
            image[:, :, ::-1], 
            metadata=metadata, 
            scale=1.0,
            instance_mode=ColorMode.IMAGE_BW
        )
        
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        result_image = out.get_image()[:, :, ::-1]
        
        # Sauvegarder
        output_path = os.path.join(output_dir, f"pred_visualization_{i+1:03d}.png")
        cv2.imwrite(output_path, result_image)
    
    print(f"Prédictions sauvegardées dans: {output_dir}")


def compare_gt_and_predictions(images_dir: str, annotations_file: str, 
                              model_config: str, model_weights: str,
                              output_dir: str, num_images: int = 5):
    """
    Compare les annotations ground truth avec les prédictions du modèle.
    """
    if not DETECTRON2_AVAILABLE:
        print("Detectron2 requis pour la comparaison.")
        return
    
    print("Comparaison GT vs Prédictions...")
    
    # Configurer le modèle
    cfg = get_cfg()
    add_cutler_config(cfg)
    cfg.merge_from_file(model_config)
    cfg.MODEL.WEIGHTS = model_weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    
    predictor = DefaultPredictor(cfg)
    metadata = get_custom_metadata()
    
    # Charger les annotations
    coco_data = load_coco_annotations(annotations_file)
    images_dict = {img['id']: img for img in coco_data['images']}
    categories_dict = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # Grouper les annotations par image
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)
    
    # Créer le dossier de sortie
    os.makedirs(output_dir, exist_ok=True)
    
    # Comparer les premières images
    image_ids = list(annotations_by_image.keys())[:num_images]
    
    for i, image_id in enumerate(image_ids):
        print(f"Comparaison image {i+1}/{len(image_ids)}")
        
        image_info = images_dict[image_id]
        image_path = os.path.join(images_dir, image_info['file_name'])
        
        if not os.path.exists(image_path):
            continue
        
        # Charger l'image
        image_cv = cv2.imread(image_path)
        image_pil = Image.open(image_path).convert('RGB')
        
        # Faire la prédiction
        outputs = predictor(image_cv)
        
        # Créer la visualisation côte à côte
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Ground Truth
        ax1.imshow(image_pil)
        ax1.set_title(f"Ground Truth - {image_info['file_name']}")
        
        annotations = annotations_by_image[image_id]
        colors = plt.cm.Set3(np.linspace(0, 1, len(categories_dict)))
        
        for ann in annotations:
            category_name = categories_dict[ann['category_id']]
            color = colors[ann['category_id'] - 1]
            
            bbox = ann['bbox']
            rect = patches.Rectangle(
                (bbox[0], bbox[1]), bbox[2], bbox[3],
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax1.add_patch(rect)
            ax1.text(bbox[0], bbox[1] - 5, category_name, 
                    fontsize=10, color=color, weight='bold')
        
        ax1.axis('off')
        
        # Prédictions
        v = Visualizer(
            image_cv[:, :, ::-1], 
            metadata=metadata, 
            scale=1.0,
            instance_mode=ColorMode.IMAGE_BW
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        
        ax2.imshow(out.get_image()[:, :, ::-1])
        ax2.set_title(f"Prédictions - {image_info['file_name']}")
        ax2.axis('off')
        
        # Sauvegarder
        output_path = os.path.join(output_dir, f"comparison_{i+1:03d}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Comparaisons sauvegardées dans: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Visualiser les résultats de CutLER sur un dataset personnalisé")
    parser.add_argument("--dataset-path", type=str, required=True,
                       help="Chemin vers le dataset personnalisé")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Répertoire de sortie pour les visualisations")
    parser.add_argument("--model-config", type=str,
                       help="Fichier de configuration du modèle")
    parser.add_argument("--model-weights", type=str,
                       help="Poids du modèle entraîné")
    parser.add_argument("--num-images", type=int, default=10,
                       help="Nombre d'images à visualiser")
    parser.add_argument("--mode", choices=["gt", "pred", "compare"], default="gt",
                       help="Mode de visualisation: gt (ground truth), pred (prédictions), compare (comparaison)")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val",
                       help="Split du dataset à visualiser")
    
    args = parser.parse_args()
    
    # Chemins du dataset
    images_dir = os.path.join(args.dataset_path, "images", args.split)
    annotations_file = os.path.join(args.dataset_path, "annotations", f"instances_{args.split}.json")
    
    # Vérifier que les fichiers existent
    if not os.path.exists(images_dir):
        raise ValueError(f"Dossier d'images non trouvé: {images_dir}")
    if not os.path.exists(annotations_file):
        raise ValueError(f"Fichier d'annotations non trouvé: {annotations_file}")
    
    # Créer le dossier de sortie
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Mode de visualisation: {args.mode}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Split: {args.split}")
    print(f"Sortie: {args.output_dir}")
    
    if args.mode == "gt":
        visualize_ground_truth(images_dir, annotations_file, args.output_dir, args.num_images)
    
    elif args.mode == "pred":
        if not args.model_config or not args.model_weights:
            raise ValueError("--model-config et --model-weights requis pour le mode 'pred'")
        visualize_predictions(args.model_config, args.model_weights, images_dir, 
                            args.output_dir, args.num_images)
    
    elif args.mode == "compare":
        if not args.model_config or not args.model_weights:
            raise ValueError("--model-config et --model-weights requis pour le mode 'compare'")
        compare_gt_and_predictions(images_dir, annotations_file, args.model_config, 
                                 args.model_weights, args.output_dir, args.num_images)


if __name__ == "__main__":
    main()
