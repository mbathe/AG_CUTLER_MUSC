#!/usr/bin/env python3

import os
import cv2
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2 import model_zoo
import matplotlib.pyplot as plt
import numpy as np

def setup_predictor(model_path, config_file=None):
    """Configure le prédicteur avec le modèle entraîné"""
    cfg = get_cfg()
    
    if config_file and os.path.exists(config_file):
        cfg.merge_from_file(config_file)
    else:
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Une seule classe "object"
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Seuil de confiance
    
    predictor = DefaultPredictor(cfg)
    return predictor

def test_on_images(predictor, image_dir, output_dir):
    """Test le modèle sur des images"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Métadonnées pour la visualisation
    MetadataCatalog.get("test").thing_classes = ["defect"]
    
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))][:5]  # Test sur 5 images
    
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        img = cv2.imread(img_path)
        
        # Prédiction
        outputs = predictions = predictor(img)
        
        # Visualisation
        v = Visualizer(img[:, :, ::-1], MetadataCatalog.get("test"), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        
        # Sauvegarder
        result_img = out.get_image()[:, :, ::-1]
        output_path = os.path.join(output_dir, f"detected_{img_file}")
        cv2.imwrite(output_path, result_img)
        
        # Afficher les statistiques
        instances = outputs["instances"]
        num_detections = len(instances)
        scores = instances.scores.cpu().numpy()
        
        print(f"Image: {img_file}")
        print(f"  Objets détectés: {num_detections}")
        if num_detections > 0:
            print(f"  Scores: {scores}")
            print(f"  Score moyen: {np.mean(scores):.3f}")
        print()

def main():
    model_path = "./output_single_class/model_final.pth"
    test_images = "./test_dataset_single_class/images/val"
    output_dir = "./test_results"
    
    print("=== TEST DU MODÈLE CUTLER-STYLE ===")
    print(f"Modèle: {model_path}")
    print(f"Images test: {test_images}")
    print("Objectif: Détecter tous les objets comme 'object' générique")
    print()
    
    if not os.path.exists(model_path):
        print(f"Erreur: Modèle non trouvé à {model_path}")
        return
    
    # Configurer le prédicteur
    predictor = setup_predictor(model_path)
    print("✅ Prédicteur configuré")
    
    # Tester sur les images
    test_on_images(predictor, test_images, output_dir)
    
    print(f"✅ Tests terminés! Résultats dans {output_dir}")
    print("\nCe modèle implémente l'approche CutLER:")
    print("- Détection d'objets sans classification spécifique")
    print("- Une seule classe 'object' pour tous les objets")
    print("- Focus sur la localisation plutôt que la catégorisation")

if __name__ == "__main__":
    main()
