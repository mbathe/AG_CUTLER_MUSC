#!/usr/bin/env python3

import os
import cv2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2 import model_zoo
import numpy as np

def test_defect_model():
    """Test du modèle de détection de défauts"""
    model_path = "./output_defect/model_final.pth"
    test_images = "./dataset_defect/images/val"
    output_dir = "./test_defect_results"
    
    print("=== TEST MODÈLE DÉTECTION DE DÉFAUTS ===")
    print(f"Modèle: {model_path}")
    print(f"Images test: {test_images}")
    print("Classe: 'defect' uniquement")
    print()
    
    if not os.path.exists(model_path):
        print(f"❌ Modèle non trouvé: {model_path}")
        return
    
    # Configuration
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    
    # Métadonnées
    MetadataCatalog.get("defect_test").thing_classes = ["defect"]
    
    # Prédicteur
    predictor = DefaultPredictor(cfg)
    print("✅ Prédicteur configuré")
    
    # Créer dossier résultats
    os.makedirs(output_dir, exist_ok=True)
    
    # Tester sur images
    image_files = [f for f in os.listdir(test_images) if f.endswith(('.jpg', '.png'))][:5]
    
    total_detections = 0
    total_score = 0
    
    for img_file in image_files:
        img_path = os.path.join(test_images, img_file)
        img = cv2.imread(img_path)
        
        # Prédiction
        outputs = predictor(img)
        
        # Visualisation
        v = Visualizer(img[:, :, ::-1], MetadataCatalog.get("defect_test"), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        
        # Sauvegarder
        result_img = out.get_image()[:, :, ::-1]
        output_path = os.path.join(output_dir, f"defect_detected_{img_file}")
        cv2.imwrite(output_path, result_img)
        
        # Statistiques
        instances = outputs["instances"]
        num_detections = len(instances)
        scores = instances.scores.cpu().numpy()
        
        print(f"Image: {img_file}")
        print(f"  Défauts détectés: {num_detections}")
        if num_detections > 0:
            print(f"  Scores: {scores}")
            print(f"  Score moyen: {np.mean(scores):.3f}")
            total_detections += num_detections
            total_score += np.sum(scores)
        print()
    
    if total_detections > 0:
        avg_score = total_score / total_detections
        print(f"📊 RÉSUMÉ:")
        print(f"  Total défauts détectés: {total_detections}")
        print(f"  Score moyen global: {avg_score:.3f}")
    
    print(f"✅ Résultats sauvés dans: {output_dir}")

if __name__ == "__main__":
    test_defect_model()
