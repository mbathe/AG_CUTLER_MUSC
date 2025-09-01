#!/usr/bin/env python3

import os
import cv2
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2 import model_zoo
import matplotlib.pyplot as plt
from PIL import Image

def test_industrial_model():
    """Test du modèle de détection industriel"""
    model_path = "./output_defect_gpu/model_final.pth"
    test_images = "./industrial_dataset/images/val"
    test_probs = "./industrial_dataset/probability_maps/val"
    output_dir = "./test_industrial_results"
    
    print("=== TEST MODÈLE DÉTECTION INDUSTRIELLE ===")
    print(f"Modèle: {model_path}")
    print(f"Images test: {test_images}")
    print("Application: Contrôle qualité industriel")
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
    MetadataCatalog.get("industrial_test").thing_classes = ["defect"]
    
    # Prédicteur
    predictor = DefaultPredictor(cfg)
    print("✅ Prédicteur industriel configuré")
    
    # Créer dossier résultats
    os.makedirs(output_dir, exist_ok=True)
    
    # Tester sur images
    image_files = [f for f in os.listdir(test_images) if f.endswith('.jpg')][:20]
    
    total_detections = 0
    total_score = 0
    
    for img_file in image_files:
        # Charger image industrielle
        img_path = os.path.join(test_images, img_file)
        img_gray = np.array(Image.open(img_path))
        img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        
        # Charger matrice de probabilité correspondante
        prob_file = img_file.replace('.jpg', '_prob.npy')
        prob_path = os.path.join(test_probs, prob_file)
        probability_matrix = np.load(prob_path) if os.path.exists(prob_path) else None
        
        # Prédiction CutLER
        outputs = predictor(img_rgb)
        
        # Créer visualisation comparative
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Test Industriel: {img_file}', fontsize=14)
        
        # Image originale
        axes[0, 0].imshow(img_gray, cmap='gray')
        axes[0, 0].set_title('Image Industrielle')
        axes[0, 0].axis('off')
        
        # Matrice de probabilité (votre fonction)
        if probability_matrix is not None:
            im = axes[0, 1].imshow(probability_matrix, cmap='hot', vmin=0, vmax=1)
            axes[0, 1].set_title('Matrice de Probabilité')
            axes[0, 1].axis('off')
            plt.colorbar(im, ax=axes[0, 1], fraction=0.046)
        else:
            axes[0, 1].text(0.5, 0.5, 'Probabilité\nnon disponible', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].axis('off')
        
        # Détections CutLER
        v = Visualizer(img_rgb, MetadataCatalog.get("industrial_test"), scale=1.0)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        result_img = out.get_image()
        
        axes[1, 0].imshow(result_img)
        axes[1, 0].set_title('Détections CutLER')
        axes[1, 0].axis('off')
        
        # Comparaison masque vs détections
        if probability_matrix is not None:
            # Créer masque binaire à partir de la probabilité
            binary_mask = (probability_matrix > 0.5).astype(np.uint8) * 255
            
            # Superposer masque et détections
            comparison = img_rgb.copy()
            
            # Masque de probabilité en rouge
            mask_overlay = np.zeros_like(comparison)
            mask_overlay[binary_mask > 0] = [255, 0, 0]  # Rouge pour probabilité
            comparison = cv2.addWeighted(comparison, 0.7, mask_overlay, 0.3, 0)
            
            # Bounding boxes en vert
            instances = outputs["instances"]
            for i in range(len(instances)):
                bbox = instances.pred_boxes[i].tensor.cpu().numpy()[0]
                x1, y1, x2, y2 = bbox.astype(int)
                cv2.rectangle(comparison, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            axes[1, 1].imshow(comparison)
            axes[1, 1].set_title('Comparaison:\nRouge=Probabilité, Vert=CutLER')
            axes[1, 1].axis('off')
        else:
            axes[1, 1].imshow(result_img)
            axes[1, 1].set_title('Détections CutLER (sans comparaison)')
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Sauvegarder
        output_path = os.path.join(output_dir, f"industrial_test_{img_file.replace('.jpg', '.png')}")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Statistiques
        instances = outputs["instances"]
        num_detections = len(instances)
        scores = instances.scores.cpu().numpy()
        
        print(f"Image: {img_file}")
        print(f"  Défauts détectés par CutLER: {num_detections}")
        if num_detections > 0:
            print(f"  Scores: {[f'{s:.3f}' for s in scores]}")
            print(f"  Score moyen: {np.mean(scores):.3f}")
            total_detections += num_detections
            total_score += np.sum(scores)
        
        if probability_matrix is not None:
            high_prob_pixels = np.sum(probability_matrix > 0.5)
            print(f"  Pixels haute probabilité: {high_prob_pixels}")
            if high_prob_pixels > 0:
                avg_prob = np.mean(probability_matrix[probability_matrix > 0.5])
                print(f"  Probabilité moyenne: {avg_prob:.3f}")
        print()
    
    if total_detections > 0:
        avg_score = total_score / total_detections
        print(f"📊 RÉSUMÉ INDUSTRIEL:")
        print(f"  Total défauts détectés: {total_detections}")
        print(f"  Score moyen global: {avg_score:.3f}")
    
    print(f"✅ Tests industriels sauvés dans: {output_dir}")
    print("\n💡 INTÉGRATION AVEC VOTRE SYSTÈME:")
    print("  1. Remplacez la génération de matrices de probabilité par votre fonction")
    print("  2. Le modèle CutLER utilise ces matrices pour apprendre les bounding boxes")
    print("  3. En production: Probabilité → Masque → CutLER → Bounding boxes")

if __name__ == "__main__":
    test_industrial_model()
