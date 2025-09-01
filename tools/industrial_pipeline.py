#!/usr/bin/env python3

import os
import numpy as np
import cv2
from PIL import Image
import json
from datetime import datetime
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo
from skimage import measure

class IndustrialDefectPipeline:
    """Pipeline complet de détection de défauts industriels avec CutLER"""
    
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.predictor = None
        
        if model_path and os.path.exists(model_path):
            self.setup_cutler_model()
    
    def setup_cutler_model(self):
        """Configure le modèle CutLER entraîné"""
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.MODEL.WEIGHTS = self.model_path
        cfg.MODEL.DEVICE = "cpu"  # ou "cuda" si GPU disponible
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        
        MetadataCatalog.get("industrial").thing_classes = ["defect"]
        self.predictor = DefaultPredictor(cfg)
        print("✅ Modèle CutLER chargé et prêt")
    
    def your_probability_function(self, image):
        """
        🔄 REMPLACEZ CETTE FONCTION PAR LA VÔTRE
        
        Cette fonction simule votre algorithme qui calcule la probabilité 
        qu'un pixel soit un défaut.
        
        Args:
            image: numpy array (H, W) en niveaux de gris
            
        Returns:
            probability_matrix: numpy array (H, W) avec valeurs entre 0 et 1
        """
        # SIMULATION - remplacez par votre vraie fonction
        print("⚠️  Utilisation de la fonction de probabilité SIMULÉE")
        print("    Remplacez cette fonction par votre algorithme réel")
        
        # Exemple simplifié de détection de défauts
        # Votre vraie fonction sera bien plus sophistiquée
        
        # Calculer gradient d'intensité
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normaliser entre 0 et 1
        gradient_norm = gradient_magnitude / (np.max(gradient_magnitude) + 1e-8)
        
        # Détecter anomalies de luminosité
        mean_intensity = np.mean(image)
        intensity_deviation = np.abs(image - mean_intensity) / (np.std(image) + 1e-8)
        intensity_norm = np.clip(intensity_deviation / 3.0, 0, 1)
        
        # Combiner les indices
        probability_matrix = 0.6 * gradient_norm + 0.4 * intensity_norm
        
        # Appliquer seuillage doux
        probability_matrix = np.clip(probability_matrix, 0, 1)
        
        return probability_matrix.astype(np.float32)
    
    def create_binary_mask(self, probability_matrix, threshold=0.5):
        """Crée un masque binaire à partir de la matrice de probabilité"""
        return (probability_matrix > threshold).astype(np.uint8) * 255
    
    def extract_bounding_boxes(self, binary_mask, min_area=50):
        """Extrait les bounding boxes des régions de défauts"""
        # Trouver les composantes connexes
        labeled_mask = measure.label(binary_mask > 0)
        regions = measure.regionprops(labeled_mask)
        
        bounding_boxes = []
        areas = []
        
        for region in regions:
            if region.area < min_area:
                continue
            
            # Extraire bounding box
            min_row, min_col, max_row, max_col = region.bbox
            x, y = min_col, min_row
            width, height = max_col - min_col, max_row - min_row
            
            bounding_boxes.append([x, y, width, height])
            areas.append(width * height)
        
        return bounding_boxes, areas
    
    def detect_defects_traditional(self, image_path):
        """Détection traditionnelle basée sur votre fonction de probabilité"""
        # Charger image
        image = np.array(Image.open(image_path).convert('L'))
        
        # Calculer probabilités avec votre fonction
        probability_matrix = self.your_probability_function(image)
        
        # Créer masque binaire
        binary_mask = self.create_binary_mask(probability_matrix)
        
        # Extraire bounding boxes
        bboxes, areas = self.extract_bounding_boxes(binary_mask)
        
        return {
            'image': image,
            'probability_matrix': probability_matrix,
            'binary_mask': binary_mask,
            'bounding_boxes': bboxes,
            'areas': areas,
            'method': 'traditional_probability'
        }
    
    def detect_defects_cutler(self, image_path):
        """Détection avec CutLER (si modèle disponible)"""
        if self.predictor is None:
            raise ValueError("Modèle CutLER non chargé. Utilisez setup_cutler_model() d'abord.")
        
        # Charger image
        image = np.array(Image.open(image_path).convert('L'))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Prédiction CutLER
        outputs = self.predictor(image_rgb)
        instances = outputs["instances"]
        
        # Extraire résultats
        bboxes = []
        scores = []
        for i in range(len(instances)):
            bbox = instances.pred_boxes[i].tensor.cpu().numpy()[0]
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            bboxes.append([int(x1), int(y1), int(w), int(h)])
            scores.append(float(instances.scores[i].cpu().numpy()))
        
        return {
            'image': image,
            'bounding_boxes': bboxes,
            'scores': scores,
            'method': 'cutler'
        }
    
    def compare_methods(self, image_path, output_dir="./comparison_results"):
        """Compare les deux méthodes de détection"""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"🔍 Analyse de: {os.path.basename(image_path)}")
        
        # Méthode traditionnelle
        traditional_result = self.detect_defects_traditional(image_path)
        print(f"  Méthode traditionnelle: {len(traditional_result['bounding_boxes'])} défauts")
        
        # Méthode CutLER (si disponible)
        cutler_result = None
        if self.predictor is not None:
            cutler_result = self.detect_defects_cutler(image_path)
            print(f"  Méthode CutLER: {len(cutler_result['bounding_boxes'])} défauts")
        
        # Visualisation
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Comparaison des méthodes: {os.path.basename(image_path)}', fontsize=14)
        
        # Image originale
        axes[0, 0].imshow(traditional_result['image'], cmap='gray')
        axes[0, 0].set_title('Image Industrielle')
        axes[0, 0].axis('off')
        
        # Matrice de probabilité
        im = axes[0, 1].imshow(traditional_result['probability_matrix'], cmap='hot', vmin=0, vmax=1)
        axes[0, 1].set_title('Matrice de Probabilité')
        axes[0, 1].axis('off')
        plt.colorbar(im, ax=axes[0, 1], fraction=0.046)
        
        # Détection traditionnelle
        img_traditional = cv2.cvtColor(traditional_result['image'], cv2.COLOR_GRAY2RGB)
        for bbox in traditional_result['bounding_boxes']:
            x, y, w, h = bbox
            cv2.rectangle(img_traditional, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        axes[1, 0].imshow(img_traditional)
        axes[1, 0].set_title(f'Détection Traditionnelle ({len(traditional_result["bounding_boxes"])} défauts)')
        axes[1, 0].axis('off')
        
        # Détection CutLER
        if cutler_result is not None:
            img_cutler = cv2.cvtColor(cutler_result['image'], cv2.COLOR_GRAY2RGB)
            for i, bbox in enumerate(cutler_result['bounding_boxes']):
                x, y, w, h = bbox
                score = cutler_result['scores'][i]
                cv2.rectangle(img_cutler, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img_cutler, f'{score:.2f}', (x, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            axes[1, 1].imshow(img_cutler)
            axes[1, 1].set_title(f'Détection CutLER ({len(cutler_result["bounding_boxes"])} défauts)')
        else:
            axes[1, 1].text(0.5, 0.5, 'Modèle CutLER\nnon disponible', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('CutLER (non disponible)')
        
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Sauvegarder
        output_path = os.path.join(output_dir, f"comparison_{os.path.basename(image_path).replace('.jpg', '.png')}")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  📊 Comparaison sauvée: {output_path}")
        
        return traditional_result, cutler_result

def example_usage():
    """Exemple d'utilisation du pipeline"""
    
    print("=" * 60)
    print("🏭 PIPELINE DE DÉTECTION DE DÉFAUTS INDUSTRIELS")
    print("=" * 60)
    
    # Initialiser le pipeline
    cutler_model_path = "./output_industrial/model_final.pth"
    pipeline = IndustrialDefectPipeline(cutler_model_path)
    
    # Tester sur quelques images
    test_images_dir = "./industrial_dataset/images/val"
    
    if os.path.exists(test_images_dir):
        image_files = [f for f in os.listdir(test_images_dir) if f.endswith('.jpg')][:3]
        
        for img_file in image_files:
            img_path = os.path.join(test_images_dir, img_file)
            traditional_result, cutler_result = pipeline.compare_methods(img_path)
    
    print("\n💡 INTÉGRATION DANS VOTRE SYSTÈME:")
    print("1. Remplacez 'your_probability_function()' par votre vraie fonction")
    print("2. La fonction doit retourner une matrice (H, W) avec probabilités [0, 1]")
    print("3. Le pipeline convertit automatiquement en bounding boxes")
    print("4. CutLER peut apprendre de vos données pour améliorer la détection")
    
    print("\n📋 CODE D'INTÉGRATION:")
    print("""
    # Dans votre système industriel:
    
    def ma_fonction_probabilite(image_gray):
        # Votre algorithme de détection
        # ...
        return probability_matrix  # shape (H, W), values [0, 1]
    
    # Remplacer dans le pipeline:
    pipeline.your_probability_function = ma_fonction_probabilite
    
    # Utilisation:
    result = pipeline.detect_defects_traditional("image_composant.jpg")
    defects = result['bounding_boxes']  # Liste des bounding boxes
    """)

if __name__ == "__main__":
    example_usage()
