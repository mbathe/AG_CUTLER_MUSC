#!/usr/bin/env python3

import os
import json
import numpy as np
from PIL import Image
import cv2
from datetime import datetime
import argparse
from skimage import measure
import glob
import random

from MuSc.musc_efficient_tester import MuScEfficientTester

# 1. Initialisation (une seule fois)

tester = MuScEfficientTester(
    reference_folder="/home/paul/Cours/Stage_ST/code/reference",
    model_name='dinov2_vitl14',
    image_size=(504, 504)
)


class IndustrialDefectGenerator:
    def __init__(self, images_dir, image_size=(504, 504), post_process_params=None):
        self.images_dir = images_dir
        self.image_size = image_size
        self.width, self.height = image_size
        
        # Paramètres de post-traitement par défaut
        self.default_post_process_params = {
            'threshold': 0.5,
            'morph_kernel_size': 3,
            'morph_operations': 'close',
            'min_area': 50,
            'gaussian_blur_sigma': 0.5,
            'apply_crf': False
        }
        
        # Fusionner avec les paramètres fournis
        if post_process_params:
            self.default_post_process_params.update(post_process_params)
        
        # Charger la liste des images disponibles
        self.available_images = self._load_available_images()
        if not self.available_images:
            raise ValueError(f"Aucune image trouvée dans le répertoire: {images_dir}")
        print(f"Trouvé {len(self.available_images)} images dans {images_dir}")
    
    def _load_available_images(self):
        """Charge la liste des images disponibles dans le répertoire"""
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        images = []
        
        for ext in image_extensions:
            pattern = os.path.join(self.images_dir, ext)
            images.extend(glob.glob(pattern))
            # Aussi chercher dans les sous-dossiers
            pattern = os.path.join(self.images_dir, '**', ext)
            images.extend(glob.glob(pattern, recursive=True))
        
        return sorted(list(set(images)))  # Éliminer les doublons et trier
    
    def load_and_resize_image(self, image_path):
        """Charge et redimensionne une image"""
        try:
            # Charger l'image
            image = Image.open(image_path)
            
            # Convertir en niveaux de gris si nécessaire
            if image.mode != 'L':
                image = image.convert('L')
            
            # Redimensionner à la taille souhaitée
            image = image.resize(self.image_size, Image.Resampling.LANCZOS)
            
            return np.array(image)
        except Exception as e:
            print(f"Erreur lors du chargement de l'image {image_path}: {e}")
            return None
    
    def generate_probability_matrix(self, image_path):
        """
        Génère une matrice de probabilité de défaut pour une image donnée.
        Cette méthode doit être implémentée selon votre algorithme de détection.
        
        Args:
            image_path (str): Chemin vers l'image à analyser
            
        Returns:
            np.ndarray: Matrice de probabilité (0-1) de même taille que l'image
            
        IMPORTANT: CETTE MÉTHODE DOIT ÊTRE ADAPTÉE SELON VOTRE ALGORITHME
        
        """
        # Charger l'image
        
        probability_matrix, score = tester.get_anomaly_map(image_path)
        
        
        return probability_matrix.astype(np.float32)
        
    def post_process_probability_matrix(self, probability_matrix, 
                                      threshold=0.5, 
                                      morph_kernel_size=3, 
                                      morph_operations='close',
                                      min_area=50,
                                      gaussian_blur_sigma=0.5,
                                      apply_crf=False):
        """
        Post-traite la matrice de probabilité pour améliorer la précision du masque
        
        Args:
            probability_matrix (np.ndarray): Matrice de probabilité d'anomalie (0-1)
            threshold (float): Seuil pour la binarisation (défaut: 0.5)
            morph_kernel_size (int): Taille du noyau pour les opérations morphologiques
            morph_operations (str): Types d'opérations morphologiques ('close', 'open', 'both')
            min_area (int): Aire minimale des composantes connexes à conserver
            gaussian_blur_sigma (float): Sigma pour le lissage gaussien (0 = pas de lissage)
            apply_crf (bool): Appliquer un CRF pour raffiner les contours
            
        Returns:
            np.ndarray: Matrice de probabilité post-traitée
        """
        # Copie pour éviter de modifier l'original
        processed_matrix = probability_matrix.copy()
        
        # 1. Lissage gaussien optionnel pour réduire le bruit
        if gaussian_blur_sigma > 0:
            processed_matrix = cv2.GaussianBlur(processed_matrix, 
                                              (0, 0), 
                                              gaussian_blur_sigma)
        
        # 2. Binarisation avec le seuil
        binary_mask = (processed_matrix > threshold).astype(np.uint8)
        
        # 3. Opérations morphologiques pour nettoyer le masque
        if morph_kernel_size > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                             (morph_kernel_size, morph_kernel_size))
            
            if morph_operations in ['close', 'both']:
                # Fermeture pour combler les trous
                binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
            
            if morph_operations in ['open', 'both']:
                # Ouverture pour éliminer les petits objets
                binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        
        # 4. Filtrage par aire minimale
        if min_area > 0:
            binary_mask = self._filter_small_components(binary_mask, min_area)
        
        # 5. CRF optionnel pour raffiner les contours (si disponible)
        if apply_crf:
            try:
                binary_mask = self._apply_crf_refinement(processed_matrix, binary_mask)
            except Exception as e:
                print(f"Attention: CRF non disponible ou erreur: {e}")
        
        # 6. Reconvertir en probabilités (0-1)
        refined_matrix = binary_mask.astype(np.float32)
        
        # 7. Optionnel: appliquer un gradient près des bords pour des transitions plus douces
        refined_matrix = self._apply_soft_edges(refined_matrix, edge_width=3)
        
        return refined_matrix
    
    def _filter_small_components(self, binary_mask, min_area):
        """Filtre les composantes connexes trop petites"""
        # Trouver les composantes connexes
        labeled_mask = measure.label(binary_mask)
        regions = measure.regionprops(labeled_mask)
        
        # Créer un nouveau masque sans les petites composantes
        filtered_mask = np.zeros_like(binary_mask)
        
        for region in regions:
            if region.area >= min_area:
                # Conserver cette composante
                coords = region.coords
                filtered_mask[coords[:, 0], coords[:, 1]] = 1
        
        return filtered_mask
    
    def _apply_crf_refinement(self, probability_matrix, binary_mask):
        """Applique un CRF pour raffiner les contours (méthode simple)"""
        # Implémentation simple basée sur les gradients
        # Pour un CRF complet, il faudrait intégrer une bibliothèque comme pydensecrf
        
        # Calculer les gradients pour identifier les bords
        grad_x = cv2.Sobel(probability_matrix, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(probability_matrix, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normaliser
        gradient_magnitude = gradient_magnitude / gradient_magnitude.max()
        
        # Affiner les bords en utilisant les gradients
        refined_mask = binary_mask.copy()
        
        # Zones avec fort gradient = bords potentiels
        edge_threshold = 0.3
        edge_pixels = gradient_magnitude > edge_threshold
        
        # Pour les pixels de bord, ajuster selon le gradient
        refined_mask[edge_pixels] = (probability_matrix[edge_pixels] > 0.3).astype(np.uint8)
        
        return refined_mask
    
    def _apply_soft_edges(self, binary_matrix, edge_width=3):
        """Applique des bords adoucis pour des transitions plus naturelles"""
        if edge_width <= 0:
            return binary_matrix
        
        # Calculer la distance à partir des bords
        distance_map = cv2.distanceTransform(binary_matrix.astype(np.uint8), 
                                           cv2.DIST_L2, 5)
        
        # Créer un masque avec des bords adoucis
        soft_mask = binary_matrix.copy()
        
        # Appliquer un gradient sur les bords
        edge_mask = (distance_map > 0) & (distance_map <= edge_width)
        soft_mask[edge_mask] = distance_map[edge_mask] / edge_width
        
        return soft_mask

    def create_binary_mask_from_probability(self, probability_matrix, threshold=0.5):
        """Crée un masque binaire à partir de la matrice de probabilité"""
        return (probability_matrix > threshold).astype(np.uint8) * 255
    
    def process_real_image(self, image_path, 
                          probability_threshold=0.5,
                          post_process_params=None):
        """
        Traite une vraie image pour extraire les informations nécessaires
        
        Args:
            image_path (str): Chemin vers l'image
            probability_threshold (float): Seuil de probabilité pour la binarisation
            post_process_params (dict): Paramètres de post-traitement
        """
        if post_process_params is None:
            post_process_params = {
                'threshold': probability_threshold,
                'morph_kernel_size': 3,
                'morph_operations': 'close',
                'min_area': 50,
                'gaussian_blur_sigma': 0.5,
                'apply_crf': False
            }
        
        # Charger l'image
        image = self.load_and_resize_image(image_path)
        if image is None:
            return None
        
        # Générer la matrice de probabilité brute
        raw_probability_matrix = self.generate_probability_matrix(image_path)
        
        # Appliquer le post-traitement pour améliorer le masque
        processed_probability_matrix = self.post_process_probability_matrix(
            raw_probability_matrix, **post_process_params
        )
        
        # Créer le masque binaire à partir de la matrice post-traitée
        binary_mask = self.create_binary_mask_from_probability(
            processed_probability_matrix, 
            post_process_params['threshold']
        )
        
        # Extraire les types de défauts basés sur l'analyse de l'image
        defect_types = self._analyze_defect_types(processed_probability_matrix)
        
        return {
            'image': image,
            'raw_probability_matrix': raw_probability_matrix,
            'probability_matrix': processed_probability_matrix,
            'binary_mask': binary_mask,
            'defect_types': defect_types,
            'original_path': image_path
        }
    
    def _analyze_defect_types(self, probability_matrix):
        """Analyse la matrice de probabilité pour déterminer les types de défauts"""
        defect_types = []
        
        # Analyse simple basée sur les caractéristiques de la matrice
        # Cette logique peut être adaptée selon vos besoins
        
        # Détecter des régions compactes (taches)
        if self._has_compact_regions(probability_matrix):
            defect_types.append("spots")
        
        # Détecter des structures linéaires (rayures)
        if self._has_linear_structures(probability_matrix):
            defect_types.append("scratches")
        
        # Détecter des régions étendues (variations de luminosité)
        if self._has_extended_regions(probability_matrix):
            defect_types.append("brightness_variations")
        
        # Si aucun type spécifique détecté, utiliser "unknown"
        if not defect_types:
            defect_types.append("unknown")
            
        return defect_types
    
    def _has_compact_regions(self, probability_matrix):
        """Détecte la présence de régions compactes"""
        binary = (probability_matrix > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.3 and area > 100:  # Région relativement circulaire
                    return True
        return False
    
    def _has_linear_structures(self, probability_matrix):
        """Détecte la présence de structures linéaires"""
        binary = (probability_matrix > 0.5).astype(np.uint8) * 255
        
        # Utiliser la transformée de Hough pour détecter des lignes
        lines = cv2.HoughLinesP(binary, 1, np.pi/180, threshold=30, minLineLength=30, maxLineGap=10)
        
        return lines is not None and len(lines) > 0
    
    def _has_extended_regions(self, probability_matrix):
        """Détecte la présence de régions étendues"""
        binary = (probability_matrix > 0.3).astype(np.uint8)  # Seuil plus bas
        total_area = np.sum(binary)
        
        # Si une grande portion de l'image est affectée avec une probabilité modérée
        return total_area > (self.width * self.height * 0.1)  # Plus de 10% de l'image

def extract_bounding_boxes_from_mask(binary_mask):
    """Extrait les bounding boxes des régions de défauts"""
    # Trouver les composantes connexes
    labeled_mask = measure.label(binary_mask > 0)
    regions = measure.regionprops(labeled_mask)
    
    bounding_boxes = []
    areas = []
    
    for region in regions:
        # Filtrer les régions trop petites
        if region.area < 50:  # Minimum 50 pixels
            continue
        
        # Extraire bounding box (min_row, min_col, max_row, max_col)
        min_row, min_col, max_row, max_col = region.bbox
        
        # Convertir au format COCO (x, y, width, height)
        x = min_col
        y = min_row
        width = max_col - min_col
        height = max_row - min_row
        
        bounding_boxes.append([x, y, width, height])
        areas.append(width * height)
    
    return bounding_boxes, areas

def generate_industrial_dataset(images_dir: str, output_dir: str, 
                              num_train: int = 200, num_val: int = 40,
                              post_process_params: dict = None):
    """
    Génère un dataset complet de défauts industriels à partir d'images réelles
    
    Args:
        images_dir (str): Répertoire des images sources
        output_dir (str): Répertoire de sortie
        num_train (int): Nombre d'images d'entraînement
        num_val (int): Nombre d'images de validation
        post_process_params (dict): Paramètres de post-traitement du masque
    """
    
    generator = IndustrialDefectGenerator(images_dir, post_process_params=post_process_params)
    
    # Vérifier qu'on a assez d'images
    total_needed = num_train + num_val
    if len(generator.available_images) < total_needed:
        print(f"Attention: Seulement {len(generator.available_images)} images disponibles, "
              f"mais {total_needed} demandées. Les images seront réutilisées.")
    
    # Créer structure de dossiers
    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'masks', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'probability_maps', split), exist_ok=True)
    
    os.makedirs(os.path.join(output_dir, 'annotations'), exist_ok=True)
    
    def generate_split(split_name, num_images):
        print(f"\nGénération du dataset {split_name}...")
        print(f"Traitement de {num_images} images industrielles...")
        
        # Structure COCO
        coco_data = {
            "info": {
                "description": "Industrial Defect Detection Dataset from Real Images",
                "version": "1.0",
                "year": 2025,
                "contributor": "Industrial QC System",
                "date_created": datetime.now().isoformat(),
                "source_images_dir": images_dir
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "Industrial License",
                    "url": ""
                }
            ],
            "categories": [
                {"id": 0, "name": "defect", "supercategory": "industrial_anomaly"}
            ],
            "images": [],
            "annotations": []
        }
        
        annotation_id = 1
        total_defects = 0
        processed_images = 0
        
        # Mélanger les images pour éviter les biais
        available_images = generator.available_images.copy()
        random.shuffle(available_images)
        
        for i in range(1, num_images + 1):
            # Sélectionner une image (avec réutilisation si nécessaire)
            image_idx = (i - 1) % len(available_images)
            source_image_path = available_images[image_idx]
            
            # Traiter l'image réelle avec les paramètres de post-traitement
            result = generator.process_real_image(
                source_image_path, 
                post_process_params=generator.default_post_process_params
            )
            
            if result is None:
                print(f"Erreur lors du traitement de l'image {source_image_path}, passage à la suivante")
                continue
            
            # Noms de fichiers
            img_filename = f"{split_name}_{i:06d}.jpg"
            mask_filename = f"{split_name}_{i:06d}_mask.png"
            prob_filename = f"{split_name}_{i:06d}_prob.npy"
            raw_prob_filename = f"{split_name}_{i:06d}_raw_prob.npy"
            
            # Sauvegarder image
            img_path = os.path.join(output_dir, 'images', split_name, img_filename)
            Image.fromarray(result['image']).save(img_path)
            
            # Sauvegarder masque binaire
            mask_path = os.path.join(output_dir, 'masks', split_name, mask_filename)
            Image.fromarray(result['binary_mask']).save(mask_path)
            
            # Sauvegarder matrice de probabilité post-traitée
            prob_path = os.path.join(output_dir, 'probability_maps', split_name, prob_filename)
            np.save(prob_path, result['probability_matrix'])
            
            # Sauvegarder matrice de probabilité brute (pour comparaison)
            raw_prob_path = os.path.join(output_dir, 'probability_maps', split_name, raw_prob_filename)
            np.save(raw_prob_path, result['raw_probability_matrix'])
            
            # Extraire bounding boxes
            bboxes, areas = extract_bounding_boxes_from_mask(result['binary_mask'])
            
            # Ajouter image aux métadonnées COCO
            coco_data["images"].append({
                "id": i,
                "width": generator.width,
                "height": generator.height,
                "file_name": img_filename,
                "license": 1,
                "date_captured": datetime.now().isoformat(),
                "defect_types": result['defect_types'],
                "source_image": os.path.basename(source_image_path)
            })
            
            # Ajouter annotations
            for bbox, area in zip(bboxes, areas):
                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": i,
                    "category_id": 0,  # "defect"
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                    "segmentation": []  # Pas de segmentation fine pour l'instant
                })
                annotation_id += 1
                total_defects += 1
            
            processed_images += 1
            if processed_images % 20 == 0:
                print(f"Traité {processed_images}/{num_images} images")
        
        # Sauvegarder annotations COCO
        annotations_path = os.path.join(output_dir, 'annotations', f'instances_{split_name}.json')
        with open(annotations_path, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        print(f"Dataset {split_name} généré avec succès!")
        print(f"Images: {output_dir}/images/{split_name}")
        print(f"Annotations: {annotations_path}")
        print(f"Nombre total de défauts: {total_defects}")
        
        return total_defects
    
    # Générer train et validation
    train_defects = generate_split('train', num_train)
    val_defects = generate_split('val', num_val)
    
    print("\n=== DATASET INDUSTRIEL COMPLET ===")
    print(f"Répertoire source: {images_dir}")
    print(f"Répertoire destination: {output_dir}")
    print(f"Images d'entraînement: {num_train} ({train_defects} défauts)")
    print(f"Images de validation: {num_val} ({val_defects} défauts)")
    print(f"Total défauts: {train_defects + val_defects}")
    print("\nStructure:")
    print(f"  {output_dir}/images/train/          # Images d'entraînement")
    print(f"  {output_dir}/images/val/            # Images de validation")
    print(f"  {output_dir}/masks/train/           # Masques binaires")
    print(f"  {output_dir}/probability_maps/train/ # Matrices de probabilité")
    print(f"  {output_dir}/annotations/           # Annotations COCO")

def main():
    parser = argparse.ArgumentParser(description="Générateur de dataset de défauts industriels à partir d'images réelles")
    parser.add_argument("--images-dir", type=str, required=True,
                       help="Répertoire contenant les images sources avec défauts")
    parser.add_argument("--output-dir", type=str, required=True, 
                       help="Répertoire de sortie pour le dataset")
    parser.add_argument("--num-train", type=int, default=200,
                       help="Nombre d'images d'entraînement (défaut: 200)")
    parser.add_argument("--num-val", type=int, default=40,
                       help="Nombre d'images de validation (défaut: 40)")
    
    # Paramètres de post-traitement
    parser.add_argument("--threshold", type=float, default=0.6,
                       help="Seuil de binarisation pour les masques (défaut: 0.5)")
    parser.add_argument("--morph-kernel-size", type=int, default=3,
                       help="Taille du noyau morphologique (défaut: 3)")
    parser.add_argument("--morph-operations", type=str, default="close",
                       choices=["close", "open", "both"],
                       help="Type d'opérations morphologiques (défaut: close)")
    parser.add_argument("--min-area", type=int, default=50,
                       help="Aire minimale des composantes à conserver (défaut: 50)")
    parser.add_argument("--gaussian-blur", type=float, default=0.5,
                       help="Sigma pour le lissage gaussien (0=pas de lissage, défaut: 0.5)")
    parser.add_argument("--apply-crf", action="store_true",
                       help="Appliquer un CRF pour raffiner les contours")
    
    args = parser.parse_args()
    
    # Construire les paramètres de post-traitement
    post_process_params = {
        'threshold': args.threshold,
        'morph_kernel_size': args.morph_kernel_size,
        'morph_operations': args.morph_operations,
        'min_area': args.min_area,
        'gaussian_blur_sigma': args.gaussian_blur,
        'apply_crf': args.apply_crf
    }
    
    print("=== GÉNÉRATEUR DE DATASET INDUSTRIEL (IMAGES RÉELLES) ===")
    print("Traitement d'images réelles avec défauts industriels")
    print("IMPORTANT: La méthode generate_probability_matrix() doit être adaptée")
    print("           avec votre algorithme de détection spécifique")
    print("Format: Images réelles + matrices de probabilité + masques binaires")
    print("\nParamètres de post-traitement:")
    for key, value in post_process_params.items():
        print(f"  {key}: {value}")
    
    # Vérifier que le répertoire d'images existe
    if not os.path.exists(args.images_dir):
        print(f"Erreur: Le répertoire d'images '{args.images_dir}' n'existe pas.")
        return
    
    print(f"\nRépertoire source: {args.images_dir}")
    print(f"Répertoire destination: {args.output_dir}")
    
    generate_industrial_dataset(args.images_dir, args.output_dir, 
                              args.num_train, args.num_val, 
                              post_process_params)

if __name__ == "__main__":
    main()
