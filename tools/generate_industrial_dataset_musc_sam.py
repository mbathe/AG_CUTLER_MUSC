#!/usr/bin/env python3

import os
import json
import numpy as np
from PIL import Image
import cv2
from datetime import datetime
import argparse
from skimage import measure, morphology
import glob
import random
import torch
import torch.nn.functional as F
from scipy import ndimage

from MuSc.musc_efficient_tester import MuScEfficientTester

# Import SAM
try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    print("WARNING: SAM non installé. Fonctionnement en mode MuSc seul.")
    print("Installez SAM avec: pip install git+https://github.com/facebookresearch/segment-anything.git")
    SAM_AVAILABLE = False

# 1. Initialisation (une seule fois)
tester = MuScEfficientTester(
    reference_folder="/home/paul/Cours/Stage_ST/code/reference",
    model_name='ViT-L-14-336',
    image_size=(512, 512)
)


try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    print("WARNING: SAM n'est pas installé. Le script fonctionnera en mode MuSc seul.")
    print("Installez SAM avec: pip install git+https://github.com/facebookresearch/segment-anything.git")
    SAM_AVAILABLE = False

# Initialisation de votre classe MuSc
# Assurez-vous que le chemin est correct
tester = MuScEfficientTester(
    reference_folder="/home/paul/Cours/Stage_ST/code/reference",
    model_name='dinov2_vitl14',
    image_size=(504, 504)
)


class AnomalyGuidedSAM:
    """
    Implémentation AG-SAM avec regroupement des défauts et filtrage des faux positifs.
    """

    def __init__(self, sam_checkpoint_path=None, model_type="vit_h", device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.sam_available = SAM_AVAILABLE and sam_checkpoint_path is not None and os.path.exists(
            sam_checkpoint_path)

        if self.sam_available:
            try:
                sam = sam_model_registry[model_type](
                    checkpoint=sam_checkpoint_path)
                sam.to(device=self.device)
                self.predictor = SamPredictor(sam)
                print(f"AG-SAM initialisé avec {model_type} sur {self.device}")
            except Exception as e:
                print(f"Erreur lors de l'initialisation de SAM: {e}")
                self.sam_available = False

        # Paramètres de base
        self.max_iterations = 25
        self.min_region_size = 1000
        self.convergence_threshold = 0.1

        # NOUVEAU: Paramètres pour le regroupement et le filtrage
        # Taille du noyau pour fusionner les défauts proches. Mettre à 0 pour désactiver.
        self.grouping_kernel_size = 2
        # Seuil minimum basé sur le percentile pour éviter les faux positifs.
        self.min_score_percentile = 98.5
        # Le score moyen d'une région doit être X fois > au seuil pour être valide.
        self.min_prompt_score_factor = 1.5

    def adaptive_threshold_otsu(self, scores):
        # MODIFIÉ: Intègre un seuil percentile pour plus de robustesse
        scores_np = scores.copy()
        if scores_np.max() - scores_np.min() < 1e-6:
            return scores_np.mean()

        # Calcul du seuil Otsu sur les valeurs significatives
        # Ignorer le bruit de fond
        scores_positive = scores_np[scores_np > 0.01]
        if len(scores_positive) < 50:
            return np.percentile(scores_np, 99.9)  # Cas limite

        scores_normalized = ((scores_positive - scores_positive.min()) /
                             (scores_positive.max() - scores_positive.min() + 1e-8) * 255).astype(np.uint8)
        threshold_otsu_val, _ = cv2.threshold(
            scores_normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold_otsu = (threshold_otsu_val / 255.0) * \
            (scores_positive.max() - scores_positive.min()) + scores_positive.min()

        # NOUVEAU: Calculer un seuil basé sur un percentile élevé
        threshold_percentile = np.percentile(
            scores_np, self.min_score_percentile)

        # Utiliser le seuil le plus strict (le plus élevé) des deux
        final_threshold = max(threshold_otsu, threshold_percentile)
        return final_threshold

    def extract_prompt_points(self, binary_mask, original_scores, threshold):
        # MODIFIÉ: Intègre le regroupement et la validation des régions

        # Étape 1: Regroupement des défauts proches via une fermeture morphologique
        if self.grouping_kernel_size > 1:
            kernel = np.ones((self.grouping_kernel_size,
                             self.grouping_kernel_size), np.uint8)
            grouped_mask = cv2.morphologyEx(
                binary_mask, cv2.MORPH_CLOSE, kernel)
        else:
            grouped_mask = binary_mask

        # Nettoyage des petits objets après regroupement
        cleaned_mask = morphology.remove_small_objects(
            grouped_mask.astype(bool), min_size=self.min_region_size)

        # Labellisation des régions candidates
        labeled_mask = measure.label(cleaned_mask, connectivity=2)
        regions = measure.regionprops(
            labeled_mask, intensity_image=original_scores)

        prompts = []

        # NOUVEAU: Seuil de validation pour le score moyen d'une région
        min_region_mean_score = threshold * self.min_prompt_score_factor

        for region in regions:
            # Étape 2: Validation de la région basée sur son score MuSc moyen
            if region.mean_intensity >= min_region_mean_score:
                cy, cx = region.centroid
                prompts.append([int(cx), int(cy)])

        return np.array(prompts) if prompts else np.array([])

    # MODIFIÉ: on passe les régions et le masque
    def generate_sam_masks(self, image_rgb, prompt_regions, binary_mask):
        """
        Génère des masques avec une stratégie de multi-prompting (point, boîte, points négatifs).
        """
        if not prompt_regions or not self.sam_available:
            return None

        self.predictor.set_image(image_rgb)
        all_masks = []

        for region in prompt_regions:
            # 1. Prompt positif (centroïde)
            cy, cx = region.centroid
            positive_point = np.array([[int(cx), int(cy)]])

            # 2. Prompt contexte (boîte englobante)
            minr, minc, maxr, maxc = region.bbox
            box = np.array([minc, minr, maxc, maxr])

            # 3. Prompts négatifs (anneau extérieur)
            # On dilate légèrement le masque de la région pour trouver la frontière extérieure
            region_mask = (measure.label(binary_mask) ==
                        region.label).astype(np.uint8)
            dilated_mask = cv2.dilate(region_mask, np.ones(
                (15, 15), np.uint8), iterations=1)
            negative_ring = dilated_mask - region_mask

            # Échantillonner des points dans cet anneau
            negative_points_coords = np.argwhere(negative_ring > 0)
            if len(negative_points_coords) > 10:
                # Échantillonner 10 points négatifs pour éviter la surcharge
                sample_indices = np.random.choice(
                    len(negative_points_coords), size=10, replace=False)
                # Inverser (row, col) -> (x, y)
                negative_points = negative_points_coords[sample_indices][:, ::-1]

                # Combiner les prompts
                prompt_points = np.vstack([positive_point, negative_points])
                # 1 pour positif, 0 pour négatif
                prompt_labels = np.array([1] + [0] * len(negative_points))
            else:
                # Pas assez de points négatifs, on utilise juste le point positif
                prompt_points = positive_point
                prompt_labels = np.array([1])

            # Prédiction avec tous les prompts
            masks, scores, _ = self.predictor.predict(
                point_coords=prompt_points,
                point_labels=prompt_labels,
                box=box,
                multimask_output=True
            )

            all_masks.append(masks[np.argmax(scores)])

        return np.array(all_masks) if all_masks else None

    def score_sam_masks_with_musc(self, sam_masks, musc_scores):
        if sam_masks is None:
            return np.array([])
        mask_scores = []
        for mask in sam_masks:
            mask_sum = mask.sum()
            avg_score = (musc_scores * mask).sum() / \
                mask_sum if mask_sum > 0 else 0
            mask_scores.append(avg_score)
        return np.array(mask_scores)

    def iterative_refinement_musc_sam(self, image_rgb, original_musc_scores):
        if not self.sam_available:
            print(
                "  -> SAM non disponible. Utilisation du seuillage Otsu sur les scores MuSc bruts.")
            threshold = self.adaptive_threshold_otsu(original_musc_scores)
            binary_mask = (original_musc_scores > threshold).astype(np.uint8)
            cleaned_mask = morphology.remove_small_objects(
                binary_mask.astype(bool), min_size=self.min_region_size)
            return np.array([cleaned_mask.astype(np.float32)]) if np.any(cleaned_mask) else None, [], None

        print(f"Début du raffinement par exclusion successive...")
        search_mask = np.ones_like(original_musc_scores, dtype=np.float32)
        all_valid_masks, history = [], []

        for iteration in range(self.max_iterations):
            print(f"AG-SAM Itération {iteration + 1}/{self.max_iterations}")
            masked_scores = original_musc_scores * search_mask

            # MODIFIÉ: Appel à la nouvelle fonction de seuillage
            threshold = self.adaptive_threshold_otsu(masked_scores)
            binary_mask = (masked_scores > threshold).astype(np.uint8)

            # MODIFIÉ: Appel à la nouvelle fonction d'extraction de prompts
            prompt_points, prompt_regions = self.extract_prompt_points(
                binary_mask,
                original_scores=original_musc_scores,
                threshold=threshold
            )

            if len(prompt_points) == 0:
                print("  -> Aucun nouveau prompt valide trouvé, arrêt.")
                break

            sam_masks = self.generate_sam_masks(
                image_rgb, prompt_regions, binary_mask)
            if sam_masks is None:
                print("  -> Aucun masque généré par SAM.")
                search_mask[binary_mask > 0] = 0
                continue

            mask_scores = self.score_sam_masks_with_musc(
                sam_masks, original_musc_scores)
            valid_masks = sam_masks[mask_scores > threshold]

            print(
                f"  -> {len(prompt_points)} prompts -> {len(valid_masks)} masques valides.")

            if len(valid_masks) == 0:
                search_mask[binary_mask > 0] = 0
                continue

            all_valid_masks.extend(valid_masks)
            newly_found_mask = np.any(valid_masks, axis=0).astype(np.float32)
            search_mask *= (1.0 - newly_found_mask)

            if search_mask.sum() / search_mask.size < self.convergence_threshold:
                print(f"  -> Convergence atteinte.")
                break

        return np.array(all_valid_masks) if all_valid_masks else None, history, None

    def create_final_probability_matrix(self, final_masks, original_scores, alpha=0.7):
        if final_masks is None or len(final_masks) == 0:
            return original_scores
        combined_mask = np.any(final_masks, axis=0).astype(np.float32)
        return alpha * combined_mask + (1 - alpha) * original_scores


class IndustrialDefectGenerator:
    def __init__(self, images_dir, image_size=(512, 512), sam_checkpoint_path=None):
        self.images_dir = images_dir
        self.image_size = image_size
        self.width, self.height = image_size
        self.ag_sam = AnomalyGuidedSAM(sam_checkpoint_path)
        self.available_images = self._load_available_images()
        if not self.available_images:
            raise ValueError(
                f"Aucune image trouvée dans le répertoire: {images_dir}")
        print(f"Trouvé {len(self.available_images)} images dans {images_dir}")

    def _load_available_images(self):
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        images = []
        for ext in image_extensions:
            images.extend(glob.glob(os.path.join(self.images_dir, ext)))
            images.extend(glob.glob(os.path.join(
                self.images_dir, '**', ext), recursive=True))
        return sorted(list(set(images)))

    def load_and_resize_image(self, image_path):
        try:
            image = Image.open(image_path).convert('L').resize(
                self.image_size, Image.Resampling.LANCZOS)
            return np.array(image)
        except Exception as e:
            print(f"Erreur lors du chargement de l'image {image_path}: {e}")
            return None

    def load_and_resize_image_rgb(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB').resize(
                self.image_size, Image.Resampling.LANCZOS)
            return np.array(image)
        except Exception as e:
            print(
                f"Erreur lors du chargement de l'image RGB {image_path}: {e}")
            return None

    def generate_probability_matrix(self, image_path):
        probability_matrix, score = tester.get_anomaly_map(image_path)
        return probability_matrix.astype(np.float32)

    def post_process_probability_matrix_with_ag_sam(self, image_path, musc_probability_matrix):
        print(f"Application AG-SAM sur {os.path.basename(image_path)}")
        image_rgb = self.load_and_resize_image_rgb(image_path)
        if image_rgb is None:
            print("Erreur chargement image RGB, utilisation MuSc seul")
            return musc_probability_matrix

        refined_masks, history, _ = self.ag_sam.iterative_refinement_musc_sam(
            image_rgb, musc_probability_matrix)

        if refined_masks is not None:
            final_probability_matrix = self.ag_sam.create_final_probability_matrix(
                refined_masks, musc_probability_matrix)
            print(
                f"  -> AG-SAM terminé: {len(refined_masks)} masques, {len(history)} itérations")
        else:
            final_probability_matrix = musc_probability_matrix
            print(f"  -> Utilisation MuSc original (pas de masques SAM)")
        return final_probability_matrix

    def create_binary_mask_from_probability(self, probability_matrix, threshold=0.5):
        return (probability_matrix > threshold).astype(np.uint8) * 255

    def process_real_image(self, image_path, probability_threshold=0.5):
        image = self.load_and_resize_image(image_path)
        if image is None:
            return None

        print(f"Génération scores MuSc pour {os.path.basename(image_path)}")
        raw_probability_matrix = self.generate_probability_matrix(image_path)
        processed_probability_matrix = self.post_process_probability_matrix_with_ag_sam(
            image_path, raw_probability_matrix)
        binary_mask = self.create_binary_mask_from_probability(
            processed_probability_matrix, probability_threshold)
        defect_types = self._analyze_defect_types(processed_probability_matrix)

        return {'image': image, 'raw_probability_matrix': raw_probability_matrix, 'probability_matrix': processed_probability_matrix,
                'binary_mask': binary_mask, 'defect_types': defect_types, 'original_path': image_path}

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
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
        lines = cv2.HoughLinesP(binary, 1, np.pi/180,
                                threshold=30, minLineLength=30, maxLineGap=10)

        return lines is not None and len(lines) > 0

    def _has_extended_regions(self, probability_matrix):
        """Détecte la présence de régions étendues"""
        binary = (probability_matrix > 0.3).astype(np.uint8)  # Seuil plus bas
        total_area = np.sum(binary)

        # Si une grande portion de l'image est affectée avec une probabilité modérée
        # Plus de 10% de l'image
        return total_area > (self.width * self.height * 0.1)


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
                                sam_checkpoint_path: str = None,
                                probability_threshold: float = 0.5):
    """
    Génère un dataset complet avec l'approche MuSc-SAM intégrée
    
    Args:
        images_dir (str): Répertoire des images sources
        output_dir (str): Répertoire de sortie
        num_train (int): Nombre d'images d'entraînement
        num_val (int): Nombre d'images de validation
        sam_checkpoint_path (str): Chemin vers le checkpoint SAM (optionnel)
        probability_threshold (float): Seuil pour masque binaire
    """

    generator = IndustrialDefectGenerator(
        images_dir, sam_checkpoint_path=sam_checkpoint_path)

    # Vérifier qu'on a assez d'images
    total_needed = num_train + num_val
    if len(generator.available_images) < total_needed:
        print(f"Attention: Seulement {len(generator.available_images)} images disponibles, "
              f"mais {total_needed} demandées. Les images seront réutilisées.")

    # Créer structure de dossiers
    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'masks', split), exist_ok=True)
        os.makedirs(os.path.join(
            output_dir, 'probability_maps', split), exist_ok=True)

    os.makedirs(os.path.join(output_dir, 'annotations'), exist_ok=True)

    def generate_split(split_name, num_images):
        print(f"\n=== Génération du dataset {split_name} avec MuSc-SAM ===")
        print(f"Traitement de {num_images} images industrielles...")

        # Structure COCO
        coco_data = {
            "info": {
                "description": "Industrial Defect Detection Dataset with MuSc-SAM",
                "version": "2.0",
                "year": 2025,
                "contributor": "MuSc-SAM Industrial QC System",
                "date_created": datetime.now().isoformat(),
                "source_images_dir": images_dir,
                "sam_enabled": generator.ag_sam.sam_available
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

            print(
                f"\n[{i}/{num_images}] Traitement: {os.path.basename(source_image_path)}")

            # Traiter l'image avec MuSc-SAM
            result = generator.process_real_image(
                source_image_path,
                probability_threshold=probability_threshold
            )

            if result is None:
                print(
                    f"Erreur lors du traitement de l'image {source_image_path}, passage à la suivante")
                continue

            # Noms de fichiers
            img_filename = f"{split_name}_{i:06d}.jpg"
            mask_filename = f"{split_name}_{i:06d}_mask.png"
            prob_filename = f"{split_name}_{i:06d}_prob.npy"
            raw_prob_filename = f"{split_name}_{i:06d}_raw_prob.npy"

            # Sauvegarder image
            img_path = os.path.join(
                output_dir, 'images', split_name, img_filename)
            Image.fromarray(result['image']).save(img_path)

            # Sauvegarder masque binaire (résultat AG-SAM)
            mask_path = os.path.join(
                output_dir, 'masks', split_name, mask_filename)
            Image.fromarray(result['binary_mask']).save(mask_path)

            # Sauvegarder matrice de probabilité AG-SAM raffinée
            prob_path = os.path.join(
                output_dir, 'probability_maps', split_name, prob_filename)
            np.save(prob_path, result['probability_matrix'])

            # Sauvegarder matrice de probabilité MuSc brute (pour comparaison)
            raw_prob_path = os.path.join(
                output_dir, 'probability_maps', split_name, raw_prob_filename)
            np.save(raw_prob_path, result['raw_probability_matrix'])

            # Extraire bounding boxes
            bboxes, areas = extract_bounding_boxes_from_mask(
                result['binary_mask'])

            # Ajouter image aux métadonnées COCO
            coco_data["images"].append({
                "id": i,
                "width": generator.width,
                "height": generator.height,
                "file_name": img_filename,
                "license": 1,
                "date_captured": datetime.now().isoformat(),
                "defect_types": result['defect_types'],
                "source_image": os.path.basename(source_image_path),
                "processing_method": "MuSc-SAM"
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
            print(f"  -> {len(bboxes)} défauts détectés")

        # Sauvegarder annotations COCO
        annotations_path = os.path.join(
            output_dir, 'annotations', f'instances_{split_name}.json')
        with open(annotations_path, 'w') as f:
            json.dump(coco_data, f, indent=2)

        print(f"\n=== Dataset {split_name} MuSc-SAM généré avec succès! ===")
        print(f"Images: {output_dir}/images/{split_name}")
        print(f"Annotations: {annotations_path}")
        print(f"Nombre total de défauts: {total_defects}")
        print(f"SAM activé: {generator.ag_sam.sam_available}")

        return total_defects

    # Générer train et validation
    train_defects = generate_split('train', num_train)
    val_defects = generate_split('val', num_val)

    print("\n" + "="*60)
    print("DATASET INDUSTRIEL MuSc-SAM COMPLET")
    print("="*60)
    print(f"Répertoire source: {images_dir}")
    print(f"Répertoire destination: {output_dir}")
    print(f"Images d'entraînement: {num_train} ({train_defects} défauts)")
    print(f"Images de validation: {num_val} ({val_defects} défauts)")
    print(f"Total défauts: {train_defects + val_defects}")
    print(f"SAM disponible: {generator.ag_sam.sam_available}")
    print("\nStructure:")
    print(f"  {output_dir}/images/train/          # Images d'entraînement")
    print(f"  {output_dir}/images/val/            # Images de validation")
    print(f"  {output_dir}/masks/train/           # Masques binaires MuSc-SAM")
    print(f"  {output_dir}/probability_maps/train/ # Matrices probabilité raffinées")
    print(f"  {output_dir}/annotations/           # Annotations COCO")


def main():
    parser = argparse.ArgumentParser(
        description="Générateur de dataset MuSc-SAM pour défauts industriels")
    parser.add_argument("--images-dir", type=str, required=True,
                        help="Répertoire contenant les images sources avec défauts")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Répertoire de sortie pour le dataset")
    parser.add_argument("--num-train", type=int, default=200,
                        help="Nombre d'images d'entraînement (défaut: 200)")
    parser.add_argument("--num-val", type=int, default=40,
                        help="Nombre d'images de validation (défaut: 40)")

    # Paramètres MuSc-SAM
    parser.add_argument("--sam-checkpoint", type=str, default=None,
                        help="Chemin vers le checkpoint SAM (ex: sam_vit_h_4b8939.pth)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Seuil de binarisation pour les masques (défaut: 0.5)")

    # Paramètres AG-SAM avancés
    parser.add_argument("--sam-model", type=str, default="vit_h",
                        choices=["vit_h", "vit_l", "vit_b"],
                        help="Type de modèle SAM à utiliser (défaut: vit_h)")
    parser.add_argument("--max-iterations", type=int, default=3,
                        help="Nombre maximum d'itérations AG-SAM (défaut: 3)")
    parser.add_argument("--lambda-refinement", type=float, default=0.5,
                        help="Coefficient lambda pour raffinement itératif (défaut: 0.5)")

    args = parser.parse_args()

    print("=" * 70)
    print("GÉNÉRATEUR DE DATASET INDUSTRIEL AVEC MuSc-SAM")
    print("=" * 70)
    print("Approche hybride: MuSc (scoring sémantique) + SAM (segmentation fine)")
    print("Pipeline: Images réelles → Scores MuSc → AG-SAM → Masques raffinés")
    print("\nCaractéristiques:")
    print("✓ Seuillage adaptatif Otsu")
    print("✓ Prompts automatiques basés centroids")
    print("✓ Boucle itérative co-raffinement")
    print("✓ Validation croisée MuSc-SAM")
    print("✓ Format COCO pour entraînement")

    # Afficher paramètres
    print(f"\nParamètres MuSc-SAM:")
    print(
        f"  SAM checkpoint: {args.sam_checkpoint or 'Non spécifié (mode MuSc seul)'}")
    print(f"  Modèle SAM: {args.sam_model}")
    print(f"  Seuil binarisation: {args.threshold}")
    print(f"  Itérations max: {args.max_iterations}")
    print(f"  Lambda raffinement: {args.lambda_refinement}")

    # Vérifier que le répertoire d'images existe
    if not os.path.exists(args.images_dir):
        print(
            f"\n❌ ERREUR: Le répertoire d'images '{args.images_dir}' n'existe pas.")
        return

    # Vérifier SAM checkpoint si spécifié
    if args.sam_checkpoint and not os.path.exists(args.sam_checkpoint):
        print(
            f"\n⚠️  ATTENTION: Checkpoint SAM '{args.sam_checkpoint}' non trouvé.")
        print("Le système fonctionnera en mode MuSc seul (sans raffinement SAM).")
        args.sam_checkpoint = None

    print(f"\n📁 Répertoire source: {args.images_dir}")
    print(f"📁 Répertoire destination: {args.output_dir}")

    # Télécharger SAM si nécessaire
    if args.sam_checkpoint is None and SAM_AVAILABLE:
        print("\n💡 Pour utiliser SAM, téléchargez un checkpoint depuis:")
        print("   https://github.com/facebookresearch/segment-anything#model-checkpoints")
        print("   Modèles recommandés:")
        print("   - sam_vit_h_4b8939.pth (ViT-H, meilleure qualité)")
        print("   - sam_vit_l_0b3195.pth (ViT-L, bon compromis)")
        print("   - sam_vit_b_01ec64.pth (ViT-B, plus rapide)")

    print(f"\n🚀 Démarrage génération dataset MuSc-SAM...")

    # Modifier les paramètres AG-SAM si spécifiés
    if args.sam_checkpoint:
        # Ces paramètres seront utilisés lors de l'initialisation
        print(f"🔧 Configuration AG-SAM personnalisée activée")

    try:
        generate_industrial_dataset(
            images_dir=args.images_dir,
            output_dir=args.output_dir,
            num_train=args.num_train,
            num_val=args.num_val,
            sam_checkpoint_path=args.sam_checkpoint,
            probability_threshold=args.threshold
        )

        print("\n✅ SUCCÈS: Dataset MuSc-SAM généré avec succès!")
        print(f"📊 Dataset disponible dans: {args.output_dir}")

    except Exception as e:
        print(f"\n❌ ERREUR lors de la génération: {e}")
        import traceback
        traceback.print_exc()


def demo_musc_sam():
    """
    Démonstration rapide de MuSc-SAM sur une image
    """
    print("=== DÉMONSTRATION MuSc-SAM ===")

    # Paramètres de démonstration
    demo_image_path = "/chemin/vers/image/test.jpg"  # À adapter
    sam_checkpoint = "/chemin/vers/sam_vit_h_4b8939.pth"  # À adapter

    if not os.path.exists(demo_image_path):
        print("❌ Image de démo non trouvée. Veuillez spécifier une image valide.")
        return

    try:
        # Initialiser le générateur
        generator = IndustrialDefectGenerator(
            ".", sam_checkpoint_path=sam_checkpoint)

        # Traiter une image
        print(f"🔍 Traitement de: {demo_image_path}")
        result = generator.process_real_image(demo_image_path)

        if result:
            print("✅ Traitement réussi!")
            print(f"   Types de défauts: {result['defect_types']}")
            print(f"   Taille image: {result['image'].shape}")
            print(
                f"   Score anomalie max: {result['probability_matrix'].max():.3f}")

            # Affichage optionnel (nécessite matplotlib)
            try:
                import matplotlib.pyplot as plt

                fig, axes = plt.subplots(1, 4, figsize=(16, 4))

                axes[0].imshow(result['image'], cmap='gray')
                axes[0].set_title('Image Originale')
                axes[0].axis('off')

                axes[1].imshow(result['raw_probability_matrix'], cmap='hot')
                axes[1].set_title('Scores MuSc Bruts')
                axes[1].axis('off')

                axes[2].imshow(result['probability_matrix'], cmap='hot')
                axes[2].set_title('Scores AG-SAM Raffinés')
                axes[2].axis('off')

                axes[3].imshow(result['binary_mask'], cmap='gray')
                axes[3].set_title('Masque Final')
                axes[3].axis('off')

                plt.tight_layout()
                plt.show()

            except ImportError:
                print("Matplotlib non disponible pour l'affichage")

        else:
            print("❌ Échec du traitement")

    except Exception as e:
        print(f"❌ Erreur durant la démo: {e}")


if __name__ == "__main__":
    # Vérifier si on veut la démo ou l'exécution normale
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo_musc_sam()
    else:
        main()
