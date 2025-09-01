#!/usr/bin/env python3

import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2 import model_zoo
import matplotlib.pyplot as plt
from PIL import Image


def parse_xml_annotation(xml_path):
    """Parse une annotation XML et retourne les bounding boxes ground truth"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    boxes = []
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)

        boxes.append([xmin, ymin, xmax, ymax])

    return np.array(boxes)


def calculate_iou(box1, box2):
    """Calcule l'IoU entre deux bounding boxes"""
    # Coordonnées de l'intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Aire de l'intersection
    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)

    # Aires des boîtes
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Union
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def calculate_ap_at_iou_threshold(predictions, ground_truths, iou_threshold=0.5):
    """Calcule l'AP à un seuil IoU donné"""
    if len(predictions) == 0:
        return 0.0 if len(ground_truths) > 0 else 1.0

    if len(ground_truths) == 0:
        return 0.0

    # Trier les prédictions par score décroissant
    sorted_indices = np.argsort([-p['score'] for p in predictions])
    sorted_predictions = [predictions[i] for i in sorted_indices]

    # Variables pour calcul AP
    tp = np.zeros(len(sorted_predictions))
    fp = np.zeros(len(sorted_predictions))
    used_gt = np.zeros(len(ground_truths), dtype=bool)

    # Pour chaque prédiction
    for i, pred in enumerate(sorted_predictions):
        pred_box = pred['bbox']
        best_iou = 0.0
        best_gt_idx = -1

        # Trouver la GT avec la meilleure IoU
        for j, gt_box in enumerate(ground_truths):
            if used_gt[j]:
                continue

            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j

        # Déterminer si c'est un TP ou FP
        if best_iou >= iou_threshold and best_gt_idx != -1:
            tp[i] = 1
            used_gt[best_gt_idx] = True
        else:
            fp[i] = 1

    # Calcul précision et recall cumulés
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    recalls = tp_cumsum / len(ground_truths)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

    # Calcul AP (aire sous la courbe PR)
    # Méthode 11-point interpolation
    ap = 0.0
    for threshold in np.arange(0, 1.1, 0.1):
        if np.sum(recalls >= threshold) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= threshold])
        ap += p / 11.0

    return ap


def test_cutler_with_map():
    """Test du modèle CutLER avec calcul du mAP@0.5"""
    model_path = "./output_defect_gpu/model_final.pth"
    validation_dir = "/home/paul/Cours/Stage_ST/code/dataset/NEU-DET/validation"
    images_dir = os.path.join(validation_dir, "images")
    annotations_dir = os.path.join(validation_dir, "annotations")
    output_dir = "./cutler_validation_results"

    print("=== TEST CUTLER AVEC ÉVALUATION mAP@0.5 ===")
    print(f"Modèle: {model_path}")
    print(f"Dataset validation: {validation_dir}")
    print("Métrique: mAP@0.5 (détection générique)")
    print()

    if not os.path.exists(model_path):
        print(f"❌ Modèle non trouvé: {model_path}")
        return

    if not os.path.exists(images_dir):
        print(f"❌ Dossier images non trouvé: {images_dir}")
        return

    if not os.path.exists(annotations_dir):
        print(f"❌ Dossier annotations non trouvé: {annotations_dir}")
        return

    # Configuration CutLER
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Détection générique
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.DEVICE = "cpu"
    # Seuil plus bas pour plus de détections
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3

    # Métadonnées
    MetadataCatalog.get("validation_test").thing_classes = ["defect"]

    # Prédicteur
    predictor = DefaultPredictor(cfg)
    print("✅ Prédicteur CutLER configuré")

    # Créer dossier résultats
    os.makedirs(output_dir, exist_ok=True)

    # Parcourir les sous-dossiers de catégories
    all_predictions = []
    all_ground_truths = []
    category_stats = {}

    categories = [d for d in os.listdir(
        images_dir) if os.path.isdir(os.path.join(images_dir, d))]
    print(f"Catégories trouvées: {categories}")
    print()

    for category in categories:
        print(f"=== CATÉGORIE: {category.upper()} ===")
        category_path = os.path.join(images_dir, category)
        image_files = [f for f in os.listdir(
            category_path) if f.endswith('.jpg')]

        category_predictions = []
        category_ground_truths = []

        for img_file in image_files:
            # Chemins des fichiers
            img_path = os.path.join(category_path, img_file)
            xml_file = img_file.replace('.jpg', '.xml')
            xml_path = os.path.join(annotations_dir, xml_file)

            if not os.path.exists(xml_path):
                print(f"⚠️  Annotation manquante pour {img_file}")
                continue

            # Charger image
            img_gray = np.array(Image.open(img_path))
            if len(img_gray.shape) == 2:  # Image en niveaux de gris
                img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
            else:
                img_rgb = img_gray

            # Charger annotations ground truth
            gt_boxes = parse_xml_annotation(xml_path)

            # Prédiction CutLER
            outputs = predictor(img_rgb)
            instances = outputs["instances"]

            # Convertir prédictions au format standard
            image_predictions = []
            if len(instances) > 0:
                pred_boxes = instances.pred_boxes.tensor.cpu().numpy()
                scores = instances.scores.cpu().numpy()

                for box, score in zip(pred_boxes, scores):
                    image_predictions.append({
                        'bbox': box,
                        'score': score
                    })

            # Stocker pour calcul global
            category_predictions.append(image_predictions)
            category_ground_truths.append(gt_boxes)
            all_predictions.append(image_predictions)
            all_ground_truths.append(gt_boxes)

            print(
                f"  {img_file}: {len(image_predictions)} détections, {len(gt_boxes)} GT")

        # Calculer AP pour cette catégorie
        flat_pred = [p for img_pred in category_predictions for p in img_pred]
        flat_gt = [gt for gt_list in category_ground_truths for gt in gt_list]

        if len(flat_gt) > 0:
            category_ap = calculate_ap_at_iou_threshold(
                flat_pred, flat_gt, 0.5)
            category_stats[category] = {
                'ap': category_ap,
                'num_images': len(image_files),
                'total_predictions': len(flat_pred),
                'total_gt': len(flat_gt)
            }
            print(f"  AP@0.5 pour {category}: {category_ap:.3f}")
        else:
            category_stats[category] = {
                'ap': 0.0,
                'num_images': len(image_files),
                'total_predictions': 0,
                'total_gt': 0
            }
            print(f"  Pas de GT pour {category}")
        print()

    # Calcul mAP@0.5 global
    flat_all_pred = [p for img_pred in all_predictions for p in img_pred]
    flat_all_gt = [gt for gt_list in all_ground_truths for gt in gt_list]

    if len(flat_all_gt) > 0:
        map_50 = calculate_ap_at_iou_threshold(flat_all_pred, flat_all_gt, 0.5)
    else:
        map_50 = 0.0

    # Affichage des résultats
    print("📊 RÉSULTATS FINAUX:")
    print(f"mAP@0.5 GLOBAL: {map_50:.3f}")
    print()

    print("Détail par catégorie:")
    for category, stats in category_stats.items():
        print(f"  {category:15s}: AP={stats['ap']:.3f} | "
              f"Images={stats['num_images']:3d} | "
              f"Détections={stats['total_predictions']:3d} | "
              f"GT={stats['total_gt']:3d}")

    print()
    print(
        f"Total images testées: {sum(stats['num_images'] for stats in category_stats.values())}")
    print(f"Total détections: {len(flat_all_pred)}")
    print(f"Total ground truth: {len(flat_all_gt)}")

    # Créer visualisation comparative pour quelques exemples
    print("\n🎨 Génération d'exemples visuels...")

    examples_per_category = 2
    for category in categories[:3]:  # Limiter à 3 catégories pour l'exemple
        category_path = os.path.join(images_dir, category)
        image_files = [f for f in os.listdir(
            category_path) if f.endswith('.jpg')][:examples_per_category]

        for img_file in image_files:
            img_path = os.path.join(category_path, img_file)
            xml_file = img_file.replace('.jpg', '.xml')
            xml_path = os.path.join(annotations_dir, xml_file)

            if not os.path.exists(xml_path):
                continue

            # Charger image et annotations
            img_gray = np.array(Image.open(img_path))
            if len(img_gray.shape) == 2:
                img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
            else:
                img_rgb = img_gray

            gt_boxes = parse_xml_annotation(xml_path)
            outputs = predictor(img_rgb)

            # Créer visualisation
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(f'{category}: {img_file}', fontsize=14)

            # Image originale
            display_img = img_gray if len(img_gray.shape) == 2 else img_rgb
            cmap = 'gray' if len(img_gray.shape) == 2 else None
            axes[0].imshow(display_img, cmap=cmap)
            axes[0].set_title('Image Originale')
            axes[0].axis('off')

            # Ground Truth
            gt_img = img_rgb.copy()
            for box in gt_boxes:
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(gt_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            axes[1].imshow(gt_img)
            axes[1].set_title(f'Ground Truth ({len(gt_boxes)} objets)')
            axes[1].axis('off')

            # Prédictions CutLER
            v = Visualizer(img_rgb, MetadataCatalog.get(
                "validation_test"), scale=1.0)
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            result_img = out.get_image()

            axes[2].imshow(result_img)
            instances = outputs["instances"]
            axes[2].set_title(f'CutLER ({len(instances)} détections)')
            axes[2].axis('off')

            plt.tight_layout()

            # Sauvegarder
            output_path = os.path.join(
                output_dir, f"{category}_{img_file.replace('.jpg', '.png')}")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()

    # Sauvegarder résultats détaillés
    results_file = os.path.join(output_dir, "evaluation_results.txt")
    with open(results_file, 'w') as f:
        f.write("=== ÉVALUATION CUTLER SUR DATASET VALIDATION ===\n\n")
        f.write(f"mAP@0.5 GLOBAL: {map_50:.3f}\n\n")
        f.write("Détail par catégorie:\n")
        for category, stats in category_stats.items():
            f.write(f"{category:15s}: AP={stats['ap']:.3f} | "
                    f"Images={stats['num_images']:3d} | "
                    f"Détections={stats['total_predictions']:3d} | "
                    f"GT={stats['total_gt']:3d}\n")
        f.write(
            f"\nTotal images testées: {sum(stats['num_images'] for stats in category_stats.values())}\n")
        f.write(f"Total détections: {len(flat_all_pred)}\n")
        f.write(f"Total ground truth: {len(flat_all_gt)}\n")

    print(f"✅ Résultats sauvés dans: {output_dir}")
    print(f"📄 Rapport détaillé: {results_file}")

    return map_50, category_stats


def analyze_detection_performance(predictions, ground_truths, iou_threshold=0.5):
    """Analyse détaillée des performances de détection"""
    if len(predictions) == 0 or len(ground_truths) == 0:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'tp': 0,
            'fp': len(predictions),
            'fn': len(ground_truths)
        }

    # Calculer matches
    used_gt = np.zeros(len(ground_truths), dtype=bool)
    tp = 0

    for pred in predictions:
        best_iou = 0.0
        best_gt_idx = -1

        for j, gt_box in enumerate(ground_truths):
            if used_gt[j]:
                continue

            iou = calculate_iou(pred['bbox'], gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j

        if best_iou >= iou_threshold and best_gt_idx != -1:
            tp += 1
            used_gt[best_gt_idx] = True

    fp = len(predictions) - tp
    fn = len(ground_truths) - tp

    precision = tp / len(predictions) if len(predictions) > 0 else 0.0
    recall = tp / len(ground_truths) if len(ground_truths) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision +
                                     recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


if __name__ == "__main__":
    map_score, category_results = test_cutler_with_map()

    print("\n💡 INTERPRÉTATION DES RÉSULTATS:")
    print(f"• mAP@0.5 = {map_score:.3f}")
    if map_score > 0.5:
        print("  → Excellente performance de détection")
    elif map_score > 0.3:
        print("  → Bonne performance de détection")
    elif map_score > 0.1:
        print("  → Performance modérée, améliorations possibles")
    else:
        print("  → Performance faible, révision du modèle recommandée")

    print("\n🔧 RECOMMANDATIONS:")
    print("• Si mAP faible: ajuster le seuil de score ou réentraîner")
    print("• Analyser les catégories avec AP faible pour cibler les améliorations")
    print("• Considérer l'augmentation de données pour les catégories difficiles")
