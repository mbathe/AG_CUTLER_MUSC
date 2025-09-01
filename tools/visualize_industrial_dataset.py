#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
import cv2

def visualize_industrial_sample(dataset_path, sample_id=1, split='train'):
    """Visualise un échantillon du dataset industriel"""
    
    # Chemins des fichiers
    img_path = os.path.join(dataset_path, 'images', split, f'{split}_{sample_id:06d}.jpg')
    mask_path = os.path.join(dataset_path, 'masks', split, f'{split}_{sample_id:06d}_mask.png')
    prob_path = os.path.join(dataset_path, 'probability_maps', split, f'{split}_{sample_id:06d}_prob.npy')
    
    # Charger les fichiers
    if not all(os.path.exists(p) for p in [img_path, mask_path, prob_path]):
        print(f"Fichiers manquants pour l'échantillon {sample_id}")
        return
    
    image = np.array(Image.open(img_path))
    mask = np.array(Image.open(mask_path))
    probability_matrix = np.load(prob_path)
    
    # Charger annotations COCO
    annotations_path = os.path.join(dataset_path, 'annotations', f'instances_{split}.json')
    with open(annotations_path, 'r') as f:
        coco_data = json.load(f)
    
    # Trouver les annotations pour cette image
    image_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == sample_id]
    image_info = [img for img in coco_data['images'] if img['id'] == sample_id][0]
    
    # Créer la visualisation
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Échantillon Industriel {sample_id} - Types: {", ".join(image_info.get("defect_types", []))}', fontsize=14)
    
    # Image originale
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Image Industrielle (Niveaux de gris)')
    axes[0, 0].axis('off')
    
    # Matrice de probabilité
    im1 = axes[0, 1].imshow(probability_matrix, cmap='hot', vmin=0, vmax=1)
    axes[0, 1].set_title('Matrice de Probabilité')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    
    # Masque binaire
    axes[0, 2].imshow(mask, cmap='gray')
    axes[0, 2].set_title('Masque Binaire (seuil > 0.5)')
    axes[0, 2].axis('off')
    
    # Image avec bounding boxes
    image_with_boxes = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for ann in image_annotations:
        x, y, w, h = ann['bbox']
        cv2.rectangle(image_with_boxes, (int(x), int(y)), (int(x+w), int(y+h)), (255, 0, 0), 2)
        # Ajouter numéro d'annotation
        cv2.putText(image_with_boxes, f"ID:{ann['id']}", (int(x), int(y-5)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    axes[1, 0].imshow(image_with_boxes)
    axes[1, 0].set_title(f'Bounding Boxes ({len(image_annotations)} défauts)')
    axes[1, 0].axis('off')
    
    # Superposition probabilité + image
    overlay = image.copy()
    overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)
    prob_colored = plt.cm.hot(probability_matrix)[:, :, :3] * 255
    prob_mask = probability_matrix > 0.3  # Seuil de visualisation
    overlay[prob_mask] = 0.7 * overlay[prob_mask] + 0.3 * prob_colored[prob_mask]
    
    axes[1, 1].imshow(overlay.astype(np.uint8))
    axes[1, 1].set_title('Superposition Image + Probabilités')
    axes[1, 1].axis('off')
    
    # Histogramme des probabilités
    axes[1, 2].hist(probability_matrix.flatten(), bins=50, alpha=0.7, color='red')
    axes[1, 2].axvline(x=0.5, color='black', linestyle='--', label='Seuil = 0.5')
    axes[1, 2].set_title('Distribution des Probabilités')
    axes[1, 2].set_xlabel('Probabilité de défaut')
    axes[1, 2].set_ylabel('Nombre de pixels')
    axes[1, 2].legend()
    
    plt.tight_layout()
    
    # Sauvegarder
    output_path = os.path.join(dataset_path, f'visualization_sample_{sample_id}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualisation sauvée: {output_path}")
    
    # Afficher statistiques
    print(f"\n📊 Statistiques de l'échantillon {sample_id}:")
    print(f"  Types de défauts: {', '.join(image_info.get('defect_types', []))}")
    print(f"  Nombre de bounding boxes: {len(image_annotations)}")
    print(f"  Pixels avec probabilité > 0.5: {np.sum(probability_matrix > 0.5)}")
    print(f"  Probabilité moyenne des défauts: {np.mean(probability_matrix[probability_matrix > 0.5]):.3f}")
    print(f"  Taille image: {image.shape}")
    
    for i, ann in enumerate(image_annotations):
        x, y, w, h = ann['bbox']
        print(f"  Défaut {i+1}: bbox=({x:.0f}, {y:.0f}, {w:.0f}, {h:.0f}), aire={ann['area']}")

def analyze_dataset_statistics(dataset_path):
    """Analyse les statistiques globales du dataset"""
    
    stats = {}
    
    for split in ['train', 'val']:
        annotations_path = os.path.join(dataset_path, 'annotations', f'instances_{split}.json')
        
        if not os.path.exists(annotations_path):
            continue
            
        with open(annotations_path, 'r') as f:
            coco_data = json.load(f)
        
        num_images = len(coco_data['images'])
        num_defects = len(coco_data['annotations'])
        
        # Analyser types de défauts
        defect_types_count = {}
        for img in coco_data['images']:
            for defect_type in img.get('defect_types', []):
                defect_types_count[defect_type] = defect_types_count.get(defect_type, 0) + 1
        
        # Analyser tailles des bounding boxes
        areas = [ann['area'] for ann in coco_data['annotations']]
        
        stats[split] = {
            'num_images': num_images,
            'num_defects': num_defects,
            'defects_per_image': num_defects / num_images if num_images > 0 else 0,
            'defect_types': defect_types_count,
            'avg_defect_area': np.mean(areas) if areas else 0,
            'min_defect_area': np.min(areas) if areas else 0,
            'max_defect_area': np.max(areas) if areas else 0
        }
    
    # Afficher statistiques
    print("\n" + "="*60)
    print("📈 STATISTIQUES DU DATASET INDUSTRIEL")
    print("="*60)
    
    for split, data in stats.items():
        print(f"\n{split.upper()}:")
        print(f"  Images: {data['num_images']}")
        print(f"  Défauts totaux: {data['num_defects']}")
        print(f"  Défauts par image: {data['defects_per_image']:.2f}")
        print(f"  Aire moyenne des défauts: {data['avg_defect_area']:.0f} pixels")
        print(f"  Aire min/max: {data['min_defect_area']:.0f} / {data['max_defect_area']:.0f} pixels")
        print(f"  Types de défauts:")
        for defect_type, count in data['defect_types'].items():
            print(f"    - {defect_type}: {count} occurrences")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualisation du dataset industriel")
    parser.add_argument("--dataset-path", type=str, required=True, 
                       help="Chemin vers le dataset industriel")
    parser.add_argument("--sample-id", type=int, default=1,
                       help="ID de l'échantillon à visualiser")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"],
                       help="Split à visualiser")
    parser.add_argument("--stats-only", action="store_true",
                       help="Afficher seulement les statistiques")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset_path):
        print(f"❌ Dataset non trouvé: {args.dataset_path}")
        return
    
    # Analyser statistiques
    analyze_dataset_statistics(args.dataset_path)
    
    # Visualiser échantillon si demandé
    if not args.stats_only:
        print(f"\n🔍 Visualisation de l'échantillon {args.sample_id} ({args.split})")
        visualize_industrial_sample(args.dataset_path, args.sample_id, args.split)

if __name__ == "__main__":
    main()
