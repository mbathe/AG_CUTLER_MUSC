#!/usr/bin/env python3

import os
import sys
import argparse
import torch
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo
import logging

def setup_dataset(dataset_path):
    """Enregistre le dataset custom"""
    # Paths vers les annotations
    train_json = os.path.join(dataset_path, "annotations", "instances_train.json")
    val_json = os.path.join(dataset_path, "annotations", "instances_val.json")
    train_images = os.path.join(dataset_path, "images", "train")
    val_images = os.path.join(dataset_path, "images", "val")
    
    # Enregistrer les datasets
    register_coco_instances("custom_train", {}, train_json, train_images)
    register_coco_instances("custom_val", {}, val_json, val_images)
    
    # Configurer les métadonnées pour une seule classe "defect"
    MetadataCatalog.get("custom_train").thing_classes = ["defect"]
    MetadataCatalog.get("custom_val").thing_classes = ["defect"]
    
    print("Dataset enregistré avec 1 classe: defect")

def setup_config(output_dir, config_file=None, use_gpu=True):
    """Configure detectron2 avec support GPU/CPU automatique"""
    cfg = get_cfg()
    
    if config_file and os.path.exists(config_file):
        cfg.merge_from_file(config_file)
    else:
        # Configuration par défaut pour Faster R-CNN
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    
    # Configuration pour une seule classe "defect"
    cfg.DATASETS.TRAIN = ("custom_train",)
    cfg.DATASETS.TEST = ("custom_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Une seule classe "defect"
    
    # Configuration GPU/CPU automatique
    if use_gpu and torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cuda"
        cfg.SOLVER.IMS_PER_BATCH = 4  # Plus d'images par batch avec GPU
        cfg.SOLVER.BASE_LR = 0.001    # Learning rate plus élevé avec GPU
        print(f"🚀 Utilisation du GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        cfg.MODEL.DEVICE = "cpu"
        cfg.SOLVER.IMS_PER_BATCH = 2  # Moins d'images par batch avec CPU
        cfg.SOLVER.BASE_LR = 0.00025  # Learning rate plus faible avec CPU
        print("💻 Utilisation du CPU (GPU non disponible ou désactivé)")
    
    # Paramètres d'entraînement
    cfg.SOLVER.MAX_ITER = 5000 if use_gpu and torch.cuda.is_available() else 300
    cfg.SOLVER.STEPS = (700, 900) if use_gpu and torch.cuda.is_available() else (210, 250)
    cfg.SOLVER.WARMUP_ITERS = 200 if use_gpu and torch.cuda.is_available() else 100
    cfg.OUTPUT_DIR = output_dir
    
    return cfg

def main():
    parser = argparse.ArgumentParser(description="Entraînement CutLER pour détection de défauts")
    parser.add_argument("--dataset-path", type=str, required=True, help="Chemin vers le dataset")
    parser.add_argument("--output-dir", type=str, default="./output_defect", help="Répertoire de sortie")
    parser.add_argument("--config", type=str, help="Fichier de configuration optionnel")
    parser.add_argument("--cpu-only", action="store_true", help="Forcer l'utilisation du CPU")
    parser.add_argument("--gpu-id", type=int, default=0, help="ID du GPU à utiliser (défaut: 0)")
    
    args = parser.parse_args()
    
    # Configuration GPU
    if not args.cpu_only and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        use_gpu = True
    else:
        use_gpu = False
    
    # Créer le répertoire de sortie
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configuration du logging
    logging.basicConfig(level=logging.INFO)
    
    print("=== ENTRAÎNEMENT CUTLER POUR DÉTECTION DE DÉFAUTS ===")
    print(f"Dataset: {args.dataset_path}")
    print(f"Output: {args.output_dir}")
    print("Objectif: Détecter tous les défauts comme classe unique 'defect'")
    print()
    
    # Enregistrer le dataset
    setup_dataset(args.dataset_path)
    
    # Configurer detectron2
    cfg = setup_config(args.output_dir, args.config, use_gpu)
    
    # Afficher la configuration
    print(f"Configuration:")
    print(f"  Device: {cfg.MODEL.DEVICE}")
    print(f"  Batch size: {cfg.SOLVER.IMS_PER_BATCH}")
    print(f"  Learning rate: {cfg.SOLVER.BASE_LR}")
    print(f"  Max iterations: {cfg.SOLVER.MAX_ITER}")
    print()
    
    # Démarrer l'entraînement
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    
    print("Démarrage de l'entraînement...")
    trainer.train()
    
    print(f"Entraînement terminé! Modèle sauvé dans {args.output_dir}")

if __name__ == "__main__":
    main()
