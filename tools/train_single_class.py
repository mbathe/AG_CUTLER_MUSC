#!/usr/bin/env python3

import os
import sys
import argparse
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
    
    # Configurer les métadonnées pour une seule classe "object"
    MetadataCatalog.get("custom_train").thing_classes = ["object"]
    MetadataCatalog.get("custom_val").thing_classes = ["object"]
    
    print("Dataset enregistré avec 1 classe: object")

def setup_config(output_dir, config_file=None):
    """Configure detectron2"""
    cfg = get_cfg()
    
    if config_file and os.path.exists(config_file):
        cfg.merge_from_file(config_file)
    else:
        # Configuration par défaut pour Faster R-CNN
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    
    # Configuration pour une seule classe
    cfg.DATASETS.TRAIN = ("custom_train",)
    cfg.DATASETS.TEST = ("custom_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Une seule classe "object"
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 300
    cfg.SOLVER.STEPS = (210, 250)
    cfg.SOLVER.WARMUP_ITERS = 100
    cfg.MODEL.DEVICE = "cpu"  # Utiliser CPU
    cfg.OUTPUT_DIR = output_dir
    
    return cfg

def main():
    parser = argparse.ArgumentParser(description="Entraînement CutLER-style: détection sans classification")
    parser.add_argument("--dataset-path", type=str, required=True, help="Chemin vers le dataset")
    parser.add_argument("--output-dir", type=str, default="./output_single_class", help="Répertoire de sortie")
    parser.add_argument("--config", type=str, help="Fichier de configuration optionnel")
    
    args = parser.parse_args()
    
    # Créer le répertoire de sortie
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configuration du logging
    logging.basicConfig(level=logging.INFO)
    
    print("=== ENTRAÎNEMENT CUTLER-STYLE: DÉTECTION SANS CLASSIFICATION ===")
    print(f"Dataset: {args.dataset_path}")
    print(f"Output: {args.output_dir}")
    print("Objectif: Détecter tous les objets comme classe unique 'object'")
    
    # Enregistrer le dataset
    setup_dataset(args.dataset_path)
    
    # Configurer detectron2
    cfg = setup_config(args.output_dir, args.config)
    
    # Démarrer l'entraînement
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    
    print("Démarrage de l'entraînement...")
    trainer.train()
    
    print(f"Entraînement terminé! Modèle sauvé dans {args.output_dir}")

if __name__ == "__main__":
    main()
