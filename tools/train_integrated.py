#!/usr/bin/env python3
"""
Script d'entraînement CutLER intégré.
Fait l'enregistrement et l'entraînement dans le même processus.
"""

import sys
import argparse
from pathlib import Path

# Ajouter le répertoire parent au PYTHONPATH
cutler_root = str(Path(__file__).parent.parent)
sys.path.insert(0, cutler_root)

# Importer les modules CutLER après avoir ajusté le path
import os
os.chdir(cutler_root)  # Changer de répertoire pour que les imports relatifs fonctionnent

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json
from detectron2.config import get_cfg
from cutler.config import add_cutler_config

def register_dataset(dataset_path, dataset_name):
    """Enregistrer le dataset dans le catalogue de detectron2"""
    dataset_path = Path(dataset_path)
    
    # Enregistrer le dataset d'entraînement
    train_json = dataset_path / "annotations" / "instances_train.json"
    train_images = dataset_path / "images" / "train"
    
    DatasetCatalog.register(
        f"{dataset_name}_train",
        lambda: load_coco_json(str(train_json), str(train_images))
    )
    
    MetadataCatalog.get(f"{dataset_name}_train").set(
        json_file=str(train_json),
        image_root=str(train_images),
        evaluator_type="coco",
        thing_classes=["rectangle", "circle", "triangle"]
    )
    
    # Enregistrer le dataset de validation
    val_json = dataset_path / "annotations" / "instances_val.json"
    val_images = dataset_path / "images" / "val"
    
    DatasetCatalog.register(
        f"{dataset_name}_val",
        lambda: load_coco_json(str(val_json), str(val_images))
    )
    
    MetadataCatalog.get(f"{dataset_name}_val").set(
        json_file=str(val_json),
        image_root=str(val_images),
        evaluator_type="coco",
        thing_classes=["rectangle", "circle", "triangle"]
    )
    
    return f"{dataset_name}_train", f"{dataset_name}_val"

def setup_config(dataset_path, output_dir, num_classes):
    """Configurer CutLER"""
    cfg = get_cfg()
    add_cutler_config(cfg)
    
    # Configuration de base
    cfg.MODEL.META_ARCHITECTURE = "GeneralizedRCNN"
    cfg.MODEL.BACKBONE.NAME = "build_resnet_fpn_backbone"
    cfg.MODEL.RESNETS.DEPTH = 50
    cfg.MODEL.RESNETS.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
    
    # ROI Heads
    cfg.MODEL.ROI_HEADS.NAME = "CustomStandardROIHeads"
    cfg.MODEL.ROI_HEADS.IN_FEATURES = ["p2", "p3", "p4", "p5"]
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.25
    
    # Masques activés
    cfg.MODEL.MASK_ON = True
    cfg.MODEL.WEIGHTS = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"
    cfg.MODEL.DEVICE = "cpu"
    
    # Datasets - seront mis à jour après enregistrement
    dataset_name = Path(dataset_path).name
    train_name, val_name = register_dataset(dataset_path, dataset_name)
    cfg.DATASETS.TRAIN = (train_name,)
    cfg.DATASETS.TEST = (val_name,)
    
    # Solver
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MOMENTUM = 0.9
    cfg.SOLVER.WEIGHT_DECAY = 0.0001
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.STEPS = (200, 250)
    cfg.SOLVER.MAX_ITER = 300
    cfg.SOLVER.WARMUP_FACTOR = 0.001
    cfg.SOLVER.WARMUP_ITERS = 100
    cfg.SOLVER.WARMUP_METHOD = "linear"
    cfg.SOLVER.CHECKPOINT_PERIOD = 100
    
    # Input
    cfg.INPUT.MIN_SIZE_TRAIN = (640, 800)
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333
    cfg.INPUT.MASK_FORMAT = "polygon"
    cfg.INPUT.FORMAT = "BGR"
    
    # Test
    cfg.TEST.EVAL_PERIOD = 100
    cfg.TEST.DETECTIONS_PER_IMAGE = 100
    
    # Output
    cfg.OUTPUT_DIR = output_dir
    
    return cfg

def train_model(cfg):
    """Entraîner le modèle"""
    from cutler.engine import DefaultTrainer
    
    class CustomTrainer(DefaultTrainer):
        @classmethod
        def build_evaluator(cls, cfg, dataset_name, output_folder=None):
            from cutler.evaluation import COCOEvaluator
            if output_folder is None:
                output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            return COCOEvaluator(dataset_name, cfg, True, output_folder)
    
    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    
    return trainer

def main():
    parser = argparse.ArgumentParser(description="Entraîner CutLER sur un dataset personnalisé")
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-classes", type=int, default=3)
    
    args = parser.parse_args()
    
    print("🚀 Entraînement CutLER intégré")
    print(f"   Dataset: {args.dataset_path}")
    print(f"   Output: {args.output_dir}")
    print(f"   Classes: {args.num_classes}")
    
    # Valider le dataset
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"❌ Dataset non trouvé: {dataset_path}")
        return 1
    
    # Créer le répertoire de sortie
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("🔧 Configuration...")
    try:
        cfg = setup_config(args.dataset_path, args.output_dir, args.num_classes)
        print(f"✓ Configuration créée")
        print(f"  - Train dataset: {cfg.DATASETS.TRAIN}")
        print(f"  - Test dataset: {cfg.DATASETS.TEST}")
    except Exception as e:
        print(f"❌ Erreur de configuration: {e}")
        return 1
    
    print("🎯 Entraînement...")
    try:
        trainer = train_model(cfg)
        print("✅ Entraînement terminé!")
        return 0
    except Exception as e:
        print(f"❌ Erreur d'entraînement: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    import os
    sys.exit(main())
