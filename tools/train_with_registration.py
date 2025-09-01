#!/usr/bin/env python3
"""
Script d'entraînement CutLER avec enregistrement de dataset.
"""

import sys
import argparse
import json
import subprocess
from pathlib import Path

# Ajouter le répertoire parent au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

def register_dataset(dataset_path, dataset_name):
    """Enregistrer le dataset dans le catalogue de detectron2"""
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.data.datasets import load_coco_json
    
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
        thing_classes=["rectangle", "circle", "triangle"]  # Nos 3 classes
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

def main():
    parser = argparse.ArgumentParser(description="Entraîner CutLER sur un dataset personnalisé")
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-classes", type=int, default=3)
    
    args = parser.parse_args()
    
    print("🚀 Entraînement CutLER")
    print(f"   Dataset: {args.dataset_path}")
    print(f"   Output: {args.output_dir}")
    print(f"   Classes: {args.num_classes}")
    
    # Valider le dataset
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"❌ Dataset non trouvé: {dataset_path}")
        return 1
    
    print("🔧 Enregistrement du dataset...")
    try:
        dataset_name = dataset_path.name
        train_name, val_name = register_dataset(dataset_path, dataset_name)
        print(f"✓ Dataset enregistré:")
        print(f"  - Train: {train_name}")
        print(f"  - Val: {val_name}")
    except Exception as e:
        print(f"❌ Erreur d'enregistrement: {e}")
        return 1
    
    # Configuration
    config_file = Path(__file__).parent / "standalone_config.yaml"
    
    # Arguments d'entraînement
    train_args = [
        sys.executable, "cutler/train_net.py",
        "--config-file", str(config_file),
        "--num-gpus", "1",
        "DATASETS.TRAIN", f"('{train_name}',)",
        "DATASETS.TEST", f"('{val_name}',)",
        "OUTPUT_DIR", args.output_dir,
        "MODEL.ROI_HEADS.NUM_CLASSES", str(args.num_classes)
    ]
    
    print("📝 Commande:")
    print("  " + " ".join(train_args))
    
    print("🎯 Lancement...")
    result = subprocess.run(train_args, cwd=str(Path(__file__).parent.parent))
    
    if result.returncode == 0:
        print("✅ Entraînement terminé!")
    else:
        print(f"❌ Erreur (code: {result.returncode})")
    
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())
