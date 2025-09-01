#!/usr/bin/env python3
"""
Script d'entraînement simple pour CutLER sur dataset personnalisé.
"""

import sys
import argparse
import json
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Entraîner CutLER sur un dataset personnalisé")
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-classes", type=int, default=3)
    
    args = parser.parse_args()
    
    print(f"🚀 Entraînement CutLER")
    print(f"   Dataset: {args.dataset_path}")
    print(f"   Output: {args.output_dir}")
    print(f"   Classes: {args.num_classes}")
    
    # Valider le dataset
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"❌ Dataset non trouvé: {dataset_path}")
        return 1
    
    # Enregistrer le dataset
    print("🔧 Enregistrement du dataset...")
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    try:
        from cutler.data.datasets.custom_datasets import register_dataset_from_path
        dataset_name = dataset_path.name
        registered_datasets = register_dataset_from_path(dataset_name, str(dataset_path))
        print(f"✓ Dataset enregistré: {registered_datasets}")
    except Exception as e:
        print(f"⚠️ Erreur d'enregistrement: {e}")
        print("Continuons quand même...")
    
    # Configuration autonome
    config_file = Path(__file__).parent / "standalone_config.yaml"
    
    # Arguments d'entraînement
    train_args = [
        sys.executable, "cutler/train_net.py",
        "--config-file", str(config_file),
        "--num-gpus", "1",
        f"DATASETS.TRAIN", f"('{dataset_path.name}_train',)",
        f"DATASETS.TEST", f"('{dataset_path.name}_val',)",
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
