#!/usr/bin/env python3
"""
Script d'entraînement pour CutLER sur dataset personnalisé.
Basé sur train_net.py original de CutLER.
"""

import sys
import argparse
import json
from pathlib import Path

# Ajouter le répertoire parent au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

def setup_environment():
    """Configure l'environnement pour CutLER"""
    # Importer les modules nécessaires
    try:
        # Import detectron2
        import detectron2
        from detectron2.config import get_cfg
        from detectron2.engine import launch
        from detectron2.utils.logger import setup_logger
        
        # Import CutLER modules
        from cutler.config import add_cutler_config
        from cutler.data.datasets.custom_datasets import register_dataset_from_path
        
        return True
    except ImportError as e:
        print(f"❌ Erreur d'import: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Entraîner CutLER sur un dataset personnalisé")
    parser.add_argument("--dataset-path", type=str, required=True,
                       help="Chemin vers le dataset personnalisé")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Répertoire de sortie pour les résultats")
    parser.add_argument("--num-classes", type=int, default=3,
                       help="Nombre de classes dans le dataset")
    parser.add_argument("--resume", action="store_true",
                       help="Reprendre l'entraînement depuis le dernier checkpoint")
    parser.add_argument("--eval-only", action="store_true",
                       help="Seulement évaluer le modèle")
    parser.add_argument("--num-gpus", type=int, default=1,
                       help="Nombre de GPUs à utiliser")
    
    args = parser.parse_args()
    
    print(f"Configuration de l'entraînement:")
    print(f"  Dataset: {args.dataset_path}")
    print(f"  Sortie: {args.output_dir}")
    print(f"  Classes: {args.num_classes}")
    print(f"  GPUs: {args.num_gpus}")
    
    # Vérifier l'environnement
    if not setup_environment():
        print("❌ Impossible d'importer les modules nécessaires")
        return 1
    
    # Valider les chemins
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"❌ Erreur: Dataset non trouvé: {dataset_path}")
        return 1
    
    # Vérifier la structure du dataset
    required_files = [
        dataset_path / "images" / "train",
        dataset_path / "images" / "val", 
        dataset_path / "annotations" / "instances_train.json",
        dataset_path / "annotations" / "instances_val.json"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not file_path.exists():
            missing_files.append(str(file_path))
    
    if missing_files:
        print(f"❌ Erreur: Fichiers/dossiers manquants:")
        for file in missing_files:
            print(f"  - {file}")
        return 1
    
    # Valider les annotations JSON
    try:
        train_annotations = dataset_path / "annotations" / "instances_train.json"
        val_annotations = dataset_path / "annotations" / "instances_val.json"
        
        with open(train_annotations) as f:
            train_data = json.load(f)
        with open(val_annotations) as f:
            val_data = json.load(f)
            
        print(f"✓ Annotations valides")
        print(f"  - Images d'entraînement: {len(train_data['images'])}")
        print(f"  - Annotations d'entraînement: {len(train_data['annotations'])}")
        print(f"  - Images de validation: {len(val_data['images'])}")
        print(f"  - Annotations de validation: {len(val_data['annotations'])}")
    except json.JSONDecodeError as e:
        print(f"❌ Erreur dans les annotations JSON: {e}")
        return 1
    
    # Créer le répertoire de sortie
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Lancer l'entraînement via le train_net.py original
    print("🚀 Lancement de l'entraînement via train_net.py...")
    
    # Créer un fichier de configuration temporaire
    config_content = f"""_BASE_: ../Base-CutLER.yaml

MODEL:
  ROI_HEADS:
    NUM_CLASSES: {args.num_classes}

DATASETS:
  TRAIN: ("custom_train",)
  TEST: ("custom_val",)

OUTPUT_DIR: "{args.output_dir}"

SOLVER:
  MAX_ITER: 300
  STEPS: (200, 250)
  IMS_PER_BATCH: 2
  CHECKPOINT_PERIOD: 100

TEST:
  EVAL_PERIOD: 100
"""
    
    config_file = output_path / "custom_config.yaml"
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    # Importer et utiliser les fonctions de train_net.py
    from cutler.data.datasets.custom_datasets import register_dataset_from_path
    
    # Enregistrer le dataset
    dataset_name = dataset_path.name
    registered_datasets = register_dataset_from_path(dataset_name, str(dataset_path))
    print(f"✓ Dataset enregistré: {registered_datasets}")
    
    # Utiliser le train_net.py existant via subprocess
    import subprocess
    
    # Créer les arguments pour train_net.py
    train_args = [
        sys.executable, "cutler/train_net.py",
        "--config-file", str(config_file),
        "--num-gpus", str(args.num_gpus),
        "OUTPUT_DIR", args.output_dir
    ]
    
    if args.resume:
        train_args.append("--resume")
    
    if args.eval_only:
        train_args.append("--eval-only")
    
    print("📝 Configuration:")
    print(f"  - Commande: {' '.join(train_args)}")
    
    print("🎯 Démarrage de l'entraînement...")
    result = subprocess.run(train_args, cwd=str(Path(__file__).parent.parent))
    
    if result.returncode == 0:
        print("✅ Entraînement terminé!")
    else:
        print("❌ Erreur pendant l'entraînement")
    
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())
