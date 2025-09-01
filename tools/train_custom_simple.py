#!/usr/bin/env python3
"""
Script d'entraînement simple pour CutLER sur dataset personnalisé.
Cette version évite les problèmes d'imports complexes.
"""

import os
import sys
import argparse
import json
from pathlib import Path

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
    
    # Vérifier que le dataset existe
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"❌ Erreur: Le dataset n'existe pas: {dataset_path}")
        return 1
    
    # Vérifier la structure du dataset
    train_images = dataset_path / "images" / "train"
    val_images = dataset_path / "images" / "val"
    train_annotations = dataset_path / "annotations" / "instances_train.json"
    val_annotations = dataset_path / "annotations" / "instances_val.json"
    
    missing_files = []
    if not train_images.exists():
        missing_files.append(str(train_images))
    if not val_images.exists():
        missing_files.append(str(val_images))
    if not train_annotations.exists():
        missing_files.append(str(train_annotations))
    if not val_annotations.exists():
        missing_files.append(str(val_annotations))
    
    if missing_files:
        print(f"❌ Erreur: Fichiers/dossiers manquants:")
        for file in missing_files:
            print(f"  - {file}")
        return 1
    
    # Valider les annotations JSON
    try:
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
    
    print(f"\n🚀 Lancement de l'entraînement CutLER...")
    
    # Ajouter le répertoire du projet au PYTHONPATH
    project_root = Path(__file__).parent.parent.absolute()
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "cutler"))
    
    try:
        # Essayer d'importer detectron2
        from detectron2.config import get_cfg
        from detectron2.engine import launch
        print("✓ Detectron2 disponible")
        
        # Essayer d'importer les modules CutLER
        from cutler.config import add_cutler_config
        from cutler.engine import DefaultTrainer, default_setup
        from cutler.data.datasets.custom_datasets import register_dataset_from_path
        print("✓ Modules CutLER disponibles")
        
        # Configuration
        cfg = get_cfg()
        add_cutler_config(cfg)
        
        # Charger la configuration de base
        config_file = project_root / "cutler" / "model_zoo" / "configs" / "Custom-Dataset" / "cascade_mask_rcnn_R_50_FPN_custom.yaml"
        if config_file.exists():
            cfg.merge_from_file(str(config_file))
        else:
            print(f"⚠️ Fichier de configuration non trouvé: {config_file}")
            print("Utilisation de la configuration par défaut")
        
        # Enregistrer le dataset
        dataset_name = dataset_path.name
        registered_datasets = register_dataset_from_path(dataset_name, str(dataset_path))
        print(f"✓ Dataset enregistré: {registered_datasets}")
        
        # Configuration du dataset
        train_datasets = [d for d in registered_datasets if d.endswith('_train')]
        val_datasets = [d for d in registered_datasets if d.endswith('_val')]
        
        if train_datasets:
            cfg.DATASETS.TRAIN = tuple(train_datasets)
        if val_datasets:
            cfg.DATASETS.TEST = tuple(val_datasets)
        
        # Configuration du modèle
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_classes
        cfg.OUTPUT_DIR = str(output_path)
        
        # Configuration pour un entraînement rapide (test)
        cfg.SOLVER.MAX_ITER = 300  # Très court pour test
        cfg.SOLVER.STEPS = (200, 250)
        cfg.SOLVER.IMS_PER_BATCH = 2
        cfg.TEST.EVAL_PERIOD = 100
        cfg.SOLVER.CHECKPOINT_PERIOD = 100
        
        # Setup
        default_setup(cfg, args=None)
        
        print(f"📝 Configuration sauvegardée dans: {output_path / 'config.yaml'}")
        
        # Créer le trainer
        class CustomTrainer(DefaultTrainer):
            @classmethod
            def build_evaluator(cls, cfg, dataset_name, output_folder=None):
                from cutler.evaluation import COCOEvaluator
                if output_folder is None:
                    output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
                return COCOEvaluator(dataset_name, cfg, True, output_folder)
        
        trainer = CustomTrainer(cfg)
        trainer.resume_or_load(resume=args.resume)
        
        if args.eval_only:
            print("🔍 Mode évaluation seulement")
            res = trainer.test(cfg, trainer.model)
            return 0
        else:
            print("🎯 Démarrage de l'entraînement...")
            trainer.train()
            print("✅ Entraînement terminé!")
            return 0
            
    except ImportError as e:
        print(f"❌ Erreur d'import: {e}")
        print("\n💡 Solutions possibles:")
        print("1. Installer detectron2:")
        print("   pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html")
        print("2. Ou utiliser l'environnement conda avec detectron2 pré-installé")
        print("3. Vérifier que CUDA est disponible si vous utilisez GPU")
        return 1
    
    except Exception as e:
        print(f"❌ Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
