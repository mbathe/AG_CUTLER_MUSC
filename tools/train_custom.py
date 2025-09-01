#!/usr/bin/env python3
"""
Script pour entraîner CutLER sur un dataset personnalisé.
Ce script configure automatiquement l'entraînement pour votre dataset.
"""

import os
import sys
import argparse
from typing import Dict, List

# Ajouter le chemin du projet au PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "cutler"))

try:
    from cutler.data.datasets.custom_datasets import register_dataset_from_path
    from detectron2.config import get_cfg
    from cutler.config import add_cutler_config
    from cutler.engine import DefaultTrainer, default_setup
    from detectron2.engine import launch
except ImportError as e:
    print(f"Erreur d'import: {e}")
    print("Assurez-vous que detectron2 est installé et que PYTHONPATH est correctement configuré.")
    sys.exit(1)


class CustomDatasetTrainer(DefaultTrainer):
    """
    Trainer personnalisé pour les datasets personnalisés.
    """
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Crée l'évaluateur pour le dataset personnalisé.
        """
        from cutler.evaluation import COCOEvaluator
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        
        return COCOEvaluator(dataset_name, cfg, True, output_folder)


def setup_custom_config(dataset_path: str, output_dir: str, num_classes: int = 3) -> object:
    """
    Configure la configuration pour l'entraînement avec un dataset personnalisé.
    
    Args:
        dataset_path: Chemin vers le dataset personnalisé
        output_dir: Répertoire de sortie pour les résultats
        num_classes: Nombre de classes dans le dataset
    
    Returns:
        Configuration Detectron2
    """
    cfg = get_cfg()
    add_cutler_config(cfg)
    
    # Configuration de base
    cfg.merge_from_file("cutler/model_zoo/configs/CutLER-ImageNet/cascade_mask_rcnn_R_50_FPN_3x.yaml")
    
    # Enregistrer le dataset personnalisé
    dataset_name = os.path.basename(dataset_path.rstrip('/'))
    registered_datasets = register_dataset_from_path(dataset_name, dataset_path)
    
    if not registered_datasets:
        raise ValueError(f"Aucun dataset valide trouvé dans {dataset_path}")
    
    # Configuration du dataset
    train_datasets = [d for d in registered_datasets if d.endswith('_train')]
    val_datasets = [d for d in registered_datasets if d.endswith('_val')]
    
    if train_datasets:
        cfg.DATASETS.TRAIN = tuple(train_datasets)
    if val_datasets:
        cfg.DATASETS.TEST = tuple(val_datasets)
    
    # Configuration du modèle
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.25
    
    # Configuration de l'entraînement
    cfg.SOLVER.IMS_PER_BATCH = 2  # Réduire si problème de mémoire
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 3000  # Ajuster selon vos besoins
    cfg.SOLVER.STEPS = (2000, 2500)
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.WARMUP_ITERS = 100
    cfg.SOLVER.CHECKPOINT_PERIOD = 500
    
    # Configuration de l'évaluation
    cfg.TEST.EVAL_PERIOD = 500
    cfg.TEST.DETECTIONS_PER_IMAGE = 100
    
    # Configuration de sortie
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # Configuration des workers
    cfg.DATALOADER.NUM_WORKERS = 2
    
    # Configuration de la visualisation
    cfg.VIS_PERIOD = 100
    
    return cfg


def train_custom_dataset(
    dataset_path: str,
    output_dir: str,
    num_classes: int = 3,
    resume: bool = False,
    eval_only: bool = False
):
    """
    Lance l'entraînement sur le dataset personnalisé.
    
    Args:
        dataset_path: Chemin vers le dataset personnalisé
        output_dir: Répertoire de sortie
        num_classes: Nombre de classes
        resume: Reprendre l'entraînement
        eval_only: Seulement évaluer
    """
    
    # Configurer
    cfg = setup_custom_config(dataset_path, output_dir, num_classes)
    default_setup(cfg, args=None)
    
    # Créer le trainer
    trainer = CustomDatasetTrainer(cfg)
    trainer.resume_or_load(resume=resume)
    
    if eval_only:
        # Seulement évaluer
        res = trainer.test(cfg, trainer.model)
        return res
    else:
        # Entraîner
        return trainer.train()


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
    
    # Vérifier que le dataset existe
    if not os.path.exists(args.dataset_path):
        raise ValueError(f"Le dataset n'existe pas: {args.dataset_path}")
    
    print(f"Entraînement de CutLER sur le dataset: {args.dataset_path}")
    print(f"Résultats sauvegardés dans: {args.output_dir}")
    print(f"Nombre de classes: {args.num_classes}")
    
    # Lancer l'entraînement
    launch(
        train_custom_dataset,
        args.num_gpus,
        num_machines=1,
        machine_rank=0,
        dist_url="auto",
        args=(args.dataset_path, args.output_dir, args.num_classes, args.resume, args.eval_only),
    )


if __name__ == "__main__":
    main()
