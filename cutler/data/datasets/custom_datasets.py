# Copyright (c) Meta Platforms, Inc. and affiliates.
# Module pour enregistrer des datasets personnalisés

"""
Module pour l'enregistrement et la gestion de datasets personnalisés au format COCO.
Ce module permet d'ajouter facilement de nouveaux datasets sans modifier le code principal.
"""

import os
from detectron2.data.datasets import register_coco_instances
from detectron2.data.catalog import MetadataCatalog

# Métadonnées pour le dataset personnalisé
CUSTOM_CATEGORIES = [
    {"id": 1, "name": "circle", "supercategory": "shape"},
    {"id": 2, "name": "rectangle", "supercategory": "shape"},
    {"id": 3, "name": "triangle", "supercategory": "shape"}
]

def get_custom_metadata():
    """
    Retourne les métadonnées pour le dataset personnalisé.
    """
    return {
        "thing_classes": [cat["name"] for cat in CUSTOM_CATEGORIES],
        "thing_colors": [
            (255, 0, 0),    # rouge pour circle
            (0, 255, 0),    # vert pour rectangle  
            (0, 0, 255),    # bleu pour triangle
        ],
        "evaluator_type": "coco",
    }

def register_custom_dataset(dataset_name: str, images_dir: str, annotations_file: str):
    """
    Enregistre un dataset personnalisé.
    
    Args:
        dataset_name: Nom du dataset (ex: "custom_train", "custom_val")
        images_dir: Chemin vers le dossier contenant les images
        annotations_file: Chemin vers le fichier d'annotations COCO JSON
    """
    register_coco_instances(
        dataset_name,
        get_custom_metadata(),
        annotations_file,
        images_dir,
    )
    
    # Définir les métadonnées supplémentaires
    MetadataCatalog.get(dataset_name).set(
        **get_custom_metadata()
    )

def register_all_custom_datasets(root_dir: str):
    """
    Enregistre tous les datasets personnalisés trouvés dans le répertoire racine.
    
    Args:
        root_dir: Répertoire racine contenant les datasets personnalisés
    """
    custom_datasets = {}
    
    # Chercher les datasets personnalisés dans le répertoire
    for dataset_folder in os.listdir(root_dir):
        dataset_path = os.path.join(root_dir, dataset_folder)
        if not os.path.isdir(dataset_path):
            continue
            
        images_dir = os.path.join(dataset_path, "images")
        annotations_dir = os.path.join(dataset_path, "annotations")
        
        if not (os.path.exists(images_dir) and os.path.exists(annotations_dir)):
            continue
        
        # Enregistrer les splits trouvés
        for split in ["train", "val", "test"]:
            images_split_dir = os.path.join(images_dir, split)
            annotations_file = os.path.join(annotations_dir, f"instances_{split}.json")
            
            if os.path.exists(images_split_dir) and os.path.exists(annotations_file):
                dataset_name = f"{dataset_folder}_{split}"
                register_custom_dataset(dataset_name, images_split_dir, annotations_file)
                custom_datasets[dataset_name] = (images_split_dir, annotations_file)
                print(f"Dataset enregistré: {dataset_name}")
    
    return custom_datasets

# Fonction pour enregistrer un dataset spécifique par chemin
def register_dataset_from_path(dataset_name: str, dataset_root: str):
    """
    Enregistre un dataset depuis un chemin spécifique.
    
    Args:
        dataset_name: Nom de base du dataset
        dataset_root: Chemin racine du dataset
    """
    registered = []
    
    for split in ["train", "val", "test"]:
        images_dir = os.path.join(dataset_root, "images", split)
        annotations_file = os.path.join(dataset_root, "annotations", f"instances_{split}.json")
        
        if os.path.exists(images_dir) and os.path.exists(annotations_file):
            split_name = f"{dataset_name}_{split}"
            register_custom_dataset(split_name, images_dir, annotations_file)
            registered.append(split_name)
            print(f"Dataset enregistré: {split_name}")
    
    return registered

# Auto-enregistrement des datasets dans le dossier datasets/custom
def auto_register_custom_datasets():
    """
    Enregistre automatiquement tous les datasets personnalisés trouvés.
    """
    datasets_root = os.path.expanduser(os.getenv("DETECTRON2_DATASETS", "datasets"))
    custom_root = os.path.join(datasets_root, "custom")
    
    if os.path.exists(custom_root):
        return register_all_custom_datasets(custom_root)
    else:
        print(f"Répertoire des datasets personnalisés non trouvé: {custom_root}")
        return {}

# Exécuter l'auto-enregistrement au moment de l'import
if __name__ != "__main__":
    auto_register_custom_datasets()
