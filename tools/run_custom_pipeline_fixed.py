#!/usr/bin/env python3
"""
Script principal pour entraîner CutLER sur votre dataset personnalisé.
Ce script guide l'utilisateur à travers toutes les étapes nécessaires.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def print_banner():
    """Affiche une bannière d'accueil."""
    print("=" * 80)
    print("   ENTRAÎNEMENT CUTLER SUR DATASET PERSONNALISÉ")
    print("=" * 80)
    print()

def check_requirements():
    """Vérifie que les dépendances nécessaires sont installées."""
    print("Vérification des dépendances...")
    
    required_packages = ['numpy', 'opencv-python', 'matplotlib', 'Pillow']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_').replace('opencv_python', 'cv2').replace('Pillow', 'PIL'))
            print(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package}")
    
    if missing_packages:
        print(f"\nPaquets manquants: {', '.join(missing_packages)}")
        print("Veuillez les installer avec: pip install " + " ".join(missing_packages))
        return False
    
    print("✓ Toutes les dépendances sont installées.")
    return True

def generate_dataset(output_dir: str, num_train: int = 100, num_val: int = 20):
    """Génère un dataset d'exemple."""
    print("\nGénération du dataset d'exemple...")
    print(f"Images d'entraînement: {num_train}")
    print(f"Images de validation: {num_val}")
    print(f"Dossier de sortie: {output_dir}")
    
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "generate_custom_dataset.py"))
    output_dir = os.path.abspath(output_dir)
    
    cmd = [
        sys.executable, script_path,
        "--output-dir", output_dir,
        "--num-train", str(num_train),
        "--num-val", str(num_val)
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("✓ Dataset généré avec succès!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Erreur lors de la génération du dataset: {e}")
        return False

def train_model(dataset_path: str, output_dir: str, num_classes: int = 3):
    """Entraîne le modèle."""
    print("\nEntraînement du modèle...")
    print(f"Dataset: {dataset_path}")
    print(f"Classes: {num_classes}")
    print(f"Sortie: {output_dir}")
    
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "train_custom.py"))
    dataset_path = os.path.abspath(dataset_path)
    output_dir = os.path.abspath(output_dir)
    
    cmd = [
        sys.executable, script_path,
        "--dataset-path", dataset_path,
        "--output-dir", output_dir,
        "--num-classes", str(num_classes)
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("✓ Entraînement terminé avec succès!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Erreur lors de l'entraînement: {e}")
        return False

def visualize_results(dataset_path: str, output_dir: str, model_config: str = None, model_weights: str = None):
    """Visualise les résultats."""
    print("\nVisualisation des résultats...")
    
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "visualize_custom.py"))
    dataset_path = os.path.abspath(dataset_path)
    output_dir = os.path.abspath(output_dir)
    
    # Visualiser les annotations ground truth
    gt_cmd = [
        sys.executable, script_path,
        "--dataset-path", dataset_path,
        "--output-dir", os.path.join(output_dir, "ground_truth"),
        "--mode", "gt",
        "--num-images", "10"
    ]
    
    try:
        subprocess.run(gt_cmd, check=True)
        print("✓ Visualisation ground truth créée!")
    except subprocess.CalledProcessError as e:
        print(f"✗ Erreur lors de la visualisation GT: {e}")
        return False
    
    # Visualiser les prédictions si le modèle est disponible
    if model_config and model_weights and os.path.exists(model_weights):
        pred_cmd = [
            sys.executable, script_path,
            "--dataset-path", dataset_path,
            "--output-dir", os.path.join(output_dir, "predictions"),
            "--mode", "compare",
            "--model-config", model_config,
            "--model-weights", model_weights,
            "--num-images", "5"
        ]
        
        try:
            subprocess.run(pred_cmd, check=True)
            print("✓ Visualisation des prédictions créée!")
        except subprocess.CalledProcessError as e:
            print(f"✗ Erreur lors de la visualisation des prédictions: {e}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Script principal pour CutLER dataset personnalisé")
    parser.add_argument("--workspace", type=str, default="./workspace_custom",
                       help="Répertoire de travail pour tous les fichiers")
    parser.add_argument("--num-train", type=int, default=100,
                       help="Nombre d'images d'entraînement à générer")
    parser.add_argument("--num-val", type=int, default=20,
                       help="Nombre d'images de validation à générer")
    parser.add_argument("--num-classes", type=int, default=3,
                       help="Nombre de classes dans le dataset")
    parser.add_argument("--skip-generation", action="store_true",
                       help="Ignorer la génération du dataset")
    parser.add_argument("--skip-training", action="store_true",
                       help="Ignorer l'entraînement")
    parser.add_argument("--skip-visualization", action="store_true",
                       help="Ignorer la visualisation")
    
    args = parser.parse_args()
    
    print_banner()
    
    # Vérifier les dépendances
    if not check_requirements():
        return 1
    
    # Créer le répertoire de travail
    workspace = Path(args.workspace)
    workspace.mkdir(exist_ok=True)
    
    dataset_dir = workspace / "dataset"
    output_dir = workspace / "training_output"
    visualization_dir = workspace / "visualizations"
    
    print(f"\nRépertoire de travail: {workspace.absolute()}")
    
    # Étape 1: Génération du dataset
    if not args.skip_generation:
        if not generate_dataset(str(dataset_dir), args.num_train, args.num_val):
            return 1
    else:
        print("Génération du dataset ignorée.")
        if not dataset_dir.exists():
            print(f"Erreur: Le dataset n'existe pas à {dataset_dir}")
            return 1
    
    # Étape 2: Entraînement
    if not args.skip_training:
        if not train_model(str(dataset_dir), str(output_dir), args.num_classes):
            return 1
    else:
        print("Entraînement ignoré.")
    
    # Étape 3: Visualisation
    if not args.skip_visualization:
        # Chercher le modèle entraîné
        model_config = None
        model_weights = None
        
        if output_dir.exists():
            config_file = output_dir / "config.yaml"
            if config_file.exists():
                model_config = str(config_file)
            
            # Chercher le dernier checkpoint
            for file in output_dir.glob("model_*.pth"):
                model_weights = str(file)
                break
        
        if not visualize_results(str(dataset_dir), str(visualization_dir), 
                               model_config, model_weights):
            return 1
    else:
        print("Visualisation ignorée.")
    
    print("\n" + "=" * 80)
    print("   PROCESSUS TERMINÉ AVEC SUCCÈS!")
    print("=" * 80)
    print(f"\nRésultats disponibles dans: {workspace.absolute()}")
    print(f"- Dataset: {dataset_dir}")
    print(f"- Modèle entraîné: {output_dir}")
    print(f"- Visualisations: {visualization_dir}")
    print("\nPour entraîner sur votre propre dataset, remplacez le contenu")
    print("du dossier dataset par vos propres images et annotations.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
