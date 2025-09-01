#!/usr/bin/env python3
"""
Guide d'installation et d'utilisation pour CutLER avec datasets personnalisés.
"""

import os
import sys
import subprocess

def main():
    print("=" * 80)
    print("   GUIDE CUTLER - DATASETS PERSONNALISÉS")
    print("=" * 80)
    print()
    
    print("✅ ÉTAPE 1: Vérification de l'environnement")
    print("-" * 50)
    
    # Vérifier Python
    python_version = sys.version_info
    print(f"Python: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Vérifier CUDA
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"CUDA: ✓ Disponible (GPU: {torch.cuda.get_device_name(0)})")
        else:
            print("CUDA: ⚠️ Non disponible (CPU seulement)")
    except ImportError:
        print("PyTorch: ❌ Non installé")
    
    # Vérifier detectron2
    try:
        import detectron2
        print("Detectron2: ✓ Installé")
        detectron2_ok = True
    except ImportError:
        print("Detectron2: ❌ Non installé")
        detectron2_ok = False
    
    print()
    print("✅ ÉTAPE 2: Installation (si nécessaire)")
    print("-" * 50)
    
    if not detectron2_ok:
        print("Pour installer detectron2:")
        print("1. Avec CUDA (recommandé si vous avez une GPU):")
        print("   pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html")
        print()
        print("2. Version CPU seulement:")
        print("   pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.0/index.html")
        print()
    
    print("Dépendances pour la génération de datasets:")
    print("  pip install numpy opencv-python matplotlib Pillow")
    print()
    
    print("✅ ÉTAPE 3: Test rapide")
    print("-" * 50)
    print("Générer et visualiser un dataset d'exemple:")
    print("  python tools/run_custom_pipeline.py \\")
    print("    --workspace ./test_workspace \\")
    print("    --num-train 20 \\")
    print("    --num-val 5 \\")
    print("    --skip-training")
    print()
    
    print("✅ ÉTAPE 4: Utilisation avec votre dataset")
    print("-" * 50)
    print("Structure requise pour votre dataset:")
    print("  mon_dataset/")
    print("  ├── images/")
    print("  │   ├── train/")
    print("  │   │   ├── image1.jpg")
    print("  │   │   └── ...")
    print("  │   └── val/")
    print("  │       └── ...")
    print("  └── annotations/")
    print("      ├── instances_train.json")
    print("      └── instances_val.json")
    print()
    
    print("Entraînement:")
    print("  python tools/train_custom_simple.py \\")
    print("    --dataset-path ./mon_dataset \\")
    print("    --output-dir ./output \\")
    print("    --num-classes NOMBRE_DE_CLASSES")
    print()
    
    print("Visualisation:")
    print("  python tools/visualize_custom.py \\")
    print("    --dataset-path ./mon_dataset \\")
    print("    --output-dir ./visualizations \\")
    print("    --mode gt")
    print()
    
    print("✅ ÉTAPE 5: Fichiers créés/modifiés")
    print("-" * 50)
    
    files_to_check = [
        "tools/generate_custom_dataset.py",
        "tools/train_custom_simple.py", 
        "tools/visualize_custom.py",
        "tools/run_custom_pipeline.py",
        "cutler/data/datasets/custom_datasets.py",
        "CUSTOM_DATASET_README.md"
    ]
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"  ✓ {file_path}")
        else:
            print(f"  ❌ {file_path}")
    
    print()
    print("✅ AIDE ET DOCUMENTATION")
    print("-" * 50)
    print("• README détaillé: CUSTOM_DATASET_README.md")
    print("• Notebook de démonstration: test.ipynb")
    print("• Format COCO: https://cocodataset.org/#format-data")
    print("• Documentation CutLER: https://github.com/facebookresearch/CutLER")
    print()
    
    print("✅ PRÊT À UTILISER!")
    print("-" * 50)
    print("Commencez par le test rapide (étape 3) pour vérifier que tout fonctionne.")
    print()
    
    # Proposer de lancer le test
    response = input("Voulez-vous lancer le test rapide maintenant? [y/N]: ")
    if response.lower() in ['y', 'yes', 'oui']:
        print("\nLancement du test...")
        cmd = [
            sys.executable, "tools/run_custom_pipeline.py",
            "--workspace", "./demo_workspace",
            "--num-train", "5",
            "--num-val", "2",
            "--skip-training"
        ]
        try:
            subprocess.run(cmd, check=True)
            print("\n🎉 Test réussi! Votre environnement est prêt.")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Test échoué: {e}")

if __name__ == "__main__":
    main()
