#!/bin/bash

# Script pour configurer l'environnement CutLER avec support des datasets personnalisés

echo "Configuration de l'environnement CutLER pour datasets personnalisés..."

# Rendre les scripts exécutables
echo "Activation des permissions d'exécution..."
chmod +x tools/generate_custom_dataset.py
chmod +x tools/train_custom.py
chmod +x tools/visualize_custom.py
chmod +x tools/run_custom_pipeline.py

# Créer les dossiers nécessaires
echo "Création des dossiers de travail..."
mkdir -p datasets/custom
mkdir -p output
mkdir -p visualizations

# Vérifier que les dépendances Python sont installées
echo "Vérification des dépendances Python..."
python3 -c "
try:
    import numpy
    print('✓ numpy')
except ImportError:
    print('✗ numpy - Installer avec: pip install numpy')

try:
    import cv2
    print('✓ opencv-python')
except ImportError:
    print('✗ opencv-python - Installer avec: pip install opencv-python')

try:
    import matplotlib
    print('✓ matplotlib')
except ImportError:
    print('✗ matplotlib - Installer avec: pip install matplotlib')

try:
    from PIL import Image
    print('✓ Pillow')
except ImportError:
    print('✗ Pillow - Installer avec: pip install Pillow')
"

echo ""
echo "Configuration terminée!"
echo ""
echo "Pour tester le système avec un dataset synthétique:"
echo "  python tools/run_custom_pipeline.py --workspace ./test_workspace --num-train 50 --num-val 10"
echo ""
echo "Pour entraîner sur votre propre dataset:"
echo "  python tools/train_custom.py --dataset-path /chemin/vers/votre/dataset --output-dir ./output --num-classes NOMBRE_CLASSES"
echo ""
echo "Pour visualiser les résultats:"
echo "  python tools/visualize_custom.py --dataset-path /chemin/vers/dataset --output-dir ./visualizations --mode gt"
echo ""
echo "Consultez CUSTOM_DATASET_README.md pour plus de détails."
