#!/usr/bin/env python3
"""
Script de test simple pour vérifier la génération de dataset.
"""

import os
import sys
import subprocess

def test_dataset_generation():
    """Test simple de la génération de dataset."""
    
    print("Test de génération de dataset...")
    
    # Installer les dépendances si nécessaire
    try:
        import numpy
        import cv2
        import matplotlib
        from PIL import Image
        print("✓ Toutes les dépendances sont disponibles")
    except ImportError as e:
        print(f"Installation des dépendances manquantes...")
        packages = ["numpy", "opencv-python", "matplotlib", "Pillow"]
        
        for package in packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✓ {package} installé")
            except subprocess.CalledProcessError:
                print(f"✗ Erreur lors de l'installation de {package}")
                return False
    
    # Tester la génération
    script_path = os.path.join(os.path.dirname(__file__), "generate_custom_dataset.py")
    test_output = "./test_dataset_simple"
    
    cmd = [
        sys.executable, script_path,
        "--output-dir", test_output,
        "--num-train", "5",
        "--num-val", "2"
    ]
    
    try:
        print("Exécution de la génération de dataset...")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✓ Dataset généré avec succès!")
        print(f"Sortie: {result.stdout}")
        
        # Vérifier que les fichiers ont été créés
        expected_files = [
            os.path.join(test_output, "annotations", "instances_train.json"),
            os.path.join(test_output, "annotations", "instances_val.json"),
        ]
        
        for file_path in expected_files:
            if os.path.exists(file_path):
                print(f"✓ {file_path}")
            else:
                print(f"✗ {file_path}")
                return False
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Erreur: {e}")
        print(f"Sortie: {e.stdout}")
        print(f"Erreur: {e.stderr}")
        return False

if __name__ == "__main__":
    success = test_dataset_generation()
    if success:
        print("\n🎉 Test réussi! La génération de dataset fonctionne.")
        print("Vous pouvez maintenant utiliser:")
        print("  python tools/run_custom_pipeline.py --workspace ./test_workspace --skip-training")
    else:
        print("\n❌ Test échoué. Vérifiez les erreurs ci-dessus.")
    
    sys.exit(0 if success else 1)
