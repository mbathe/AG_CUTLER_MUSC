#!/usr/bin/env python3
"""
Script d'entraînement direct utilisant les outils CutLER existants.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    print("🚀 ENTRAÎNEMENT CUTLER - MÉTHODE DIRECTE")
    print("=" * 50)
    
    # Vérifier les arguments
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python train_direct.py <dataset_path> <output_dir> [num_classes]")
        print()
        print("Exemple:")
        print("  python train_direct.py ./mon_dataset ./output 4")
        return 1
    
    dataset_path = sys.argv[1]
    output_dir = sys.argv[2]
    num_classes = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    
    # Vérifier que le dataset existe
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset non trouvé: {dataset_path}")
        return 1
    
    print(f"📁 Dataset: {dataset_path}")
    print(f"📁 Sortie: {output_dir}")
    print(f"🏷️ Classes: {num_classes}")
    print()
    
    # Créer le répertoire de sortie
    os.makedirs(output_dir, exist_ok=True)
    
    # Utiliser le script d'entraînement original avec modifications
    config_file = "cutler/model_zoo/configs/CutLER-ImageNet/cascade_mask_rcnn_R_50_FPN_3x.yaml"
    
    if not os.path.exists(config_file):
        print(f"❌ Fichier de config non trouvé: {config_file}")
        print("Utilisation d'un config alternatif...")
        config_file = "cutler/model_zoo/configs/Base-RCNN-FPN.yaml"
    
    if not os.path.exists(config_file):
        print("❌ Aucun fichier de configuration trouvé")
        return 1
    
    # Préparer les variables d'environnement
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{os.getcwd()}:{os.getcwd()}/cutler"
    
    # Commande d'entraînement
    cmd = [
        sys.executable, "-m", "cutler.train_net",
        "--config-file", config_file,
        "--num-gpus", "1",
        f"MODEL.ROI_HEADS.NUM_CLASSES {num_classes}",
        f"OUTPUT_DIR {output_dir}",
        f"SOLVER.MAX_ITER 1000",  # Entraînement court pour test
        f"SOLVER.STEPS (500, 800)",
        f"SOLVER.IMS_PER_BATCH 2",
        f"TEST.EVAL_PERIOD 250"
    ]
    
    print("📝 Commande d'entraînement:")
    print(" ".join(cmd))
    print()
    
    try:
        print("🎯 Démarrage de l'entraînement...")
        result = subprocess.run(cmd, env=env, check=True)
        print("✅ Entraînement terminé avec succès!")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur d'entraînement: {e}")
        print()
        print("💡 Alternatives:")
        print("1. Vérifier l'installation de detectron2:")
        print("   conda list detectron2")
        print()
        print("2. Essayer avec un environnement conda spécifique:")
        print("   conda activate cutler")
        print("   python train_direct.py ...")
        print()
        print("3. Installer detectron2 manuellement:")
        print("   pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html")
        return 1

if __name__ == "__main__":
    sys.exit(main())
