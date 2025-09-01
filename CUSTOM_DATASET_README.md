# CutLER - Support pour Datasets Personnalisés

Ce guide explique comment utiliser CutLER avec vos propres datasets pour l'entraînement de modèles de détection d'objets non supervisés.

## 🚀 Fonctionnalités Ajoutées

- **Génération automatique de datasets synthétiques**
- **Support complet des datasets au format COCO**
- **Scripts d'entraînement adaptés pour datasets personnalisés**
- **Outils de visualisation et de comparaison**
- **Pipeline automatisé de bout en bout**

## 📁 Structure des Fichiers Ajoutés

```
CutLER/
├── tools/
│   ├── generate_custom_dataset.py    # Génère un dataset synthétique
│   ├── train_custom.py              # Entraîne sur dataset personnalisé
│   ├── visualize_custom.py          # Visualise les résultats
│   └── run_custom_pipeline.py       # Script principal tout-en-un
├── cutler/
│   ├── data/datasets/
│   │   └── custom_datasets.py       # Support datasets personnalisés
│   └── model_zoo/configs/Custom-Dataset/
│       └── cascade_mask_rcnn_R_50_FPN_custom.yaml
└── test.ipynb                       # Notebook de démonstration
```

## 🛠️ Installation et Dépendances

### Dépendances Python Requises

```bash
pip install numpy opencv-python matplotlib Pillow
```

### Vérification de l'Installation

```python
python tools/run_custom_pipeline.py --help
```

## 📊 Utilisation Rapide

### 1. Test avec Dataset Synthétique

Générez et testez rapidement avec un dataset d'exemple :

```bash
python tools/run_custom_pipeline.py \
    --workspace ./test_workspace \
    --num-train 100 \
    --num-val 20 \
    --num-classes 3
```

### 2. Génération de Dataset Synthétique

```bash
python tools/generate_custom_dataset.py \
    --output-dir ./mon_dataset \
    --num-train 1000 \
    --num-val 200
```

Cela crée un dataset avec des formes géométriques (cercles, rectangles, triangles).

### 3. Entraînement sur Votre Dataset

```bash
python tools/train_custom.py \
    --dataset-path ./mon_dataset \
    --output-dir ./output \
    --num-classes 3
```

### 4. Visualisation des Résultats

```bash
# Annotations ground truth
python tools/visualize_custom.py \
    --dataset-path ./mon_dataset \
    --output-dir ./visualizations \
    --mode gt

# Comparaison GT vs Prédictions
python tools/visualize_custom.py \
    --dataset-path ./mon_dataset \
    --output-dir ./visualizations \
    --mode compare \
    --model-config ./output/config.yaml \
    --model-weights ./output/model_final.pth
```

## 📋 Format du Dataset

### Structure Requise

```
mon_dataset/
├── images/
│   ├── train/
│   │   ├── image001.jpg
│   │   ├── image002.jpg
│   │   └── ...
│   ├── val/
│   │   ├── image001.jpg
│   │   └── ...
│   └── test/ (optionnel)
│       └── ...
└── annotations/
    ├── instances_train.json
    ├── instances_val.json
    └── instances_test.json (optionnel)
```

### Format des Annotations (COCO JSON)

```json
{
  "info": {
    "description": "Mon Dataset Personnalisé",
    "version": "1.0",
    "year": 2024
  },
  "categories": [
    {"id": 1, "name": "classe1", "supercategory": "objet"},
    {"id": 2, "name": "classe2", "supercategory": "objet"}
  ],
  "images": [
    {
      "id": 1,
      "width": 640,
      "height": 480,
      "file_name": "image001.jpg"
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "segmentation": [[x1, y1, x2, y2, ...]],
      "area": 12345,
      "bbox": [x, y, width, height],
      "iscrowd": 0
    }
  ]
}
```

## ⚙️ Configuration Avancée

### Paramètres d'Entraînement

Modifiez `cutler/model_zoo/configs/Custom-Dataset/cascade_mask_rcnn_R_50_FPN_custom.yaml` :

```yaml
MODEL:
  ROI_HEADS:
    NUM_CLASSES: 10  # Votre nombre de classes

SOLVER:
  IMS_PER_BATCH: 4     # Taille du batch
  BASE_LR: 0.002       # Taux d'apprentissage
  MAX_ITER: 5000       # Nombre d'itérations
  STEPS: (3000, 4000)  # Étapes de réduction du LR

DATALOADER:
  NUM_WORKERS: 4       # Nombre de workers
```

### Classes Personnalisées

Dans `cutler/data/datasets/custom_datasets.py`, modifiez :

```python
CUSTOM_CATEGORIES = [
    {"id": 1, "name": "ma_classe1", "supercategory": "objet"},
    {"id": 2, "name": "ma_classe2", "supercategory": "objet"},
    # Ajoutez vos classes...
]
```

## 📈 Monitoring et Évaluation

### TensorBoard

```bash
tensorboard --logdir ./output
```

### Métriques d'Évaluation

Le système utilise les métriques COCO standard :
- **AP (Average Precision)** à différents seuils IoU
- **AR (Average Recall)** 
- **AP par classe**

### Logs d'Entraînement

Consultez `./output/log.txt` pour suivre la progression.

## 🔧 Dépannage

### Problèmes Courants

1. **Erreur de mémoire GPU**
   ```yaml
   SOLVER:
     IMS_PER_BATCH: 1  # Réduire la taille du batch
   ```

2. **Dataset non trouvé**
   - Vérifiez la structure des dossiers
   - Vérifiez les chemins dans les fichiers JSON

3. **Erreur de format JSON**
   - Validez votre JSON avec un validateur en ligne
   - Vérifiez que tous les IDs sont uniques

### Validation du Dataset

```python
python tools/visualize_custom.py \
    --dataset-path ./mon_dataset \
    --output-dir ./validation \
    --mode gt \
    --num-images 10
```

## 🎯 Examples d'Utilisation

### Dataset d'Images Médicales

```bash
python tools/train_custom.py \
    --dataset-path ./medical_images \
    --output-dir ./medical_model \
    --num-classes 5
```

### Dataset de Véhicules

```bash
python tools/train_custom.py \
    --dataset-path ./vehicles_dataset \
    --output-dir ./vehicle_detector \
    --num-classes 8
```

### Dataset de Produits

```bash
python tools/train_custom.py \
    --dataset-path ./products_dataset \
    --output-dir ./product_detector \
    --num-classes 20
```

## 📚 Ressources Supplémentaires

- [Documentation COCO Format](https://cocodataset.org/#format-data)
- [Documentation Detectron2](https://detectron2.readthedocs.io/)
- [Paper CutLER Original](https://arxiv.org/abs/2301.11320)

## 🤝 Contribution

Pour contribuer à l'amélioration du support des datasets personnalisés :

1. Créez une issue pour discuter des changements
2. Fork le repository
3. Créez votre feature branch
4. Testez vos modifications
5. Soumettez une pull request

## 📄 License

Ce code suit la même licence que le projet CutLER original.

---

**Note** : Ce système est conçu pour être facile à utiliser tout en conservant la flexibilité nécessaire pour des cas d'usage avancés. N'hésitez pas à adapter les configurations selon vos besoins spécifiques.
