# Générateur de Dataset Industriel - Images Réelles

## Description

Le script `generate_industrial_dataset.py` a été modifié pour traiter de vraies images avec défauts au lieu de générer des images synthétiques. Il conserve exactement la même structure d'annotation et la même logique, mais utilise maintenant des images provenant d'un répertoire source.

## Modifications apportées

### Changements principaux

1. **Constructeur modifié** : Prend maintenant un répertoire d'images en paramètre
2. **Nouvelles méthodes** :
   - `load_and_resize_image()` : Charge et redimensionne les images réelles
   - `process_real_image()` : Traite une image réelle pour extraire toutes les informations nécessaires
   - Méthodes d'analyse des types de défauts basées sur l'analyse d'image

3. **Méthode critique à adapter** : `generate_probability_matrix(image_path)`

## Utilisation

### Ligne de commande

```bash
python tools/generate_industrial_dataset.py \
    --images-dir /path/to/your/defect/images \
    --output-dir /path/to/output/dataset \
    --num-train 200 \
    --num-val 40
```

### Paramètres

- `--images-dir` : **OBLIGATOIRE** - Répertoire contenant vos images avec défauts
- `--output-dir` : **OBLIGATOIRE** - Répertoire de sortie pour le dataset
- `--num-train` : Nombre d'images d'entraînement (défaut: 200)
- `--num-val` : Nombre d'images de validation (défaut: 40)

## ⚠️ IMPORTANT : Adapter l'algorithme de détection

### Méthode à personnaliser

La méthode `generate_probability_matrix(image_path)` doit être adaptée avec votre algorithme de détection spécifique. 

**Actuellement** : Utilise un algorithme placeholder basé sur la détection de contours

**À faire** : Remplacer par votre véritable algorithme

### Exemples d'implémentation

#### Option 1: Utiliser MuSc
```python
def generate_probability_matrix(self, image_path):
    from Musc.musc_efficient_tester import MuScEfficientTester
    
    detector = MuScEfficientTester()
    probability_matrix = detector.detect_anomalies(image_path)
    
    # Redimensionner si nécessaire
    if probability_matrix.shape != self.image_size:
        probability_matrix = cv2.resize(probability_matrix, self.image_size)
    
    return probability_matrix.astype(np.float32)
```

#### Option 2: Utiliser un modèle pré-entraîné
```python
def generate_probability_matrix(self, image_path):
    # Charger votre modèle
    model = load_your_trained_model()
    
    # Préprocesser l'image
    image = self.load_and_resize_image(image_path)
    preprocessed = preprocess_for_model(image)
    
    # Prédiction
    probability_matrix = model.predict(preprocessed)
    
    return probability_matrix.astype(np.float32)
```

#### Option 3: Algorithme personnalisé
```python
def generate_probability_matrix(self, image_path):
    image = self.load_and_resize_image(image_path)
    
    # Votre algorithme personnalisé
    probability_matrix = your_custom_detection_algorithm(image)
    
    # S'assurer que les valeurs sont entre 0 et 1
    probability_matrix = np.clip(probability_matrix, 0, 1)
    
    return probability_matrix.astype(np.float32)
```

## Structure du dataset généré

```
output_dir/
├── images/
│   ├── train/           # Images d'entraînement redimensionnées
│   └── val/             # Images de validation redimensionnées
├── masks/
│   ├── train/           # Masques binaires (PNG)
│   └── val/
├── probability_maps/
│   ├── train/           # Matrices de probabilité (NPY)
│   └── val/
└── annotations/
    ├── instances_train.json  # Annotations COCO format
    └── instances_val.json
```

## Format des images sources

### Formats supportés
- `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`
- Recherche récursive dans les sous-dossiers

### Préprocessing automatique
- Conversion en niveaux de gris si nécessaire
- Redimensionnement à la taille configurée (défaut: 512x512)
- Préservation du ratio d'aspect avec remplissage si nécessaire

## Réutilisation des images

Si le nombre d'images demandées dépasse le nombre d'images disponibles, les images seront réutilisées automatiquement avec un mélange aléatoire.

## Métadonnées conservées

Le format COCO généré inclut :
- Informations sur l'image source (`source_image`)
- Types de défauts détectés automatiquement
- Bounding boxes extraites des masques binaires
- Répertoire source d'origine

## Exemple complet

```bash
# Préparer vos images
mkdir -p /data/defect_images
# Copier vos images avec défauts dans ce répertoire

# Générer le dataset
python tools/generate_industrial_dataset.py \
    --images-dir /data/defect_images \
    --output-dir /data/cutler_dataset \
    --num-train 150 \
    --num-val 30

# Le dataset est prêt pour l'entraînement CutLER
```

## Notes importantes

1. **Adaptez la méthode `generate_probability_matrix()`** avec votre algorithme avant utilisation en production
2. Les images sont automatiquement redimensionnées - assurez-vous que cela convient à votre cas d'usage
3. La détection automatique des types de défauts est basique - adaptez selon vos besoins
4. Le script affiche des avertissements quand il utilise l'algorithme placeholder

## Intégration avec CutLER

Le dataset généré est directement compatible avec l'entraînement CutLER. Utilisez le répertoire `annotations/` pour configurer votre entraînement.
