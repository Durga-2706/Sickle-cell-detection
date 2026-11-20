# 🩸 Sickle Cell Detection Using Deep Learning

An educational image-classification project: a MobileNetV2 transfer-learning model that classifies
single red blood cell images from blood smears as **normal (circular)** or **sickle-shaped (elongated)**,
plus a small Streamlit web app.

> ## ⚠️ NOT A MEDICAL DIAGNOSTIC TOOL
> This is a student/educational prototype. It must **never** be used to diagnose, screen for, or rule out
> sickle cell disease or any other condition. It was trained on a very small dataset and has not been
> clinically validated. Always consult a qualified healthcare professional.

## Dataset

**erythrocytesIDB** (Version 2, 2017) - peripheral blood smear images from patients with sickle cell disease,
provided by the UGiVIA research group, University of the Balearic Islands (Spain).
Official page: http://erythrocytesidb.uib.es/

- Access requires accepting the authors' **release agreement**, so **this repository does not include the dataset**.
  Please request it yourself from the official page and cite the authors' publications listed there.
- Used classes: `circular` → **normal**, `elongated` → **sickle**. The `other` class is not used.
- The dataset provides 50 smear photos, each with one mask per cell type. `src/prepare_dataset.py` cuts each
  masked cell out of its smear photo.

## How it works

1. **Prepare data** - crop every cell using the masks, keep the ID of the smear it came from (`manifest.csv`).
2. **Split by smear, not by cell** - whole smears go to train (70%), validation (15%) or test (15%), so cells from
   the same smear never appear in two splits (prevents data leakage).
3. **Preprocess** - pad each crop to a square (no stretching), resize to 224×224, normalise pixels to [-1, 1]
   (inside the model).
4. **Model** - MobileNetV2 (ImageNet weights, frozen) + global average pooling + dropout + 1 sigmoid output.
   Light augmentation (flips, rotation) and class weights (normal cells outnumber sickle cells about 3:1).
5. **Evaluate** on the untouched test set: accuracy/loss curves, confusion matrix, classification report.

## Project structure

```
sickle-cell-detection/
├── app.py                    # Streamlit web app
├── requirements.txt
├── src/
│   ├── prepare_dataset.py    # crops cells from the dataset using the masks
│   └── train_evaluate.py     # split, train, evaluate, save model
├── models/                   # trained model goes here
├── results/                  # graphs, confusion matrix, report
└── data/                     # NOT included (create it yourself, see below)
```

## Setup

Python 3.10-3.12 is recommended (TensorFlow does not support every Python version).

```bash
git clone https://github.com/<your-username>/sickle-cell-detection.git
cd sickle-cell-detection
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproduce the results

```bash
# 1. Crop cells (point --idb2 at the erythrocytesIDB2 folder of the dataset)
python src/prepare_dataset.py --idb2 "path/to/Version 2, erythrocytesIDB/erythrocytesIDB2" --out data --check

# 2. Train, evaluate and save the model (a GPU, e.g. Google Colab, makes this much faster)
python src/train_evaluate.py --data data --out .
```

Open `data/alignment_check.png` after step 1 and confirm that the green (normal) and red (sickle) outlines sit on
the right cells. Outputs: `models/sickle_cell_mobilenetv2.keras` and everything in `results/`.

## Run the web app

```bash
streamlit run app.py
```

Upload an image of a **single cell** (like the training crops). The app shows the predicted class and a confidence score.
Whole-smear photos with many cells are not supported.

## Results

> Fill this in with **your own** numbers from `results/classification_report.txt` and `results/metrics.json`.
> Do not copy numbers from anywhere else.

| Metric (test set) | Value |
|---|---|
| Test accuracy | _fill in_ |
| Precision / recall / F1 - normal | _fill in_ |
| Precision / recall / F1 - sickle | _fill in_ |
| Test images (normal / sickle) | _fill in_ |

Training curves: `results/training_curves.png`  ·  Confusion matrix: `results/confusion_matrix.png`

## Limitations (please read)

- **"Elongated" ≠ confirmed sickle cell.** The dataset's *elongated* class is an expert shape category; this
  project treats it as "sickle-shaped". Real diagnosis needs far more than cell shape.
- **Small dataset** (~1,800 crops from 50 smears, one source), so the test set is small and scores can vary a lot
  with a different split. No external validation was done.
- **Class imbalance** (about 3 normal : 1 sickle), handled with class weights, not eliminated.
- **Touching cells** appear as one blob in the masks, so a few crops contain two cells.
- **Single-cell input only.** The model has never seen full smear photos, other stains, or other microscopes.
- Confidence scores are the model's own probabilities and can be high and still wrong.

## Acknowledgements

- Dataset: erythrocytesIDB, UGiVIA group, University of the Balearic Islands. Please cite the authors' papers as
  requested on the dataset page.
- Model: MobileNetV2 (Sandler et al., 2018), ImageNet pre-trained weights via TensorFlow/Keras.

## Licence

Add a licence of your choice (e.g. MIT) for **your code**. It does not cover the dataset, which has its own terms.
