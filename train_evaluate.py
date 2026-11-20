"""
Steps 2-5 - Split, preprocess, train, evaluate and save the model.

* Split is done BY SMEAR (source_id), never by single cell, so cells from the
  same blood smear never appear in both training and testing (no data leakage).
* Images are padded to a square (so cells are not stretched), resized to 224x224.
* Pixel normalisation to [-1, 1] happens INSIDE the model (Rescaling layer), so
  the app must feed raw 0-255 RGB pixels.
* Model: MobileNetV2 (ImageNet weights, frozen) + small classification head.

Usage:
    python src/train_evaluate.py --data data --out .
(creates models/ and results/ inside --out)
"""
import argparse
import json
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models

IMG = 224
SEED = 42
NAMES = ["normal", "sickle"]  # sickle = class 1; model output = P(sickle)


def to_square(img):
    h, w = img.shape[:2]
    s = max(h, w)
    top, left = (s - h) // 2, (s - w) // 2
    return cv2.copyMakeBorder(img, top, s - h - top, left, s - w - left, cv2.BORDER_REPLICATE)


def load(df):
    X = np.zeros((len(df), IMG, IMG, 3), dtype=np.uint8)
    for i, p in enumerate(df.path):
        img = to_square(cv2.imread(p))
        X[i] = cv2.cvtColor(cv2.resize(img, (IMG, IMG)), cv2.COLOR_BGR2RGB)
    return X, (df.label == "sickle").astype("int").values


def main(data_dir, out_dir):
    models_dir = os.path.join(out_dir, "models")
    results_dir = os.path.join(out_dir, "results")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    tf.keras.utils.set_random_seed(SEED)

    # ---- 1. Split by smear -------------------------------------------------
    m = pd.read_csv(os.path.join(data_dir, "manifest.csv"), dtype={"source_id": str})
    m["path"] = data_dir + "/raw/" + m["label"] + "/" + m["filename"]
    ids = np.array(sorted(m.source_id.unique()))
    np.random.RandomState(SEED).shuffle(ids)
    n = len(ids)
    train_ids = ids[:int(.70 * n)]
    val_ids = ids[int(.70 * n):int(.85 * n)]
    test_ids = ids[int(.85 * n):]
    assert not (set(train_ids) & set(val_ids))
    assert not (set(train_ids) & set(test_ids))
    assert not (set(val_ids) & set(test_ids))
    m["split"] = np.where(m.source_id.isin(train_ids), "train",
                          np.where(m.source_id.isin(val_ids), "val", "test"))
    m.drop(columns="path").to_csv(os.path.join(results_dir, "split.csv"), index=False)
    print(pd.crosstab(m["split"], m["label"]))

    X_train, y_train = load(m[m.split == "train"])
    X_val, y_val = load(m[m.split == "val"])
    X_test, y_test = load(m[m.split == "test"])

    # ---- 2. Model ----------------------------------------------------------
    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
    class_weight = {0: cw[0], 1: cw[1]}
    print("Class weights:", class_weight)

    base = tf.keras.applications.MobileNetV2(input_shape=(IMG, IMG, 3), include_top=False,
                                             weights="imagenet")
    base.trainable = False
    inputs = layers.Input((IMG, IMG, 3))
    x = layers.RandomFlip("horizontal_and_vertical")(inputs)
    x = layers.RandomRotation(0.2)(x)
    x = layers.Rescaling(1 / 127.5, offset=-1)(x)  # normalise to [-1, 1]
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy", metrics=["accuracy"])

    history = model.fit(
        X_train, y_train, validation_data=(X_val, y_val), epochs=15, batch_size=32,
        class_weight=class_weight,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4,
                                                    restore_best_weights=True)])

    # ---- 3. Accuracy / loss graphs ----------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(history.history["accuracy"], label="train")
    ax[0].plot(history.history["val_accuracy"], label="validation")
    ax[0].set_title("Accuracy"); ax[0].set_xlabel("Epoch"); ax[0].legend()
    ax[1].plot(history.history["loss"], label="train")
    ax[1].plot(history.history["val_loss"], label="validation")
    ax[1].set_title("Loss"); ax[1].set_xlabel("Epoch"); ax[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "training_curves.png"), dpi=150)
    plt.close()

    # ---- 4. Save model -----------------------------------------------------
    model.save(os.path.join(models_dir, "sickle_cell_mobilenetv2.keras"))

    # ---- 5. Evaluate on the untouched test set ----------------------------
    probs = model.predict(X_test).ravel()
    pred = (probs >= 0.5).astype(int)
    report = classification_report(y_test, pred, labels=[0, 1], target_names=NAMES,
                                   zero_division=0)
    acc = float((pred == y_test).mean())
    print(report)
    print("Test accuracy:", round(acc, 4))
    with open(os.path.join(results_dir, "classification_report.txt"), "w") as f:
        f.write(report + f"\nTest accuracy: {acc:.4f}\n")
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump({"test_accuracy": acc, "n_test": int(len(y_test)),
                   "n_train": int(len(y_train)), "n_val": int(len(y_val))}, f, indent=2)

    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    ConfusionMatrixDisplay(cm, display_labels=NAMES).plot(cmap="Blues")
    plt.title("Confusion matrix (test set)")
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", help="Folder with manifest.csv and raw/")
    ap.add_argument("--out", default=".", help="Where to create models/ and results/")
    a = ap.parse_args()
    main(a.data, a.out)
