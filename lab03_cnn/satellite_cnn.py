"""
Классификатор спутниковых снимков на основе CNN.

Поддерживает два датасета:
  1) EuroSAT (https://www.tensorflow.org/datasets/catalog/eurosat)
     27 000 снимков Sentinel-2, 10 классов, загружается через TensorFlow Datasets.
  2) Любой датасет из папки с подпапками по классам (--dataset dir), например:
     - Satellite Image Classification (Kaggle)
       https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification
       4 класса (cloudy, desert, green_area, water), распаковать в data/.
     - EuroSAT, скачанный вручную и распакованный в data_eurosat/
       (подпапки классов лежат в data_eurosat/2750).

Запуск:
  python satellite_cnn.py --dataset eurosat --epochs 25
  python satellite_cnn.py --dataset dir --data-dir data --epochs 25
  python satellite_cnn.py --dataset dir --data-dir data_eurosat/2750 --epochs 25
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

SEED = 42
tf.keras.utils.set_random_seed(SEED)

IMG_SIZE = 64
BATCH_SIZE = 64


def load_eurosat():
    import tensorflow_datasets as tfds

    ds, info = tfds.load("eurosat/rgb", split="train", as_supervised=True,
                         with_info=True, shuffle_files=True)
    class_names = info.features["label"].names
    n = info.splits["train"].num_examples

    ds = ds.shuffle(10_000, seed=SEED, reshuffle_each_iteration=False)
    n_train, n_val = int(0.7 * n), int(0.15 * n)
    train = ds.take(n_train)
    val = ds.skip(n_train).take(n_val)
    test = ds.skip(n_train + n_val)
    return train, val, test, class_names, n_train, n_val


def load_from_dir(data_dir):
    train, val = tf.keras.utils.image_dataset_from_directory(
        data_dir, labels="inferred", label_mode="int",
        image_size=(IMG_SIZE, IMG_SIZE), batch_size=None,
        validation_split=0.3, subset="both", seed=SEED, shuffle=True)
    class_names = train.class_names

    val_n = int(0.5 * len(val))
    test = val.take(val_n)
    val = val.skip(val_n)
    return train, val, test, class_names


def prepare(ds, training, size, class_names):
    def norm(img, label):
        img = tf.cast(img, tf.float32) / 255.0
        img.set_shape([IMG_SIZE, IMG_SIZE, 3])
        label.set_shape([])
        return img, label

    ds = ds.map(norm, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(size, seed=SEED)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def build_model(num_classes):
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomContrast(0.1),
    ], name="augmentation")

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        augmentation,

        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def build_transfer_model(num_classes):
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomContrast(0.1),
    ], name="augmentation")

    base = tf.keras.applications.MobileNetV2(
        include_top=False, weights="imagenet",
        input_shape=(96, 96, 3), pooling="avg")
    base.trainable = False

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        augmentation,
        tf.keras.layers.Resizing(96, 96),
        tf.keras.layers.Rescaling(2.0, offset=-1.0),
        base,
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model, base


def unfreeze_base(base, n_last=40):
    base.trainable = True
    for layer in base.layers[:-n_last]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False


def merge_histories(*histories):
    merged = {k: [] for k in histories[0].history}
    for h in histories:
        for k in merged:
            merged[k].extend(h.history.get(k, []))
    return type("H", (), {"history": merged})()


def plot_history(history, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history["accuracy"], label="train")
    ax1.plot(history.history["val_accuracy"], label="val")
    ax1.set_title("Accuracy"); ax1.legend()
    ax2.plot(history.history["loss"], label="train")
    ax2.plot(history.history["val_loss"], label="val")
    ax2.set_title("Loss"); ax2.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_confusion(cm, class_names, path):
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_samples(model, test_ds, class_names, path):
    images, labels = next(iter(test_ds))
    preds = np.argmax(model.predict(images[:9], verbose=0), axis=1)
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for ax, img, true, pred in zip(axes.flat, images, labels, preds):
        ax.imshow(img.numpy())
        color = "green" if true == pred else "red"
        ax.set_title(f"{class_names[pred]} (true: {class_names[int(true)]})",
                     color=color, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["eurosat", "dir"], default="eurosat")
    parser.add_argument("--arch", choices=["cnn", "mobilenet"], default="cnn",
                        help="cnn — свёрточная сеть с нуля; mobilenet — transfer "
                             "learning на MobileNetV2 (ImageNet)")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.dataset == "eurosat":
        train_raw, val_raw, test_raw, class_names, n_train, _ = load_eurosat()
    else:
        train_raw, val_raw, test_raw, class_names = load_from_dir(args.data_dir)
        n_train = 10_000

    print(f"Классы ({len(class_names)}): {class_names}")

    train = prepare(train_raw, True, n_train, class_names)
    val = prepare(val_raw, False, 0, class_names)
    test = prepare(test_raw, False, 0, class_names)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5,
                                         restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=2, min_lr=1e-6),
    ]

    if args.arch == "mobilenet":
        model, base = build_transfer_model(len(class_names))
        model.summary()

        head_epochs = min(5, args.epochs // 3) or 1
        print(f"\nЭтап 1: обучение классификационной головы ({head_epochs} эпох)")
        h1 = model.fit(train, validation_data=val, epochs=head_epochs)

        unfreeze_base(base, n_last=60)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        print(f"\nЭтап 2: дообучение верхних слоёв ({args.epochs} эпох, lr=1e-5)")
        h2 = model.fit(train, validation_data=val, epochs=args.epochs,
                       callbacks=callbacks)
        history = merge_histories(h1, h2)
    else:
        model = build_model(len(class_names))
        model.summary()
        history = model.fit(train, validation_data=val, epochs=args.epochs,
                            callbacks=callbacks)

    model.save(os.path.join(args.out, "satellite_cnn.keras"))
    plot_history(history, os.path.join(args.out, "history.png"))

    loss, acc = model.evaluate(test, verbose=0)
    print(f"\nTest accuracy: {acc:.4f}, loss: {loss:.4f}")

    y_true = np.concatenate([y.numpy() for _, y in test])
    y_pred = np.argmax(model.predict(test, verbose=0), axis=1)
    print("\n", classification_report(y_true, y_pred, target_names=class_names))

    plot_confusion(confusion_matrix(y_true, y_pred), class_names,
                   os.path.join(args.out, "confusion_matrix.png"))
    plot_samples(model, test, class_names,
                 os.path.join(args.out, "predictions.png"))

    with open(os.path.join(args.out, "report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Dataset: {args.dataset}\nClasses: {class_names}\n")
        f.write(f"Test accuracy: {acc:.4f}, loss: {loss:.4f}\n\n")
        f.write(classification_report(y_true, y_pred, target_names=class_names))

    print(f"Результаты сохранены в {args.out}/")


if __name__ == "__main__":
    main()
