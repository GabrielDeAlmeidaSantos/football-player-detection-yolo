"""
prepare_dataset.py

OBJETIVO
--------
Preparar un dataset en formato YOLO listo para entrenamiento con Ultralytics, a partir
de un dataset descargado y extraído en `data/raw/soccana/V1/`.

El dataset Soccana viene con esta estructura:
- images/train, labels/train
- images/test,  labels/test

Pero para un proyecto serio (y más presentable en portfolio), se recomienda separar:
- train: para aprender
- val: para ajustar/monitorizar durante el entrenamiento
- test: "examen final" que no tocamos hasta el final

Este script hace exactamente eso:
1) Copia TRAIN tal cual.
2) Divide el split TEST en dos mitades (50/50) para obtener VAL y TEST.
3) Verifica que cada imagen tenga su label .txt correspondiente.
4) Construye el dataset final en:
   data/processed/
     images/{train,val,test}
     labels/{train,val,test}

REPRODUCIBILIDAD
----------------
Usamos una semilla fija (seed=42) para que el split val/test sea SIEMPRE el mismo.
Esto es importante para poder comparar experimentos (métricas) de forma justa.
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Iterable


# Extensiones de imagen que aceptaremos como inputs del dataset.
# (YOLO suele trabajar con jpg/png; si tu dataset tuviera otra cosa, la añades aquí)
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def ensure_dir(path: Path) -> None:
    """
    Crea una carpeta (y sus padres) si no existe.

    Por qué:
    - Evita errores al copiar archivos a directorios que aún no existen.
    - `parents=True` crea también carpetas superiores necesarias.
    """
    path.mkdir(parents=True, exist_ok=True)


def list_images(folder: Path) -> list[Path]:
    """
    Devuelve una lista ordenada de imágenes dentro de una carpeta (no recursivo).

    Por qué:
    - Queremos un conjunto claro de ficheros imagen.
    - Ordenar ayuda a tener comportamiento determinista (aunque luego barajemos para split).
    """
    if not folder.exists():
        raise FileNotFoundError(f"No existe la carpeta de imágenes: {folder}")

    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS])


def expected_label_path(img_path: Path, labels_dir: Path) -> Path:
    """
    Dada una imagen `xxx.jpg`, construye la ruta del label esperado `xxx.txt`
    dentro de `labels_dir`.

    En YOLO, la relación es por nombre base:
    - images/train/frame_001.jpg  -> labels/train/frame_001.txt
    """
    return labels_dir / f"{img_path.stem}.txt"


def copy_image_and_label(img: Path, lbl: Path, out_img_dir: Path, out_lbl_dir: Path) -> None:
    """
    Copia una imagen y su label asociado a carpetas de salida.

    Usamos copy2 para mantener metadatos (fechas) cuando sea posible.
    """
    ensure_dir(out_img_dir)
    ensure_dir(out_lbl_dir)

    shutil.copy2(img, out_img_dir / img.name)
    shutil.copy2(lbl, out_lbl_dir / lbl.name)


def filter_images_with_labels(images: Iterable[Path], labels_dir: Path) -> tuple[list[Path], int]:
    """
    Filtra imágenes dejando solo las que tengan su label .txt correspondiente.

    Devuelve:
    - lista de imágenes válidas (con label)
    - número de imágenes que faltaban label

    Por qué:
    - Si hay imágenes sin label, entrenar puede introducir ruido o errores.
    - Aquí decidimos OMITIRLAS y dejarlo registrado.
      (Alternativa: crear label vacío para indicar "sin objetos", pero aquí no lo hacemos.)
    """
    valid: list[Path] = []
    missing = 0

    for img in images:
        lbl = expected_label_path(img, labels_dir)
        if not lbl.exists():
            missing += 1
            continue
        valid.append(img)

    return valid, missing


def split_list(items: list[Path], val_ratio: float = 0.5, seed: int = 42) -> tuple[list[Path], list[Path]]:
    """
    Divide una lista en dos subconjuntos (val y test) usando barajado con semilla fija.

    val_ratio=0.5  -> 50% val, 50% test

    Por qué:
    - Necesitamos un conjunto de validación y otro de test.
    - La semilla fija hace el split reproducible.
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio debe estar entre 0 y 1 (ej: 0.5).")

    rnd = random.Random(seed)
    shuffled = items[:]  # copia para no mutar la lista original
    rnd.shuffle(shuffled)

    cut = int(len(shuffled) * val_ratio)
    val_items = shuffled[:cut]
    test_items = shuffled[cut:]

    return val_items, test_items


def main() -> None:
    """
    Orquesta todo el proceso:
    - Define rutas (raw y processed)
    - Limpia `data/processed` para garantizar una salida limpia
    - Copia train
    - Parte test -> val/test y copia
    - Imprime resumen final
    """
    repo_root = Path(__file__).resolve().parents[1]

    # RUTAS DE ENTRADA (raw)
    raw_root = repo_root / "data" / "raw" / "soccana" / "V1" / "V1"
    raw_images_train = raw_root / "images" / "train"
    raw_labels_train = raw_root / "labels" / "train"
    raw_images_test = raw_root / "images" / "test"
    raw_labels_test = raw_root / "labels" / "test"

    # Validación rápida de que el dataset está donde esperamos
    for p in [raw_images_train, raw_labels_train, raw_images_test, raw_labels_test]:
        if not p.exists():
            raise FileNotFoundError(f"Falta carpeta esperada del dataset: {p}")

    # RUTAS DE SALIDA (processed)
    processed = repo_root / "data" / "processed"

    out_img_train = processed / "images" / "train"
    out_img_val = processed / "images" / "val"
    out_img_test = processed / "images" / "test"

    out_lbl_train = processed / "labels" / "train"
    out_lbl_val = processed / "labels" / "val"
    out_lbl_test = processed / "labels" / "test"

    # Limpiamos processed para evitar mezclar ejecuciones anteriores con la actual
    # (Esto es importante cuando repites pruebas y no quieres "basura" de antes)
    if processed.exists():
        shutil.rmtree(processed)
    ensure_dir(processed)

    # 1) TRAIN: listar imágenes, filtrar por labels y copiar
    train_images_all = list_images(raw_images_train)
    train_images, missing_train = filter_images_with_labels(train_images_all, raw_labels_train)

    copied_train = 0
    for img in train_images:
        lbl = expected_label_path(img, raw_labels_train)
        copy_image_and_label(img, lbl, out_img_train, out_lbl_train)
        copied_train += 1

    # 2) TEST original: filtrar y luego dividirlo en VAL/TEST
    test_images_all = list_images(raw_images_test)
    test_images_valid, missing_test = filter_images_with_labels(test_images_all, raw_labels_test)

    val_imgs, test_imgs = split_list(test_images_valid, val_ratio=0.5, seed=42)

    copied_val = 0
    for img in val_imgs:
        lbl = expected_label_path(img, raw_labels_test)
        copy_image_and_label(img, lbl, out_img_val, out_lbl_val)
        copied_val += 1

    copied_test = 0
    for img in test_imgs:
        lbl = expected_label_path(img, raw_labels_test)
        copy_image_and_label(img, lbl, out_img_test, out_lbl_test)
        copied_test += 1

    # Resumen para que sepas exactamente qué pasó
    print("=== DATASET PREPARADO (YOLO) ===")
    print(f"Train: {copied_train} pares copiados | {missing_train} imágenes sin label (omitidas)")
    print(f"Val:   {copied_val} pares copiados")
    print(f"Test:  {copied_test} pares copiados")
    print(f"Split val/test: seed=42, ratio=50/50")
    print(f"Test originales sin label (omitidas): {missing_test}")
    print(f"Salida: {processed}")


if __name__ == "__main__":
    main()