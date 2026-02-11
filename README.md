# Football Player Detection with YOLO

Detección automática de jugadores, balón y árbitros en imágenes y vídeo de fútbol utilizando un modelo YOLOv8 entrenado sobre un dataset especializado.

Este proyecto muestra un flujo completo de Computer Vision:

* Preparación de dataset
* Entrenamiento de modelo YOLO
* Evaluación con métricas estándar
* Inferencia sobre vídeo real

---

## Objetivo del proyecto

Construir un modelo de detección de objetos capaz de identificar:

* Jugadores
* Balón
* Árbitros

Aplicado a escenas reales de fútbol, con enfoque práctico para análisis deportivo y sistemas de tracking.

---

## Dataset

Se utilizó el dataset:

**Soccana Player-Ball-Referee Detection**

Contiene imágenes etiquetadas en formato YOLO con tres clases:

```
0: Player
1: Ball
2: Referee
```

Estructura final tras preparación:

```
data/processed/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

---

## Arquitectura del modelo

Modelo utilizado:

```
YOLOv8n (Ultralytics)
```

Características:

* Arquitectura ligera (~3M parámetros)
* Optimizada para inferencia rápida
* Preentrenada en COCO
* Fine-tuning sobre dataset de fútbol

---

## Entrenamiento

Comando utilizado:

```bash
yolo detect train \
  model=yolov8n.pt \
  data=data/data.yaml \
  epochs=20 \
  imgsz=640 \
  batch=32 \
  device=0
```

Configuración clave:

* Epochs: 20
* Tamaño de imagen: 640
* Batch size: 32
* Entrenamiento en GPU

---

## Resultados

Evaluación sobre conjunto de test:

| Clase      | Precision | Recall   | mAP50    | mAP50-95 |
| ---------- | --------- | -------- | -------- | -------- |
| Player     | 0.91      | 0.90     | 0.94     | 0.60     |
| Ball       | 0.77      | 0.45     | 0.51     | 0.25     |
| Referee    | 0.81      | 0.74     | 0.80     | 0.47     |
| **Global** | **0.83**  | **0.70** | **0.75** | **0.44** |

Observaciones:

* Alta precisión en detección de jugadores.
* Detección del balón más compleja por tamaño reducido.
* Buen equilibrio general entre precisión y recall.

---

## Inferencia sobre vídeo

Ejemplo de predicción:

```bash
yolo detect predict \
  model=models/soccana_yolov8n.pt \
  source=data/raw/demo.mp4 \
  conf=0.25
```

Salida:

```
runs/detect/predict/
```

Contiene el vídeo con las detecciones dibujadas.

---

## Estructura del proyecto

```
football-player-detection-yolo/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── soccana_yolov8n.pt
│
├── src/
│   └── prepare_dataset.py
│
├── requirements.txt
├── requirements-gpu.txt
└── README.md
```

---

## Instalación

### 1. Clonar repositorio

```bash
git clone <repo-url>
cd football-player-detection-yolo
```

### 2. Crear entorno virtual

```bash
python -m venv .venv
```

En Linux / Mac:

```bash
source .venv/bin/activate
```

En Windows:

```bash
.venv\Scripts\activate
```

### 3. Instalar dependencias

CPU:

```bash
pip install -r requirements.txt
```

GPU (opcional):

```bash
pip install -r requirements-gpu.txt
pip install -r requirements.txt
```

---

## Reentrenar el modelo

1. Descargar dataset
2. Preparar estructura:

```bash
python -m src.prepare_dataset
```

3. Entrenar:

```bash
yolo detect train model=yolov8n.pt data=data/data.yaml epochs=20 imgsz=640
```

---

## Tecnologías utilizadas

* Python
* PyTorch
* Ultralytics YOLOv8
* OpenCV
* Pandas

---

## Posibles mejoras

* Entrenar con modelos más grandes (YOLOv8s, YOLOv8m).
* Aumentar número de epochs.
* Ajustar hiperparámetros.
* Añadir tracking de jugadores.
* Generar métricas de posesión o heatmaps.

---

## Autor

Proyecto desarrollado como parte de un portafolio de Data Science y Computer Vision orientado a aplicaciones reales.
