# Category Suggester — minimal web UI

A single upload page (`index.html`) + a tiny Flask server (`app.py`) that runs the
**exact** inference pipeline from `train.ipynb`
(encoder → kNN + text-prototype + supervised head + metadata → fusion → hierarchy).

## 1. Prerequisites: trained artifacts

Run `train.ipynb` first. The app reads the files it writes, under a `MODEL_DIR`:

```
MODEL_DIR/
  data/category_vocab.json
  data/category_graph_v2.json          (or category_graph.json)
  embeddings/cat_text_embs_<TAG>.npy
  embeddings/train_<TAG>_labels.npy
  models/faiss_index_<TAG>.bin
  models/supervised_head_<TAG>.pt
  models/fusion_model_<TAG>.pkl
```

`<TAG>` is the backbone name (e.g. `siglip-so400m`, `clip-L14`, `clip-B32`).
On Kaggle, download the `data/`, `embeddings/`, `models/` folders from the notebook
output and keep them together under one `MODEL_DIR`.

## 2. Install

```bash
pip install flask torch transformers faiss-cpu pillow numpy sentencepiece protobuf
```

(The project `myenv` already has these.)

## 3. Run

```bash
# MODEL_DIR = folder containing data/ embeddings/ models/  (default ".")
# BACKBONE  = optional; auto-detected from the artifacts if omitted
MODEL_DIR=.  BACKBONE=siglip-so400m  python webapp/app.py
```

Open **http://127.0.0.1:5000**, upload an image (optionally add a caption), and get
ranked category suggestions. First launch downloads the encoder weights.

## Endpoints
- `GET /`        – the upload page
- `GET /info`    – backbone, device, #categories
- `POST /predict` – multipart `image` (+ optional `caption`, `top_k`) → JSON predictions
