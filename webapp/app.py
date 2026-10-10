#!/usr/bin/env python3
"""
Minimal inference server for the Wikimedia Commons category suggester.

Serves a single upload page (index.html) and a /predict endpoint that runs the
SAME pipeline as train.ipynb: CLIP/SigLIP encoder -> kNN + text-prototype +
supervised head + metadata scorers -> calibrated fusion -> hierarchy DAG.

It loads the artifacts the notebook writes (per-backbone, cache-tagged):
    <MODEL_DIR>/data/category_vocab.json
    <MODEL_DIR>/data/category_graph_v2.json   (or category_graph.json)
    <MODEL_DIR>/embeddings/cat_text_embs_<TAG>.npy
    <MODEL_DIR>/embeddings/train_<TAG>_labels.npy
    <MODEL_DIR>/models/faiss_index_<TAG>.bin
    <MODEL_DIR>/models/supervised_head_<TAG>.pt
    <MODEL_DIR>/models/fusion_model_<TAG>.pkl

Run:
    MODEL_DIR=. BACKBONE=siglip-so400m  python webapp/app.py
    # then open http://127.0.0.1:5000
"""
import os, io, re, json, pickle, glob
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import faiss
from transformers import AutoModel, AutoProcessor
from flask import Flask, request, jsonify, send_from_directory

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
MODEL_DIR = Path(os.environ.get('MODEL_DIR', '.')).resolve()
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
HERE      = Path(__file__).resolve().parent

BACKBONES = {
    'siglip-so400m': dict(kind='siglip', name='google/siglip-so400m-patch14-384'),
    'siglip-large' : dict(kind='siglip', name='google/siglip-large-patch16-256'),
    'siglip-base'  : dict(kind='siglip', name='google/siglip-base-patch16-224'),
    'clip-laion-H' : dict(kind='clip',   name='laion/CLIP-ViT-H-14-laion2B-s32B-b79K'),
    'clip-L14'     : dict(kind='clip',   name='openai/clip-vit-large-patch14'),
    'clip-B32'     : dict(kind='clip',   name='openai/clip-vit-base-patch32'),
}
_PRIORITY = ['siglip-so400m', 'siglip-large', 'clip-laion-H', 'clip-L14',
             'siglip-base', 'clip-B32']


# --------------------------------------------------------------------------- #
# Classes needed to unpickle the fusion model (must match train.ipynb names)
# --------------------------------------------------------------------------- #
class LogisticRegression:
    def __init__(self, lr=0.05, n_iter=300, C=1.0, tol=1e-5):
        self.lr, self.n_iter, self.C, self.tol = lr, n_iter, C, tol
        self.coef_ = None; self.intercept_ = None
    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
    def predict_proba(self, X):
        p = self._sigmoid(X @ self.coef_[0] + self.intercept_[0])
        return np.column_stack([1 - p, p])


class MultiLabelHead(nn.Module):
    def __init__(self, emb_dim, n_classes):
        super().__init__()
        self.fc = nn.Linear(emb_dim, n_classes)
    def forward(self, z):
        return self.fc(z)


# --------------------------------------------------------------------------- #
# Pick backbone + resolve artifact paths
# --------------------------------------------------------------------------- #
def detect_backbone():
    env = os.environ.get('BACKBONE')
    if env:
        if env not in BACKBONES:
            raise SystemExit(f'Unknown BACKBONE={env}. Options: {list(BACKBONES)}')
        return env
    found = set()
    for p in glob.glob(str(MODEL_DIR / 'models' / 'faiss_index_*.bin')):
        tag = Path(p).stem.replace('faiss_index_', '')
        if tag in BACKBONES:
            found.add(tag)
    for b in _PRIORITY:
        if b in found:
            return b
    raise SystemExit(
        f'No trained artifacts found under {MODEL_DIR}/models. '
        f'Run train.ipynb first, then set MODEL_DIR to its folder.')


BACKBONE = detect_backbone()
TAG      = re.sub(r'[^0-9A-Za-z]+', '-', BACKBONE)
CFG      = BACKBONES[BACKBONE]
KIND     = CFG['kind']
print(f'[app] device={DEVICE}  backbone={BACKBONE}  model_dir={MODEL_DIR}')

EMB_DIR = MODEL_DIR / 'embeddings'
MDL_DIR = MODEL_DIR / 'models'
DAT_DIR = MODEL_DIR / 'data'


# --------------------------------------------------------------------------- #
# Load encoder + artifacts
# --------------------------------------------------------------------------- #
print(f'[app] loading encoder {CFG["name"]} ...')
enc_model = AutoModel.from_pretrained(CFG['name']).to(DEVICE).eval()
enc_proc  = AutoProcessor.from_pretrained(CFG['name'])

def _feats(out):
    if torch.is_tensor(out):
        return out
    pooled = getattr(out, 'pooler_output', None)
    if pooled is not None:
        return pooled
    if isinstance(out, (tuple, list)) and len(out) >= 2:
        return out[1]
    raise TypeError(type(out))

@torch.no_grad()
def embed_image(pil_img):
    inp = enc_proc(images=[pil_img], return_tensors='pt').to(DEVICE)
    return F.normalize(_feats(enc_model.get_image_features(**inp)), dim=-1).cpu().numpy().astype('float32')

@torch.no_grad()
def embed_texts(texts):
    pad = (dict(padding='max_length', max_length=64) if KIND == 'siglip'
           else dict(padding=True, max_length=77))
    inp = enc_proc(text=texts, return_tensors='pt', truncation=True, **pad).to(DEVICE)
    return F.normalize(_feats(enc_model.get_text_features(**inp)), dim=-1).cpu().numpy().astype('float32')

EMB_DIM = int(embed_texts(['test']).shape[1])

ALL_CATS     = json.load(open(DAT_DIR / 'category_vocab.json', encoding='utf-8'))
cat_text_embs = np.load(str(EMB_DIR / f'cat_text_embs_{TAG}.npy'))
train_labels  = np.load(str(EMB_DIR / f'train_{TAG}_labels.npy'))
faiss_index   = faiss.read_index(str(MDL_DIR / f'faiss_index_{TAG}.bin'))

sup_head = MultiLabelHead(EMB_DIM, len(ALL_CATS)).to(DEVICE)
sup_head.load_state_dict(torch.load(str(MDL_DIR / f'supervised_head_{TAG}.pt'), map_location=DEVICE))
sup_head.eval()

with open(MDL_DIR / f'fusion_model_{TAG}.pkl', 'rb') as f:
    fusion = pickle.load(f)

# hierarchy graph (support both filenames)
_graph = {}
for gname in ('category_graph_v2.json', 'category_graph.json'):
    gp = DAT_DIR / gname
    if gp.exists():
        _graph = json.load(open(gp, encoding='utf-8')); break

_anc = {}
def _ancestors(cat):
    if cat in _anc:
        return _anc[cat]
    ancs, visited, stack = set(), {cat}, list(_graph.get(cat, []))
    while stack:
        p = stack.pop()
        if p in visited:
            continue
        visited.add(p); ancs.add(p)
        for gp in _graph.get(p, []):
            if gp not in visited:
                stack.append(gp)
    _anc[cat] = ancs
    return ancs
for c in ALL_CATS:
    _ancestors(c)

print(f'[app] ready: {len(ALL_CATS)} categories, emb_dim={EMB_DIM}, '
      f'index={faiss_index.ntotal} vectors')


# --------------------------------------------------------------------------- #
# Scorers + fusion + hierarchy (identical logic to the notebook)
# --------------------------------------------------------------------------- #
def knn_scores(q, K=20, tau=0.07):
    q = q.copy().astype('float32'); faiss.normalize_L2(q)
    sims, ids = faiss_index.search(q, K)
    scores = np.zeros((q.shape[0], train_labels.shape[1]), 'float32')
    for qi in range(q.shape[0]):
        exp_s = np.exp(sims[qi] / tau); wsum = exp_s.sum() + 1e-9
        for rank, ni in enumerate(ids[qi]):
            if ni >= 0:
                scores[qi] += exp_s[rank] * train_labels[ni]
        scores[qi] /= wsum
    return scores

def text_scores(q):
    return ((q @ cat_text_embs.T + 1.0) / 2.0).astype('float32')

@torch.no_grad()
def supervised_scores(q):
    return torch.sigmoid(sup_head(torch.from_numpy(q).to(DEVICE))).cpu().numpy().astype('float32')

def apply_fusion(knn, cls, txt, meta):
    model, mu, sigma = fusion['model'], fusion['mu'], fusion['sigma']
    stack = np.stack([knn, cls, txt, meta], axis=-1)
    n, nc, nf = stack.shape
    Xz = (stack.reshape(-1, nf).astype('float64') - mu) / sigma
    return model.predict_proba(Xz)[:, 1].reshape(n, nc).astype('float32')

def hierarchy_postprocess(scores, top_k=10, threshold=0.05):
    order = np.argsort(-scores)
    cand = [(ALL_CATS[i], float(scores[i])) for i in order[:top_k * 5]
            if float(scores[i]) >= threshold]
    if not cand:
        cand = [(ALL_CATS[i], float(scores[i])) for i in order[:top_k]]
    selected = {c for c, _ in cand}
    suppressed = set()
    for c, _ in cand:
        for a in _anc.get(c, set()):
            if a in selected:
                suppressed.add(a)
    final = sorted([(c, s) for c, s in cand if c not in suppressed], key=lambda x: -x[1])
    return final[:top_k]

def predict(pil_img, caption='', top_k=10):
    emb    = embed_image(pil_img)
    m_emb  = embed_texts([caption.strip() or 'image'])
    meta_s = ((m_emb @ cat_text_embs.T + 1) / 2).astype('float32')
    fused  = apply_fusion(knn_scores(emb), supervised_scores(emb), text_scores(emb), meta_s)
    return hierarchy_postprocess(fused[0], top_k=top_k)


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
app = Flask(__name__)

@app.get('/')
def index():
    return send_from_directory(HERE, 'index.html')

@app.get('/info')
def info():
    return jsonify(backbone=BACKBONE, device=DEVICE, n_categories=len(ALL_CATS),
                   emb_dim=EMB_DIM, index_size=int(faiss_index.ntotal))

@app.post('/predict')
def do_predict():
    if 'image' not in request.files or request.files['image'].filename == '':
        return jsonify(error='No image uploaded'), 400
    try:
        img = Image.open(io.BytesIO(request.files['image'].read())).convert('RGB')
    except Exception as e:
        return jsonify(error=f'Could not read image: {e}'), 400
    caption = request.form.get('caption', '')
    top_k   = int(request.form.get('top_k', 10))
    preds   = predict(img, caption=caption, top_k=top_k)
    return jsonify(backbone=BACKBONE,
                   predictions=[{'category': c, 'score': round(s, 4)} for c, s in preds])


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='127.0.0.1', port=port, debug=False)
