# Chapitre 10 — Optimisation et packaging du pipeline NLP

**Module NLP · Master Data/IA · MD5 Volet 2 · 2026**  
TP final — 2 heures 30

---

## Avant-propos : assembler le livrable final

Ce TP est le dernier acte technique du module. Il ne produit pas de nouveau modèle — les modèles sont entraînés depuis le Jour 2. Il prend ces modèles, les optimise pour la production, les emballe dans un service interrogeable, les teste rigoureusement, et produit les documents qui permettront à quelqu'un d'autre de reproduire exactement ce que vous avez fait dans six mois.

Le Chapitre 9 a exposé les principes : pourquoi quantiser, pourquoi containeriser, pourquoi surveiller. Ce chapitre est leur mise en œuvre concrète. À la fin de la séance, vous disposerez d'un dépôt GitHub contenant un pipeline dockerisé avec des tests automatisés — le livrable évalué à 25 % de la note finale.

Chaque section produit un artefact précis. L'ordre est imposé par les dépendances : vous ne pouvez pas tester ce que vous n'avez pas encore empaqueté, ni mesurer la dégradation F1 d'une quantisation que vous n'avez pas appliquée.

---

## 1. Quantisation INT8 du modèle NER et mesure F1 post-quantisation

### 1.1 Pourquoi quantiser en INT8 avant le déploiement

Le modèle NER fine-tuné au Chapitre 6 est stocké en FP16 ou FP32. En production, ce format est sous-optimal pour deux raisons. D'une part, l'empreinte mémoire est inutilement large : 220 Mo en FP16 pour CamemBERT-base, contre 110 Mo en INT8. D'autre part, les opérations entières (INT8) sont exécutées sur des unités matérielles plus nombreuses que les opérations flottantes sur la plupart des GPUs et des CPUs modernes.

La quantisation INT8 appliquée à un modèle de classification de tokens comme CamemBERT-NER produit en général une dégradation de F1 inférieure à 1 point, ce qui est acceptable en production. La mesure rigoureuse de cette dégradation — sur le même jeu de test que lors de l'entraînement — est une exigence de traçabilité scientifique, pas une option.

### 1.2 Application de la quantisation

```python
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from peft import PeftModel

# ── Chargement du modèle fine-tuné ──────────────────────────────────────
MODEL_NAME     = "almanach/camembert-base"
CHECKPOINT_DIR = "./ner_lora_r8"   # checkpoint Jour 3

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Charger le modèle de base avec la tête de classification
base_model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels = 11,   # schéma BIO 5 types
)
# Fusionner les adaptateurs LoRA dans les poids du modèle de base
model_lora = PeftModel.from_pretrained(base_model, CHECKPOINT_DIR)
model_fp16 = model_lora.merge_and_unload()   # fusion LoRA → poids denses
model_fp16.eval()

print(f"Modèle FP16 : "
      f"{sum(p.numel() for p in model_fp16.parameters()) / 1e6:.1f}M paramètres")

# ── Quantisation dynamique INT8 ──────────────────────────────────────────
model_int8 = torch.quantization.quantize_dynamic(
    model_fp16,
    {torch.nn.Linear},   # quantiser toutes les couches linéaires
    dtype=torch.qint8,
)

# Taille effective (approximation par comptage des paramètres)
def model_size_mb(model: torch.nn.Module) -> float:
    """Estimation de la taille du modèle en Mo (poids only, FP32 équivalent)."""
    n_params = sum(p.numel() for p in model.parameters())
    return n_params * 4 / 1e6   # 4 octets par paramètre FP32

print(f"Taille modèle FP16  : ~{model_size_mb(model_fp16) / 2:.0f} Mo")
print(f"Taille modèle INT8  : ~{model_size_mb(model_int8) / 4:.0f} Mo")
```

**Alternative avec bitsandbytes :** `bitsandbytes` permet une quantisation INT8 au moment du chargement, sans modifier les poids stockés. Elle est recommandée si la VRAM est contrainte et si l'on veut éviter de sauvegarder deux versions du modèle :

```python
from transformers import AutoModelForTokenClassification, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(load_in_8bit=True)
model_bnb8 = AutoModelForTokenClassification.from_pretrained(
    CHECKPOINT_DIR,
    quantization_config = bnb_config,
    device_map          = "auto",
)
```

### 1.3 Mesure du F1 post-quantisation

La mesure du F1 doit s'effectuer sur le **même jeu de test** que lors de l'évaluation du modèle FP16. Utiliser le `split_hash` du Chapitre 4 garantit que les deux évaluations sont comparables.

```python
from seqeval.metrics import f1_score
from seqeval.scheme  import IOB2
import json

def evaluate_ner_model(model,
                         tokenizer,
                         test_corpus: list[dict],
                         id2label:    dict,
                         batch_size:  int = 8) -> dict:
    """
    Évalue un modèle NER (FP16 ou INT8) sur le corpus de test.
    Retourne les métriques seqeval (F1 micro/macro par type).
    """
    model.eval()
    y_true, y_pred = [], []

    for i in range(0, len(test_corpus), batch_size):
        batch   = test_corpus[i : i + batch_size]
        texts   = [" ".join(r["normalized"].split()) for r in batch]
        labels  = [r.get("ner_labels", []) for r in batch]

        inputs  = tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=128,
            is_split_into_words=False,
        )
        with torch.no_grad():
            logits = model(**inputs).logits    # (batch, seq, n_labels)

        preds = logits.argmax(dim=-1).tolist()
        # Aligner les prédictions sur les tokens originaux
        for j, (pred_seq, true_labels) in enumerate(zip(preds, labels)):
            word_ids = inputs.word_ids(batch_index=j)
            pred_labels, true_out = [], []
            prev_wid = None
            for wid, pid in zip(word_ids, pred_seq):
                if wid is None or wid == prev_wid:
                    prev_wid = wid
                    continue
                pred_labels.append(id2label[pid])
                prev_wid = wid
            # Aligner sur la longueur réelle
            min_len = min(len(pred_labels), len(true_labels))
            y_pred.append(pred_labels[:min_len])
            y_true.append(true_labels[:min_len])

    from seqeval.metrics import classification_report
    report = classification_report(
        y_true, y_pred, scheme=IOB2, output_dict=True, zero_division=0
    )
    return {
        "f1_micro":   round(f1_score(y_true, y_pred, average="micro",
                                      scheme=IOB2, zero_division=0), 4),
        "f1_macro":   round(f1_score(y_true, y_pred, average="macro",
                                      scheme=IOB2, zero_division=0), 4),
        "per_entity": {k: v for k, v in report.items()
                       if k not in ("micro avg","macro avg","weighted avg")},
    }

# Évaluer les deux versions
test_set = [r for r in enriched_corpus_v2 if "test" in r.get("split","")]

metrics_fp16 = evaluate_ner_model(model_fp16, tokenizer, test_set, ID2LABEL)
metrics_int8 = evaluate_ner_model(model_int8, tokenizer, test_set, ID2LABEL)

print(f"\nF1 post-quantisation :")
print(f"  FP16 : micro={metrics_fp16['f1_micro']:.4f}  macro={metrics_fp16['f1_macro']:.4f}")
print(f"  INT8 : micro={metrics_int8['f1_micro']:.4f}  macro={metrics_int8['f1_macro']:.4f}")
print(f"  Dégradation : {(metrics_fp16['f1_micro']-metrics_int8['f1_micro'])*100:.2f} pts")

# Sauvegarder les métriques de comparaison
with open("quantization_report.json", "w", encoding="utf-8") as f:
    json.dump({
        "split_hash":   SPLIT_HASH,
        "fp16":         metrics_fp16,
        "int8":         metrics_int8,
        "f1_delta":     round((metrics_fp16["f1_micro"] -
                               metrics_int8["f1_micro"]) * 100, 3),
        "acceptable":   (metrics_fp16["f1_micro"] -
                         metrics_int8["f1_micro"]) * 100 < 2.0,
    }, f, indent=2, ensure_ascii=False)
print("Livrable : quantization_report.json")
```

**Critère d'acceptabilité :** la dégradation INT8 doit rester en dessous de 2 points de F1 micro. Au-delà, utiliser la quantisation FP16 simple (sans INT8) pour ce modèle spécifique. Typiquement, pour CamemBERT-NER entraîné sur des données médiévales, la dégradation observée est de 0.3–0.8 points.

---

## 2. Benchmark de latence avant/après quantisation

### 2.1 Protocole de benchmark

Un benchmark de latence rigoureux requiert trois précautions. Premièrement, un **warm-up** de 10–20 requêtes avant la mesure, pour que les poids soient chargés en cache GPU/CPU et que les compilations JIT de PyTorch soient effectuées. Deuxièmement, un nombre de requêtes suffisant (100 minimum) pour que les percentiles soient stables. Troisièmement, une **distribution de textes représentative** — ni toujours la même phrase courte (cache trop favorable), ni des phrases hors de la distribution d'entraînement.

```python
import time, statistics
import numpy as np

# Corpus de benchmark : 100 phrases variées en longueur
BENCHMARK_TEXTS = [
    f"li sénéchal jean de normandie porta les lettres au roi {i}"
    for i in range(25)
] + [
    f"le duc charles de france régna vingt ans {i}" for i in range(25)
] + [
    f"l abbaye de saint denis reçut les terres de {i}" for i in range(25)
] + [
    f"en mars le roi convoqua ses vassaux {i}" for i in range(25)
]

def run_benchmark(model,
                   tokenizer,
                   texts:        list[str],
                   n_warmup:     int = 10,
                   batch_size:   int = 1) -> dict:
    """
    Benchmark de latence sur un ensemble de textes.
    Retourne p50, p90, p99, mean, throughput (req/s).
    """
    model.eval()

    # Warm-up
    for text in texts[:n_warmup]:
        inputs = tokenizer(text, return_tensors="pt",
                            max_length=128, truncation=True)
        with torch.no_grad():
            _ = model(**inputs).logits

    # Mesure
    latencies = []
    t_wall    = time.perf_counter()

    for text in texts:
        t0 = time.perf_counter()
        inputs = tokenizer(text, return_tensors="pt",
                            max_length=128, truncation=True)
        with torch.no_grad():
            _ = model(**inputs).logits
        latencies.append((time.perf_counter() - t0) * 1000)

    total_s = time.perf_counter() - t_wall
    lat     = sorted(latencies)

    return {
        "n_requests":   len(lat),
        "p50_ms":       round(np.percentile(lat, 50), 2),
        "p90_ms":       round(np.percentile(lat, 90), 2),
        "p99_ms":       round(np.percentile(lat, 99), 2),
        "mean_ms":      round(statistics.mean(lat), 2),
        "std_ms":       round(statistics.stdev(lat), 2),
        "throughput":   round(len(lat) / total_s, 1),
    }

bench_fp16 = run_benchmark(model_fp16, tokenizer, BENCHMARK_TEXTS)
bench_int8 = run_benchmark(model_int8, tokenizer, BENCHMARK_TEXTS)

print(f"\nBenchmark de latence (100 requêtes, batch_size=1) :")
print(f"{'Métrique':12s}  {'FP16':>8s}  {'INT8':>8s}  {'Gain':>8s}")
print(f"{'─'*42}")
for k in ("p50_ms","p90_ms","p99_ms","mean_ms","throughput"):
    unit = " req/s" if "throughput" in k else " ms"
    gain = (bench_fp16[k] / bench_int8[k] if bench_int8[k] > 0
            else float("inf"))
    # throughput : gain est un facteur multiplicatif
    if "throughput" in k:
        gain_str = f"+{(gain-1)*100:.0f}%"
    else:
        gain_str = f"−{(1-1/gain)*100:.0f}%"
    print(f"  {k:12s}  {bench_fp16[k]:>6.1f}{unit}  "
          f"{bench_int8[k]:>6.1f}{unit}  {gain_str:>8s}")

# Sauvegarder le benchmark
with open("benchmark_report.json", "w", encoding="utf-8") as f:
    json.dump({
        "split_hash": SPLIT_HASH,
        "fp16":       bench_fp16,
        "int8":       bench_int8,
    }, f, indent=2, ensure_ascii=False)
print("\nLivrable : benchmark_report.json")
```

**Résultats typiques sur T4 (CamemBERT-NER, séquence 128 tokens) :**

| Métrique | FP16 | INT8 | Gain |
|---|---|---|---|
| p50 | 48 ms | 28 ms | −42 % |
| p90 | 67 ms | 38 ms | −43 % |
| p99 | 79 ms | 50 ms | −37 % |
| throughput | 18 req/s | 32 req/s | +78 % |

---

## 3. Packaging : FastAPI + Docker

### 3.1 Application FastAPI complète

Le service expose deux endpoints principaux (`/transcribe` et `/analyze`) plus les endpoints d'infrastructure (`/health`, `/metrics`).

```python
# src/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import time, json, logging, os

# ── Configuration ────────────────────────────────────────────────────────
MODEL_PATH    = os.environ.get("MODEL_PATH",    "./models/ner_lora_r8")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "1.0.0")
SPLIT_HASH    = os.environ.get("SPLIT_HASH",    "unknown")

logger = logging.getLogger("nlp_pipeline")
logging.basicConfig(
    level   = logging.INFO,
    format  = '{"ts":"%(asctime)s","level":"%(levelname)s","msg":%(message)s}',
)

app = FastAPI(
    title       = "Pipeline NLP médiéval",
    description = "Normalisation HTR + NER sur manuscrits médiévaux (CREMMA)",
    version     = MODEL_VERSION,
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# ── Modèles Pydantic ─────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    text:       str   = Field(..., min_length=1, max_length=2000,
                              description="Transcription HTR brute ou texte normalisé")
    line_id:    str   = Field("", description="Identifiant dans le data contract")
    confidence: float = Field(1.0, ge=0.0, le=1.0,
                              description="Score de confiance HTR de la ligne")

class EntitySpan(BaseModel):
    start: int
    end:   int
    label: str   # BIO type : PER, LOC, DATE, ORG, TITLE
    text:  str

class AnalyzeResponse(BaseModel):
    line_id:       str
    normalized:    str
    ner_spans:     list[EntitySpan]
    pos_tags:      list[str]
    lemmas:        list[str]
    model_version: str
    split_hash:    str
    latency_ms:    float

# ── Chargement du pipeline au démarrage ──────────────────────────────────
_pipeline = None

@app.on_event("startup")
async def load_pipeline():
    global _pipeline
    from src.pipeline import NERPipeline
    _pipeline = NERPipeline(MODEL_PATH)
    logger.info(json.dumps({
        "event":         "model_loaded",
        "model_path":    MODEL_PATH,
        "model_version": MODEL_VERSION,
        "split_hash":    SPLIT_HASH,
    }))

# ── Endpoints ────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Healthcheck : retourne le statut et la version du modèle."""
    return {
        "status":        "ok",
        "model_version": MODEL_VERSION,
        "split_hash":    SPLIT_HASH,
    }

@app.get("/metrics")
def metrics():
    """Métriques de performance pour Prometheus."""
    if _pipeline is None:
        return {"status": "not_loaded"}
    return _pipeline.get_metrics()

@app.post("/transcribe", response_model=AnalyzeResponse)
def transcribe(req: AnalyzeRequest) -> AnalyzeResponse:
    """
    Normalise une transcription HTR sans NER.
    Plus rapide que /analyze ; à utiliser quand les entités ne sont pas nécessaires.
    """
    t0         = time.perf_counter()
    normalized = _pipeline.normalize(req.text)
    tokens     = normalized.split()
    latency_ms = (time.perf_counter() - t0) * 1000

    logger.info(json.dumps({
        "endpoint":   "transcribe",
        "line_id":    req.line_id,
        "input_len":  len(req.text),
        "latency_ms": round(latency_ms, 2),
    }))
    return AnalyzeResponse(
        line_id       = req.line_id,
        normalized    = normalized,
        ner_spans     = [],
        pos_tags      = ["X"] * len(tokens),
        lemmas        = tokens,
        model_version = MODEL_VERSION,
        split_hash    = SPLIT_HASH,
        latency_ms    = round(latency_ms, 2),
    )

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    """
    Normalise une transcription HTR et extrait les entités nommées.
    Pipeline complet : normalisation → NER → data contract.
    """
    t0     = time.perf_counter()
    result = _pipeline.run(req.text, req.line_id, req.confidence)
    latency_ms = (time.perf_counter() - t0) * 1000

    logger.info(json.dumps({
        "endpoint":    "analyze",
        "line_id":     req.line_id,
        "input_len":   len(req.text),
        "n_entities":  len(result["ner_spans"]),
        "entity_types":[s["label"] for s in result["ner_spans"]],
        "latency_ms":  round(latency_ms, 2),
        "confidence":  req.confidence,
    }))
    return AnalyzeResponse(
        **{k: result[k] for k in ("line_id","normalized","ner_spans",
                                    "pos_tags","lemmas","model_version",
                                    "split_hash")},
        latency_ms = round(latency_ms, 2),
    )
```

### 3.2 Dockerfile reproductible

```dockerfile
# Dockerfile
# ── Étape 1 : image de base Python slim ─────────────────────────────────
FROM python:3.11-slim AS base

WORKDIR /app

# Dépendances système minimales
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# ── Étape 2 : installation des dépendances Python ───────────────────────
FROM base AS deps

COPY requirements.txt .
# Installer sans cache, avec hashes pour la reproductibilité
RUN pip install --no-cache-dir --require-hashes -r requirements.txt

# ── Étape 3 : image finale ───────────────────────────────────────────────
FROM base AS final

# Copier les dépendances installées
COPY --from=deps /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=deps /usr/local/bin            /usr/local/bin

# Copier le code et les artefacts du modèle
COPY src/                   ./src/
COPY models/                ./models/
COPY MODEL_CARD.json        .
COPY CONVENTIONS_NLP.md     .
COPY DATA_SOURCES.md        .

# Variables d'environnement
ENV MODEL_PATH=/app/models/ner_lora_r8
ENV MODEL_VERSION=1.0.0
ENV SPLIT_HASH=76aff95784fd4d4e
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

EXPOSE 8000

# Healthcheck : vérifie l'endpoint /health toutes les 30 s
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Lancer avec un seul worker (un modèle par processus)
CMD ["uvicorn", "src.main:app",
     "--host",    "0.0.0.0",
     "--port",    "8000",
     "--workers", "1",
     "--log-level", "info"]
```

```
# requirements.txt (avec hashes SHA-256 pour la reproductibilité)
# Générer avec : pip-compile --generate-hashes requirements.in
fastapi==0.111.0 \
    --hash=sha256:...
uvicorn[standard]==0.29.0 \
    --hash=sha256:...
transformers==4.41.0 \
    --hash=sha256:...
peft==0.11.1 \
    --hash=sha256:...
torch==2.3.0 \
    --hash=sha256:...
bitsandbytes==0.43.0 \
    --hash=sha256:...
seqeval==1.2.2 \
    --hash=sha256:...
editdistance==0.8.1 \
    --hash=sha256:...
networkx==3.3 \
    --hash=sha256:...
lxml==5.2.1 \
    --hash=sha256:...
pydantic==2.7.1 \
    --hash=sha256:...
```

**Construction et test local :**

```bash
# Construire l'image
docker build -t nlp-medieval:1.0.0 .

# Lancer le service
docker run -p 8000:8000 \
    -e MODEL_VERSION=1.0.0 \
    -e SPLIT_HASH=76aff95784fd4d4e \
    nlp-medieval:1.0.0

# Test rapide
curl -X POST http://localhost:8000/analyze \
    -H "Content-Type: application/json" \
    -d '{"text":"li sénéchal jean de normandie porta les lettres","line_id":"test_001"}'
```

---

## 4. Tests pytest : schéma, non-régression F1, temps de réponse

### 4.1 Structure du dossier de tests

```
tests/
├── conftest.py          # fixtures : client TestClient, corpus de test
├── test_schema.py       # schéma JSON de sortie des endpoints
├── test_regression.py   # non-régression F1 sur le corpus de test
├── test_latency.py      # contraintes de temps de réponse
└── test_invariants.py   # idempotence, split_hash, model_version
```

### 4.2 `conftest.py` — Fixtures partagées

```python
# tests/conftest.py
import pytest, json
from fastapi.testclient import TestClient
from src.main import app

@pytest.fixture(scope="session")
def client():
    """Client de test FastAPI, chargé une seule fois pour la session."""
    with TestClient(app) as c:
        yield c

@pytest.fixture(scope="session")
def test_corpus():
    """Corpus de test fixe — correspond au split SHA-256 du Jour 2."""
    return [
        {
            "text":    "li sénéchal jean de normandie porta les lettres",
            "line_id": "charte_001_l01",
            "gold":    [("sénéchal","TITLE"),
                        ("jean de normandie","PER")],
        },
        {
            "text":    "le roi philippe signa l acte en mars",
            "line_id": "charte_001_l02",
            "gold":    [("roi","TITLE"),
                        ("philippe","PER"),
                        ("mars","DATE")],
        },
        {
            "text":    "l abbaye de saint denis reçut les terres",
            "line_id": "charte_002_l01",
            "gold":    [("abbaye de saint denis","ORG")],
        },
        {
            "text":    "le prévôt de paris rendit son jugement",
            "line_id": "charte_002_l02",
            "gold":    [("prévôt","TITLE"), ("paris","LOC")],
        },
        {
            "text":    "marguerite de flandre hérita du comté",
            "line_id": "charte_003_l01",
            "gold":    [("marguerite de flandre","PER")],
        },
    ]

@pytest.fixture(scope="session")
def expected_schema():
    """Schéma de validation pour les réponses /analyze."""
    return {
        "required_keys":      {"line_id","normalized","ner_spans","pos_tags",
                                "lemmas","model_version","split_hash","latency_ms"},
        "span_required_keys": {"start","end","label","text"},
        "valid_labels":       {"B-PER","I-PER","B-LOC","I-LOC","B-DATE",
                                "I-DATE","B-ORG","I-ORG","B-TITLE","I-TITLE","O"},
    }
```

### 4.3 `test_schema.py` — Validation du schéma JSON

```python
# tests/test_schema.py
import pytest

def test_analyze_required_keys(client, expected_schema):
    """Toutes les clés requises sont présentes dans la réponse."""
    resp = client.post("/analyze",
                       json={"text": "le sénéchal porta les lettres"})
    assert resp.status_code == 200
    data = resp.json()
    missing = expected_schema["required_keys"] - set(data.keys())
    assert not missing, f"Clés manquantes : {missing}"

def test_analyze_span_schema(client, expected_schema):
    """Chaque span NER a les clés requises et des offsets cohérents."""
    resp = client.post("/analyze",
                       json={"text": "jean de normandie porta les lettres"})
    assert resp.status_code == 200
    data = resp.json()
    for span in data["ner_spans"]:
        missing = expected_schema["span_required_keys"] - set(span.keys())
        assert not missing, f"Clés span manquantes : {missing}"
        # Vérification des offsets caractères
        extracted = data["normalized"][span["start"]:span["end"]]
        assert extracted == span["text"], (
            f"Offset incohérent : [{span['start']}:{span['end']}] = {extracted!r}"
            f" ≠ {span['text']!r}"
        )

def test_analyze_valid_ner_labels(client, expected_schema):
    """Toutes les étiquettes NER sont dans le vocabulaire BIO."""
    resp = client.post("/analyze",
                       json={"text": "le roi signa en normandie en mars"})
    assert resp.status_code == 200
    for span in resp.json()["ner_spans"]:
        label_with_prefix = f"B-{span['label']}"
        assert span["label"] in {l[2:] for l in expected_schema["valid_labels"]
                                   if l.startswith("B-")}, \
            f"Étiquette NER invalide : {span['label']}"

def test_transcribe_no_ner_spans(client):
    """L'endpoint /transcribe ne retourne jamais d'entités."""
    resp = client.post("/transcribe",
                       json={"text": "le roi jean signa en normandie"})
    assert resp.status_code == 200
    assert resp.json()["ner_spans"] == [], \
        "/transcribe ne doit pas retourner d'entités"

def test_health_schema(client):
    """/health retourne status, model_version et split_hash."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "model_version" in data
    assert "split_hash" in data

def test_input_too_long_returns_422(client):
    """Un texte dépassant 2000 caractères retourne HTTP 422."""
    resp = client.post("/analyze", json={"text": "a" * 2001})
    assert resp.status_code == 422

def test_empty_text_returns_422(client):
    """Un texte vide retourne HTTP 422."""
    resp = client.post("/analyze", json={"text": ""})
    assert resp.status_code == 422
```

### 4.4 `test_regression.py` — Non-régression F1

```python
# tests/test_regression.py
import pytest

MIN_F1_MICRO = 0.60   # seuil de non-régression (Chapitre 9, tableau §7)
SPLIT_HASH_EXPECTED = "76aff95784fd4d4e"

def _compute_f1(predictions: list[dict], corpus: list[dict]) -> dict:
    """Calcule F1 span-level sur le corpus de test."""
    tp = fp = fn = 0
    for pred, record in zip(predictions, corpus):
        pred_spans = {(s["text"].lower(), s["label"])
                      for s in pred["ner_spans"]}
        gold_spans = {(e[0].lower(), e[1]) for e in record["gold"]}
        tp += len(pred_spans & gold_spans)
        fp += len(pred_spans - gold_spans)
        fn += len(gold_spans - pred_spans)
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"precision": round(prec, 3),
            "recall":    round(rec, 3),
            "f1":        round(f1, 3)}


def test_f1_non_regression(client, test_corpus):
    """
    F1 micro sur le corpus de test ≥ MIN_F1_MICRO.
    Ce test échoue si une modification du modèle ou du pipeline
    dégrade les performances en dessous du seuil défini.
    """
    predictions = []
    for record in test_corpus:
        resp = client.post("/analyze", json={
            "text":    record["text"],
            "line_id": record["line_id"],
        })
        assert resp.status_code == 200, \
            f"Erreur sur {record['line_id']}: {resp.text}"
        predictions.append(resp.json())

    metrics = _compute_f1(predictions, test_corpus)
    assert metrics["f1"] >= MIN_F1_MICRO, (
        f"Régression F1 détectée : {metrics['f1']:.3f} < {MIN_F1_MICRO}. "
        f"Détail : P={metrics['precision']:.3f}, R={metrics['recall']:.3f}"
    )


def test_split_hash_matches(client):
    """
    Le split_hash retourné correspond au hash du split d'entraînement.
    Garantit que le modèle déployé est bien celui entraîné sur le bon split.
    """
    resp = client.post("/analyze",
                       json={"text": "le sénéchal porta les lettres"})
    assert resp.status_code == 200
    assert resp.json()["split_hash"] == SPLIT_HASH_EXPECTED, (
        f"split_hash inattendu : {resp.json()['split_hash']!r} "
        f"≠ {SPLIT_HASH_EXPECTED!r}"
    )


def test_per_entity_f1(client, test_corpus):
    """
    F1 par type d'entité — valeurs indicatives (non bloquantes).
    Avertit si une catégorie chute sous un seuil spécifique.
    """
    predictions = [
        client.post("/analyze", json={"text": r["text"]}).json()
        for r in test_corpus
    ]
    # Calcul par type
    by_type = {}
    for pred, rec in zip(predictions, test_corpus):
        pred_spans = {(s["text"].lower(), s["label"])
                      for s in pred["ner_spans"]}
        gold_spans = {(e[0].lower(), e[1]) for e in rec["gold"]}
        for _, label in gold_spans | pred_spans:
            if label not in by_type:
                by_type[label] = {"tp":0,"fp":0,"fn":0}
            if (_, label) in pred_spans & gold_spans:
                by_type[label]["tp"] += 1
            elif (_, label) in pred_spans - gold_spans:
                by_type[label]["fp"] += 1
            else:
                by_type[label]["fn"] += 1

    for ent_type, counts in by_type.items():
        p = counts["tp"] / max(counts["tp"]+counts["fp"], 1)
        r = counts["tp"] / max(counts["tp"]+counts["fn"], 1)
        f = 2*p*r / max(p+r, 1e-9)
        # Avertissement (pas d'assertion) si F1 < 0.5 sur un type
        if f < 0.50:
            import warnings
            warnings.warn(
                f"F1 faible pour {ent_type}: {f:.2f} "
                f"(P={p:.2f}, R={r:.2f})"
            )
```

### 4.5 `test_latency.py` — Contraintes de temps de réponse

```python
# tests/test_latency.py
import pytest, time
import numpy as np

P99_MAX_MS     = 500.0   # seuil p99 acceptable (ms)
N_REQUESTS     = 100     # nombre de requêtes pour le benchmark
N_WARMUP       = 10      # requêtes de warm-up non mesurées

LATENCY_TEXTS = [
    f"le sénéchal jean de normandie porta les lettres {i}"
    for i in range(N_REQUESTS)
]

def test_latency_p99(client):
    """
    La latence p99 sur 100 requêtes séquentielles est inférieure à P99_MAX_MS.
    Reproduit le benchmark du Chapitre 9 dans un test automatisé.
    """
    # Warm-up
    for text in LATENCY_TEXTS[:N_WARMUP]:
        client.post("/analyze", json={"text": text})

    # Mesure
    latencies = []
    for text in LATENCY_TEXTS:
        t0 = time.perf_counter()
        resp = client.post("/analyze", json={"text": text})
        latencies.append((time.perf_counter() - t0) * 1000)
        assert resp.status_code == 200

    p50 = np.percentile(latencies, 50)
    p99 = np.percentile(latencies, 99)

    assert p99 < P99_MAX_MS, (
        f"Latence p99 trop élevée : {p99:.1f}ms > {P99_MAX_MS}ms. "
        f"p50={p50:.1f}ms. Vérifier la quantisation et le batch size."
    )


def test_latency_p99_ratio(client):
    """
    Le ratio p99/p50 est inférieur à 3 (absence de queue longue).
    Un ratio > 3 indique une contention de ressources ou un GC intempestif.
    """
    latencies = []
    for text in LATENCY_TEXTS[:50]:
        t0 = time.perf_counter()
        client.post("/analyze", json={"text": text})
        latencies.append((time.perf_counter() - t0) * 1000)

    p50 = np.percentile(latencies, 50)
    p99 = np.percentile(latencies, 99)
    ratio = p99 / max(p50, 1e-9)

    assert ratio < 3.0, (
        f"Queue longue détectée : p99/p50 = {ratio:.2f} > 3.0. "
        f"p50={p50:.1f}ms, p99={p99:.1f}ms."
    )
```

### 4.6 `test_invariants.py` — Propriétés invariantes

```python
# tests/test_invariants.py
import pytest

def test_idempotence(client):
    """
    Deux appels identiques à /analyze produisent exactement la même sortie.
    Garantit que le pipeline est déterministe (pas d'état mutable).
    """
    payload = {
        "text":       "le roi signa l acte en normandie",
        "line_id":    "test_idem",
        "confidence": 0.90,
    }
    r1 = client.post("/analyze", json=payload).json()
    r2 = client.post("/analyze", json=payload).json()

    assert r1["normalized"]  == r2["normalized"],  "Normalisation non idempotente"
    assert r1["ner_spans"]   == r2["ner_spans"],    "NER non idempotente"
    assert r1["model_version"] == r2["model_version"], "Version inconsistante"
    assert r1["split_hash"]  == r2["split_hash"],   "Split hash inconsistant"
    # La latence peut varier — ne pas la comparer


def test_span_offsets_invariant(client):
    """
    Pour tout texte, normalized[span.start:span.end] == span.text.
    Vérifié sur une liste de textes variés.
    """
    texts = [
        "li sénéchal jean de normandie porta les lettres",
        "le duc charles de france régna vingt ans",
        "l abbaye de saint denis reçut les terres en juillet",
        "marguerite de flandre hérita du comté de bourgogne",
    ]
    for text in texts:
        resp = client.post("/analyze", json={"text": text})
        assert resp.status_code == 200
        data = resp.json()
        norm = data["normalized"]
        for span in data["ner_spans"]:
            extracted = norm[span["start"]:span["end"]]
            assert extracted == span["text"], (
                f"Offset invalide pour '{text}': "
                f"[{span['start']}:{span['end']}] = {extracted!r} "
                f"≠ {span['text']!r}"
            )


def test_pos_lemmas_length_match(client):
    """
    Les listes pos_tags et lemmas ont la même longueur que normalized.split().
    Garantit la cohérence du data contract v2.
    """
    texts = [
        "le roi porta les lettres en mars",
        "li sénéchal jean signa",
    ]
    for text in texts:
        resp = client.post("/analyze", json={"text": text})
        assert resp.status_code == 200
        data = resp.json()
        n_tokens = len(data["normalized"].split())
        assert len(data["pos_tags"]) == n_tokens, \
            f"pos_tags ({len(data['pos_tags'])}) ≠ tokens ({n_tokens})"
        assert len(data["lemmas"]) == n_tokens, \
            f"lemmas ({len(data['lemmas'])}) ≠ tokens ({n_tokens})"
```

### 4.7 Exécuter la suite de tests

```bash
# Depuis la racine du projet
pytest tests/ -v --tb=short

# Avec couverture de code
pytest tests/ --cov=src --cov-report=term-missing --cov-report=html

# Uniquement les tests de non-régression (CI rapide)
pytest tests/test_regression.py tests/test_invariants.py -v

# Exclure les tests de latence (trop lents pour la CI)
pytest tests/ -v --ignore=tests/test_latency.py
```

---

## 5. Finalisation des documents de livraison

### 5.1 MODEL_CARD.json

La model card documente ce que fait le modèle, sur quelles données il a été entraîné, et ses limitations connues. Elle est obligatoire pour un dépôt public.

```python
import json, datetime, hashlib

model_card = {
    "model_id":          "nlp-medieval-ner-v1.0",
    "base_model":        "almanach/camembert-base",
    "task":              "token-classification (NER)",
    "language":          "fr-medieval",
    "created_at":        datetime.date.today().isoformat(),

    "training": {
        "method":           "LoRA (r=8, alpha=16, target=query+value)",
        "split_hash":       SPLIT_HASH,   # SHA-256 du split Jour 2
        "corpus":           "CREMMA Medieval, chartes normandes XIVe–XVe s.",
        "n_training_pairs": 28,
        "n_epochs":         20,
        "learning_rate":    2e-4,
    },

    "evaluation": {
        "split":             "test (SHA-256: " + SPLIT_HASH + ")",
        "metrics": {
            "f1_micro":  0.818,
            "f1_macro":  0.797,
            "per_entity": {
                "PER":   {"f1": 0.818, "support": 57},
                "LOC":   {"f1": 0.912, "support": 69},
                "DATE":  {"f1": 0.889, "support": 32},
                "ORG":   {"f1": 0.610, "support": 32},
                "TITLE": {"f1": 0.756, "support": 40},
            },
        },
        "quantization_int8": {
            "f1_micro": 0.812,
            "f1_delta": -0.6,
        },
    },

    "limitations": [
        "F1-ORG faible (0.61) : confusion LOC/ORG sur institutions localisées.",
        "Prénoms féminins sous-représentés dans les données d'entraînement.",
        "Micro-toponymes ruraux non couverts par le gazetier initial.",
        "Hors-domaine : textes antérieurs à 1300 ou postérieurs à 1500.",
        "Corpus d'entraînement limité à 40 phrases — biais de généralisation.",
    ],

    "bias_analysis": (
        "Corpus biaisé vers la noblesse masculine normande. "
        "Performance attendue plus faible sur personnes féminines "
        "et sur documents gascons ou picards. "
        "Voir Chapitre 5 §7 pour l'analyse complète."
    ),

    "intended_use": [
        "Annotation NER de transcriptions HTR de chartes médiévales françaises.",
        "Construction de bases de connaissances patrimoniales.",
        "Recherche académique en humanités numériques.",
    ],

    "out_of_scope": [
        "Textes français modernes.",
        "Production commerciale sans validation supplémentaire.",
        "Traitements nécessitant une précision > 90% sur ORG.",
    ],
}

with open("MODEL_CARD.json", "w", encoding="utf-8") as f:
    json.dump(model_card, f, indent=2, ensure_ascii=False)
print("Livrable : MODEL_CARD.json")
```

### 5.2 DATA_SOURCES.md

```markdown
# DATA_SOURCES.md

## Données d'entraînement

### CREMMA Medieval
- **Source :** https://github.com/HTR-United/cremma-medieval
- **Licence :** CC BY 4.0
- **Période :** XIVe–XVe siècle, chartes et registres normands
- **Volume utilisé :** 40 lignes sélectionnées (split SHA-256: 76aff95784fd4d4e)
- **Normalisation :** pipeline règles + DMF + mT5-LoRA (Chapitre 4)

### DMF — Dictionnaire du Moyen Français
- **Source :** http://www.atilf.fr/dmf (ATILF, CNRS – Université de Lorraine)
- **Usage :** lemmatisation des formes médiévales (cache local JSON)
- **Conditions :** consultation pour usage académique

### Universal Dependencies — Old French
- **Source :** https://universaldependencies.org/treebanks/fro_srcmf/
- **Licence :** CC BY-SA 3.0
- **Usage :** référence pour les étiquettes POS UD

## Modèles pré-entraînés

### CamemBERT-base
- **Source :** https://huggingface.co/almanach/camembert-base
- **Licence :** MIT
- **Usage :** modèle de fondation pour le fine-tuning NER

## Reproductibilité

Le split train/val/test est identifié par le hash SHA-256 :
`76aff95784fd4d4e...` (voir `experiments/journal.jsonl`).

Toute expérience reproduisant ce pipeline doit utiliser ce hash
pour garantir la comparabilité des résultats.
```

### 5.3 README.md

```markdown
# Pipeline NLP médiéval — MD5 Volet 2 2026

Normalisation HTR + NER sur manuscrits médiévaux français (CREMMA).

## Installation rapide

```bash
git clone https://github.com/[votre-groupe]/nlp-medieval
cd nlp-medieval
pip install -r requirements.txt
```

## Utilisation

### Via Docker (recommandé)
```bash
docker pull ghcr.io/[votre-groupe]/nlp-medieval:1.0.0
docker run -p 8000:8000 ghcr.io/[votre-groupe]/nlp-medieval:1.0.0
```

### Endpoint /analyze
```bash
curl -X POST http://localhost:8000/analyze \
    -H "Content-Type: application/json" \
    -d '{"text": "li sénéchal jean de normandie porta les lettres"}'
```

## Performances
| Métrique | Valeur |
|---|---|
| CER (normalisation) | 4.1 % |
| F1-NER micro | 0.818 |
| Latence p50 (INT8, T4) | 28 ms |
| Latence p99 (INT8, T4) | 50 ms |

## Structure du dépôt
```
├── src/               # Code Python du pipeline
├── models/            # Checkpoints LoRA
├── tests/             # Suite pytest
├── tei_output/        # Export TEI-XML (Jour 4)
├── enriched_corpus_v2.jsonl  # Data contract NLP v2
├── MODEL_CARD.json    # Fiche descriptive du modèle
├── DATA_SOURCES.md    # Provenance des données
├── CONVENTIONS_NLP.md # Décisions de normalisation
└── Dockerfile         # Image reproductible
```

## Tests
```bash
pytest tests/ -v
```

## Citation
Si vous utilisez ce pipeline dans une publication, citez :
> [Votre groupe], *Pipeline NLP pour manuscrits médiévaux*, 2026.
> SHA-256 split : 76aff95784fd4d4e.
```

### 5.4 Export vers HuggingFace Hub

```python
from huggingface_hub import HfApi, create_repo
import os

def push_to_huggingface(model,
                          tokenizer,
                          repo_name:    str,
                          model_card:   dict,
                          private:      bool = True) -> str:
    """
    Publie le modèle sur HuggingFace Hub.
    Le dépôt privé est recommandé pendant la recherche ;
    passer à public lors de la publication de l'article.
    """
    api = HfApi()

    # Créer le dépôt si inexistant
    try:
        create_repo(repo_name, private=private)
    except Exception:
        pass   # dépôt déjà existant

    # Pousser le modèle et le tokeniseur
    model.push_to_hub(repo_name)
    tokenizer.push_to_hub(repo_name)

    # Générer le README (model card HuggingFace)
    readme_content = f"""---
language: fr-medieval
tags:
  - ner
  - token-classification
  - medieval-french
  - camembert
  - lora
license: cc-by-4.0
pipeline_tag: token-classification
---

# {model_card['model_id']}

{model_card.get('bias_analysis', '')}

## Performances
- F1 micro : {model_card['evaluation']['metrics']['f1_micro']}
- Split SHA-256 : {model_card['training']['split_hash']}

## Limitations
{chr(10).join('- ' + l for l in model_card.get('limitations', []))}
"""
    api.upload_file(
        path_or_fileobj = readme_content.encode("utf-8"),
        path_in_repo    = "README.md",
        repo_id         = repo_name,
    )

    url = f"https://huggingface.co/{repo_name}"
    print(f"Modèle publié : {url}")
    return url
```

---

## 6. Bonus : traduction automatique médiévale (OpenNMT / Seq2SeqTrainer)

*Cette section est optionnelle. Elle est accessible aux équipes ayant validé les Étapes 1–5 avec au moins 30 minutes restantes.*

### 6.1 Contexte

Le corpus de 500 paires fourni (fichier `corpus_nmt_500.jsonl`) contient des paires (moyen français normalisé → français moderne) construites depuis des éditions critiques bilingues. Votre objectif est de fine-tuner un modèle de traduction neuronal pour obtenir un score BLEU > 10 sur le jeu de test aligné.

**Modèle recommandé :** Helsinki-NLP/opus-mt-fr-ROMANCE (58M paramètres, rapide à fine-tuner) ou mBART-50 (si GPU disponible).

### 6.2 Fine-tuning avec Seq2SeqTrainer

```python
from transformers import (MarianMTModel, MarianTokenizer,
                           Seq2SeqTrainer, Seq2SeqTrainingArguments,
                           DataCollatorForSeq2Seq)
from datasets import Dataset
import json

NMT_MODEL = "Helsinki-NLP/opus-mt-fr-ROMANCE"

def load_nmt_corpus(path: str) -> tuple[list, list]:
    """Charge le corpus NMT depuis le fichier JSONL fourni."""
    sources, targets = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            pair = json.loads(line)
            sources.append(pair["medieval"])    # moyen français normalisé
            targets.append(pair["moderne"])     # français moderne
    return sources, targets

def prepare_nmt_dataset(sources, targets, tokenizer,
                          max_input_length=128,
                          max_target_length=128) -> Dataset:
    def tokenize(examples):
        model_inputs = tokenizer(
            examples["source"],
            max_length=max_input_length,
            truncation=True,
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["target"],
                max_length=max_target_length,
                truncation=True,
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    raw = Dataset.from_dict({
        "source": sources,
        "target": targets,
    })
    return raw.map(tokenize, batched=True, remove_columns=["source","target"])

# ── Chargement ───────────────────────────────────────────────────────────
nmt_tokenizer = MarianTokenizer.from_pretrained(NMT_MODEL)
nmt_model     = MarianMTModel.from_pretrained(NMT_MODEL)

sources, targets = load_nmt_corpus("corpus_nmt_500.jsonl")

# Split 80/10/10
n_train = int(len(sources) * 0.80)
n_val   = int(len(sources) * 0.10)
train_ds = prepare_nmt_dataset(
    sources[:n_train], targets[:n_train], nmt_tokenizer)
val_ds   = prepare_nmt_dataset(
    sources[n_train:n_train+n_val],
    targets[n_train:n_train+n_val], nmt_tokenizer)

# ── Entraînement ─────────────────────────────────────────────────────────
nmt_args = Seq2SeqTrainingArguments(
    output_dir              = "./nmt_opus_finetuned",
    num_train_epochs        = 10,
    per_device_train_batch_size = 16,
    per_device_eval_batch_size  = 16,
    learning_rate           = 5e-5,
    predict_with_generate   = True,
    generation_max_length   = 128,
    eval_strategy           = "epoch",
    save_strategy           = "epoch",
    load_best_model_at_end  = True,
    metric_for_best_model   = "bleu",
    greater_is_better       = True,
    fp16                    = True,
    report_to               = "none",
)

# Calcul BLEU avec sacrebleu
from evaluate import load as load_metric
bleu_metric = load_metric("sacrebleu")

def compute_bleu(eval_pred):
    preds, labels = eval_pred
    decoded_preds  = nmt_tokenizer.batch_decode(preds,   skip_special_tokens=True)
    decoded_labels = nmt_tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = bleu_metric.compute(
        predictions = decoded_preds,
        references  = [[l] for l in decoded_labels],
    )
    return {"bleu": round(result["score"], 2)}

trainer = Seq2SeqTrainer(
    model           = nmt_model,
    args            = nmt_args,
    train_dataset   = train_ds,
    eval_dataset    = val_ds,
    tokenizer       = nmt_tokenizer,
    data_collator   = DataCollatorForSeq2Seq(nmt_tokenizer, model=nmt_model),
    compute_metrics = compute_bleu,
)
trainer.train()

# Évaluation finale sur le jeu de test
test_ds = prepare_nmt_dataset(
    sources[n_train+n_val:],
    targets[n_train+n_val:],
    nmt_tokenizer,
)
test_results = trainer.evaluate(test_ds)
print(f"\nBLEU sur le jeu de test : {test_results.get('eval_bleu', 0):.2f}")
print(f"Seuil bonus : BLEU > 10 — {'VALIDÉ' if test_results.get('eval_bleu',0) > 10 else 'non atteint'}")
```

**BLEU attendu après 10 epochs sur 400 paires d'entraînement :**
- Zero-shot (Helsinki-NLP/opus-mt-fr-ROMANCE) : ~8 BLEU
- Fine-tuné 10 epochs : ~14–18 BLEU (selon la qualité du corpus)
- Seuil bonus : **BLEU > 10**

---

## Récapitulatif des livrables

| Livrable | Section | Format |
|---|---|---|
| `quantization_report.json` | 1 | JSON |
| `benchmark_report.json` | 2 | JSON |
| `src/main.py` + image Docker | 3 | Python + Dockerfile |
| `tests/` (suite pytest complète) | 4 | Python |
| `MODEL_CARD.json` | 5.1 | JSON |
| `DATA_SOURCES.md` | 5.2 | Markdown |
| `README.md` | 5.3 | Markdown |
| Dépôt HuggingFace Hub | 5.4 | — |
| `[Bonus]` BLEU > 10 | 6 | JSON (résultats) |

**Vérification finale avant soumission :**

```bash
# 1. Tests passent
pytest tests/ -v

# 2. Docker se construit et démarre
docker build -t nlp-medieval:1.0.0 . && \
docker run -p 8000:8000 nlp-medieval:1.0.0

# 3. Health check passe
curl http://localhost:8000/health

# 4. Fichiers obligatoires présents
ls MODEL_CARD.json DATA_SOURCES.md README.md CONVENTIONS_NLP.md \
   quantization_report.json benchmark_report.json

# 5. Archive zip pour soumission
zip -r [groupe]_nlp_manuscrits_MD5.zip \
    src/ tests/ models/ tei_output/ \
    enriched_corpus_v2.jsonl knowledge_graph.jsonld \
    MODEL_CARD.json DATA_SOURCES.md README.md CONVENTIONS_NLP.md \
    Dockerfile requirements.txt \
    quantization_report.json benchmark_report.json \
    experiments/journal.jsonl
```

---

## Bibliographie de référence

Ce chapitre s'appuie sur les mêmes références techniques que le Chapitre 9 pour la quantisation et le serving. Les références spécifiques à ce TP sont :

pytest (2024). [https://docs.pytest.org](https://docs.pytest.org) — Documentation officielle pytest.

FastAPI (2024). [https://fastapi.tiangolo.com](https://fastapi.tiangolo.com) — Documentation officielle FastAPI.

HuggingFace Hub (2024). [https://huggingface.co/docs/hub](https://huggingface.co/docs/hub) — Publication de modèles.

Klein, G., Kim, Y., Deng, Y., Senellart, J., & Rush, A. M. (2017). **OpenNMT: Open-Source Toolkit for Neural Machine Translation**. *ACL 2017 (Démo)*. [arXiv:1701.02810](https://arxiv.org/abs/1701.02810)

Tang, Y., Tran, C., Li, X., Chen, P.-J., Goyal, N., Chaudhary, V., Gu, J., & Fan, A. (2020). **Multilingual Translation with Extensible Multilingual Pretraining and Finetuning**. [arXiv:2008.00401](https://arxiv.org/abs/2008.00401) — mBART-50.

---

*Support de cours rédigé pour le Master Data/IA · Module NLP · MD5 Volet 2 · 2026. Ce document accompagne le TP final du Jour 5 (11h30–14h00). Les soutenances suivent à 14h30.*
