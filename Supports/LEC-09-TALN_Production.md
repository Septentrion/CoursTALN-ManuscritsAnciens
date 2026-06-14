# Chapitre 9 — NLP en production : du modèle au service

**Module NLP · Master Data/IA · MD5 Volet 2 · 2026**  
Cours magistral — 2 heures 30

---

## Avant-propos : ce qui change quand on passe en production

Les quatre jours précédents ont été consacrés à obtenir de bons modèles : CER < 5 %, F1-NER > 0.80, topics cohérents, graphe de connaissances complet. Ces métriques sont mesurées dans un notebook, sur un corpus fixe, avec autant de temps de calcul que nécessaire. La production change tout — pas les métriques cibles, mais les contraintes dans lesquelles on doit les atteindre.

Un modèle en production doit répondre en moins de 200 ms à la 99e percentile de la distribution des requêtes. Il doit tenir sous une charge de dizaines ou centaines de requêtes simultanées. Il doit consommer une quantité de mémoire GPU prévisible et raisonnable. Et il doit continuer de fonctionner correctement six mois après son déploiement, alors que les données en entrée auront changé sans que personne ne l'ait signalé.

Ce chapitre couvre les techniques qui permettent de franchir ce passage : la quantisation (réduire l'empreinte mémoire du modèle), la distillation (réduire le nombre de paramètres), les serveurs d'inférence dédiés (maximiser le débit), et l'observabilité (détecter les problèmes avant qu'ils ne deviennent des pannes).

Il se termine sur une réflexion spécifique aux humanités numériques : les projets de traitement de manuscrits médiévaux ont des contraintes de production très différentes des applications commerciales à fort trafic, et les outils présentés ici doivent être sélectionnés en conséquence.

---

## 1. Quantisation : réduire l'empreinte mémoire sans effondrer les performances

### 1.1 Le problème de la précision

Les modèles pré-entraînés comme CamemBERT sont stockés en virgule flottante 32 bits (FP32) : chaque paramètre occupe 4 octets. CamemBERT-base avec ses 110 millions de paramètres pèse donc 440 Mo en FP32, rien que pour les poids — avant les activations, les gradients et l'optimiseur, qui peuvent multiplier ce chiffre par 4 à 16 lors de l'entraînement.

En inférence pure (pas d'entraînement), cette empreinte peut être réduite drastiquement en représentant les poids avec moins de bits. Le trade-off est simple : moins de bits → moins de précision → risque de dégradation des performances.

| Format | Bits/param | Mémoire (CamemBERT-base) | Perte typique |
|---|---|---|---|
| FP32 | 32 | 440 Mo | — (référence) |
| FP16 / BF16 | 16 | 220 Mo | < 0.1 % F1 |
| INT8 | 8 | 110 Mo | 0.5–1 % F1 |
| INT4 / NF4 | 4 | 55 Mo | 1–3 % F1 |

### 1.2 INT8 : quantisation dynamique et statique

La quantisation INT8 représente les poids et/ou les activations sur 8 bits entiers. Elle est disponible dans `bitsandbytes` (Dettmers et al.) et dans `ONNX Runtime`.

**Quantisation dynamique :** les poids sont convertis en INT8 au moment du chargement ; les activations restent en FP16 ou FP32 pendant l'inférence. C'est la méthode la plus simple, applicable sans données de calibration.

```python
from transformers import AutoModelForTokenClassification
import torch

# Chargement du modèle NER fine-tuné
model = AutoModelForTokenClassification.from_pretrained("./ner_lora_r8")
model.eval()

# Quantisation dynamique INT8 (weights seulement)
model_int8 = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},   # quantiser toutes les couches linéaires
    dtype=torch.qint8,
)
print(f"Taille modèle FP32  : {sum(p.numel()*4 for p in model.parameters())/1e6:.0f} Mo")
print(f"Taille modèle INT8  : {sum(p.numel()*1 for p in model_int8.parameters())/1e6:.0f} Mo")
```

**Quantisation statique :** les activations sont aussi quantisées, ce qui nécessite un passage de calibration sur un petit ensemble de données représentatives. Elle produit une réduction de latence plus importante que la quantisation dynamique, au prix d'une légère dégradation supplémentaire.

### 1.3 GPTQ et AWQ : quantisation post-entraînement de précision

Pour les modèles de génération (LLM), où INT8 peut encore être coûteux en mémoire, des méthodes de quantisation 4 bits ont été développées.

**GPTQ** (*GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*, Frantar et al. 2022) quantise couche par couche en minimisant l'erreur de reconstruction. Pour chaque couche $\mathbf{W}$ :

$$\mathbf{W}_q = \arg\min_{\hat{\mathbf{W}}} \| \mathbf{W}\mathbf{X} - \hat{\mathbf{W}}\mathbf{X} \|_F^2$$

où $\mathbf{X}$ est un petit ensemble de données de calibration. GPTQ produit des modèles 4 bits avec une perte de perplexité souvent inférieure à 5 % par rapport au FP16.

**AWQ** (*Activation-Aware Weight Quantization*, Lin et al. 2023) observe que tous les poids ne sont pas équivalents : les poids qui correspondent à des activations de grande amplitude sont plus importants pour la précision que les poids activés faiblement. AWQ protège ces poids critiques (1 % du total environ) de la quantisation agressive, et quantise agressivement les poids peu importants.

```python
from awq import AutoAWQForCausalLM   # pip install autoawq

# Quantisation AWQ 4 bits
model = AutoAWQForCausalLM.from_pretrained("almanach/camembert-base")
quant_config = {"zero_point": True, "q_group_size": 128,
                "w_bit": 4, "version": "GEMM"}
model.quantize(tokenizer, quant_config=quant_config,
               calib_data=calibration_texts)
model.save_quantized("camembert-awq-4bit")
```

### 1.4 NF4 et double quantisation dans QLoRA

NF4 (*Normal Float 4-bit*), introduit dans QLoRA (Dettmers et al. 2023), est un format de quantisation 4 bits optimisé pour des poids distribués normalement. Il construit une grille de 16 valeurs ($2^4$) dont les quantiles correspondent à la distribution normale des poids pré-entraînés. La **double quantisation** quantise ensuite les constantes de quantisation elles-mêmes, économisant environ 0.37 bits supplémentaires par paramètre.

Ce mécanisme a déjà été utilisé au Chapitre 3 pour QLoRA. Il est pertinent ici dans un contexte de déploiement : un modèle NF4 chargé en inférence (sans les adaptateurs LoRA) occupe la même empreinte que lors de l'entraînement QLoRA.

---

## 2. Distillation : entraîner un modèle plus petit pour correspondre à un modèle plus grand

### 2.1 Le principe teacher-student

La distillation (Hinton, Vinyals & Dean 2015) est une technique d'entraînement qui transfère les connaissances d'un modèle de grande taille (*teacher*) vers un modèle de petite taille (*student*). L'idée centrale est que le modèle teacher produit, pour chaque exemple, une distribution de probabilités sur les classes — pas seulement la classe prédite. Cette distribution porte plus d'information que la simple étiquette binaire : si le teacher prédit 97 % de B-PER, 2 % de B-TITLE et 1 % de B-ORG pour un token donné, il communique que le token ressemble un peu à un TITLE et encore moins à une ORG. Un étudiant entraîné uniquement sur l'étiquette dure (B-PER) perdrait cette nuance.

La fonction de perte de distillation combine deux termes :

$$\mathcal{L}_{\text{total}} = \alpha \cdot \mathcal{L}_{\text{CE}}(y_s, y_{\text{vrai}}) + (1-\alpha) \cdot \mathcal{L}_{\text{KD}}(z_s, z_t, T)$$

Le premier terme $\mathcal{L}_{\text{CE}}$ est la cross-entropie standard entre la prédiction de l'étudiant et l'étiquette réelle. Le second terme $\mathcal{L}_{\text{KD}}$ est la divergence de Kullback-Leibler entre les distributions de probabilités adoucies du teacher et de l'étudiant :

$$\mathcal{L}_{\text{KD}}(z_s, z_t, T) = T^2 \cdot \text{KL}\!\left(\text{softmax}\!\left(\frac{z_t}{T}\right) \;\Big\|\; \text{softmax}\!\left(\frac{z_s}{T}\right)\right)$$

Le paramètre $T$ est la **température** : une valeur élevée ($T \geq 4$) adoucit les distributions, amplifiant les probabilités faibles des classes non dominantes et fournissant ainsi plus d'information à l'étudiant. Le facteur $T^2$ compense l'écrasement des gradients introduit par la division par $T$.

```python
import torch
import torch.nn.functional as F

def distillation_loss(student_logits:  torch.Tensor,
                       teacher_logits:  torch.Tensor,
                       true_labels:     torch.Tensor,
                       temperature:     float = 4.0,
                       alpha:           float = 0.5) -> torch.Tensor:
    """
    Calcule la perte de distillation combinée.

    Paramètres
    ----------
    student_logits : (batch, seq_len, n_labels)  logits du student
    teacher_logits : (batch, seq_len, n_labels)  logits du teacher (gelé)
    true_labels    : (batch, seq_len)             étiquettes de référence
    temperature    : float                         T pour l'adoucissement
    alpha          : float                         poids de la CE dure

    Retourne
    --------
    torch.Tensor  perte scalaire
    """
    T = temperature

    # Perte CE dure (étudiant vs vérité terrain)
    loss_ce = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        true_labels.view(-1),
        ignore_index = -100,
    )

    # Perte de distillation (KL entre distributions adoucies)
    student_soft = F.log_softmax(student_logits / T, dim=-1)
    teacher_soft = F.softmax(teacher_logits / T, dim=-1)

    # Masquer les tokens ignorés (-100)
    mask = (true_labels != -100).unsqueeze(-1).expand_as(student_soft)
    loss_kd = F.kl_div(
        student_soft[mask.bool()].view(-1, student_logits.size(-1)),
        teacher_soft[mask.bool()].view(-1, student_logits.size(-1)),
        reduction = "batchmean",
    ) * (T ** 2)

    return alpha * loss_ce + (1 - alpha) * loss_kd
```

### 2.2 DistilBERT et DistilCamemBERT

DistilBERT (Sanh et al. 2019) est la première application de grande échelle de la distillation à BERT. Il réduit le nombre de couches de 12 à 6 (en les initialisant avec les couches paires du teacher), conserve 66 millions des 110 millions de paramètres de BERT-base, et atteint 97 % des performances de BERT sur GLUE avec une inférence ~2× plus rapide.

**DistilCamemBERT** suit le même protocole appliqué à CamemBERT. Il n'existe pas de version publiée officiellement par l'équipe CamemBERT, mais la distillation peut être reproduite avec le corpus OSCAR français et la procédure de DistilBERT. En pratique, pour la NER médiévale, un CamemBERT-base fine-tuné avec LoRA (Chapitre 6) est déjà suffisamment compact pour les cas d'usage à faible trafic — la distillation n'est pertinente que si la latence cible est inférieure à 30 ms.

**TinyBERT** (Jiao et al. 2019) pousse la distillation plus loin : en plus des distributions de probabilités finales, il distille les représentations intermédiaires de chaque couche (*layer-wise distillation*) et les matrices d'attention. TinyBERT-4L (4 couches, 14.5M paramètres) atteint 95 % des performances de BERT-base sur GLUE, avec une inférence 9.4× plus rapide.

---

## 3. Serving : les serveurs d'inférence spécialisés

### 3.1 Pourquoi un serveur d'inférence dédié

Une application FastAPI naïve qui charge le modèle PyTorch et traite les requêtes une par une atteint typiquement 10–15 requêtes par seconde sur une T4. Cette architecture a trois défauts majeurs en production.

Premièrement, elle ne bénéficie pas du parallélisme GPU : chaque requête monopolise le GPU séquentiellement, laissant une fraction significative des unités de calcul oisives.

Deuxièmement, elle charge le modèle en mémoire plusieurs fois si plusieurs workers sont lancés en parallèle (un processus FastAPI par CPU core), ce qui peut dépasser la VRAM disponible.

Troisièmement, la gestion de la mémoire GPU est sous-optimale : les tenseurs alloués pour une requête ne sont pas réutilisés pour la suivante, générant de la fragmentation mémoire et des pics de latence.

Les serveurs d'inférence dédiés — vLLM, TGI, ONNX Runtime Serving — résolvent ces trois problèmes.

### 3.2 ONNX Runtime : portabilité et accélération CPU

ONNX (*Open Neural Network Exchange*) est un format d'échange de modèles ML indépendant du framework d'entraînement. Un modèle PyTorch exporté en ONNX peut être exécuté sur n'importe quel hardware (CPU x86, ARM, GPU, NPU) sans dépendance à PyTorch.

```python
# Export du modèle NER en ONNX
from transformers import AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("almanach/camembert-base")
model.eval()

# Exemple de séquence pour le tracing
dummy_input = tokenizer(
    "le sénéchal de normandie signa l acte",
    return_tensors="pt", max_length=128, truncation=True,
)

torch.onnx.export(
    model,
    (dummy_input["input_ids"], dummy_input["attention_mask"]),
    "ner_model.onnx",
    input_names  = ["input_ids", "attention_mask"],
    output_names = ["logits"],
    dynamic_axes = {
        "input_ids":      {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "logits":         {0: "batch", 1: "sequence"},
    },
    opset_version = 17,
)

# Inférence avec ONNX Runtime
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession(
    "ner_model.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)

def predict_onnx(text: str) -> np.ndarray:
    inputs  = tokenizer(text, return_tensors="np",
                         max_length=128, truncation=True)
    outputs = session.run(
        None,
        {"input_ids":      inputs["input_ids"],
         "attention_mask": inputs["attention_mask"]},
    )
    return outputs[0]   # logits : (1, seq_len, n_labels)
```

**Gains typiques :** ONNX Runtime FP32 est ~1.9× plus rapide que PyTorch FP32 sur CPU pour CamemBERT, grâce à la fusion de noyaux et à l'optimisation du graphe. ONNX Runtime INT8 atteint ~4.3× le débit de PyTorch FP32.

### 3.3 Batching dynamique et KV-cache

**Batching statique :** regrouper plusieurs requêtes dans un même batch avant l'inférence GPU multiplie le débit proportionnellement au batch size, jusqu'au point où la mémoire GPU sature. Pour un modèle de classification de tokens comme CamemBERT-NER, le batching statique avec `batch_size=8` multiplie typiquement le débit par 3–4× par rapport à `batch_size=1`.

**Batching dynamique :** plutôt que d'attendre que le batch soit plein avant de traiter, le serveur déclenche l'inférence dès qu'un timeout est atteint ou que le batch est plein, selon le premier événement. Cela permet d'équilibrer la latence et le débit selon la charge du système.

```python
from asyncio import Queue, wait_for
import asyncio

class DynamicBatcher:
    """
    Gestionnaire de batching dynamique pour l'inférence NER.
    Collecte les requêtes pendant max_wait_ms, puis les traite en batch.
    """
    def __init__(self, model, tokenizer,
                  max_batch_size: int   = 8,
                  max_wait_ms:    float = 10.0):
        self.model          = model
        self.tokenizer      = tokenizer
        self.max_batch_size = max_batch_size
        self.max_wait_ms    = max_wait_ms
        self.queue: Queue   = Queue()

    async def enqueue(self, text: str) -> list[str]:
        """Ajoute une requête à la file et retourne les prédictions."""
        future = asyncio.get_event_loop().create_future()
        await self.queue.put((text, future))
        return await future

    async def process_loop(self):
        """Boucle principale : collecte les requêtes et lance les batches."""
        while True:
            batch, futures = [], []
            try:
                text, future = await wait_for(
                    self.queue.get(), timeout=self.max_wait_ms / 1000
                )
                batch.append(text); futures.append(future)
                # Remplir le batch jusqu'à max_batch_size
                while len(batch) < self.max_batch_size and not self.queue.empty():
                    text, future = self.queue.get_nowait()
                    batch.append(text); futures.append(future)
            except asyncio.TimeoutError:
                if not batch:
                    continue

            # Inférence sur le batch
            results = self._predict_batch(batch)
            for future, result in zip(futures, results):
                future.set_result(result)

    def _predict_batch(self, texts: list[str]) -> list[list[str]]:
        import torch
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=128,
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits
        preds = logits.argmax(dim=-1).tolist()
        return [[self.model.config.id2label[p] for p in seq]
                for seq in preds]
```

**KV-cache :** pour les modèles génératifs (Chapitre 11), le KV-cache stocke les clés et valeurs d'attention des tokens déjà traités, évitant de les recalculer à chaque étape de génération. La taille du KV-cache est :

$$\text{Taille KV} = 2 \times L \times H \times d_k \times N_{\text{tokens}} \times 2 \text{ octets (FP16)}$$

Pour CamemBERT-base ($L=12$, $H=12$, $d_k=64$) : 36 Ko par token. Pour une séquence de 128 tokens, le KV-cache occupe 4.5 Mo par requête — négligeable pour un modèle encoder. Pour un LLM comme LLaMA-7B ($L=32$, $H=32$, $d_k=128$) avec des séquences de 2048 tokens, le KV-cache dépasse 4 Go, rendant sa gestion critique.

### 3.4 vLLM et continuous batching

vLLM (Kwon et al. 2023) introduit **PagedAttention**, une gestion du KV-cache inspirée de la pagination mémoire des systèmes d'exploitation. Plutôt que d'allouer un bloc continu de mémoire pour chaque séquence (ce qui fragmente la VRAM), PagedAttention alloue des pages de taille fixe et maintient une table de correspondances. Cela permet :

- Le **continuous batching** : de nouvelles séquences peuvent être insérées dans un batch en cours d'exécution dès qu'une séquence se termine, sans attendre que tout le batch soit terminé.
- Le partage de mémoire entre plusieurs séquences ayant le même préfixe (utile pour les prompts système identiques).

Pour le modèle NER médiéval (encoder uniquement, séquences courtes), vLLM n'est pas le choix approprié — il est optimisé pour la génération auto-régressive. TGI (*Text Generation Inference*) de Hugging Face ou simplement ONNX Runtime sont plus adaptés pour un modèle de classification.

### 3.5 Exemple d'endpoint FastAPI

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time

app = FastAPI(title="NER médiéval — Pipeline NLP MD5")

class AnalyzeRequest(BaseModel):
    text:       str
    line_id:    str = ""
    confidence: float = 1.0

class EntitySpan(BaseModel):
    start:  int
    end:    int
    label:  str
    text:   str

class AnalyzeResponse(BaseModel):
    line_id:       str
    normalized:    str
    ner_spans:     list[EntitySpan]
    latency_ms:    float
    model_version: str

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """
    Endpoint principal : normalise un texte HTR et en extrait les entités.
    Reçoit le texte brut, retourne les entités avec offsets caractères.
    """
    if not request.text.strip():
        raise HTTPException(status_code=422, detail="Texte vide")

    t0 = time.perf_counter()

    # Pipeline complet (Chapitres 2–6)
    normalized = normalize_pipeline(request.text)
    spans      = ner_pipeline(normalized)

    latency_ms = (time.perf_counter() - t0) * 1000

    return AnalyzeResponse(
        line_id        = request.line_id,
        normalized     = normalized,
        ner_spans      = [EntitySpan(**s) for s in spans],
        latency_ms     = round(latency_ms, 2),
        model_version  = MODEL_VERSION,
    )

@app.get("/health")
async def health():
    return {"status": "ok", "model_version": MODEL_VERSION}
```

---

## 4. Évaluation production : latence, débit, mémoire

### 4.1 Métriques de latence

La **latence** d'une requête est le temps entre sa réception et la livraison de la réponse. En production, la latence n'est pas une valeur fixe — elle suit une distribution, souvent log-normale avec une queue longue. Deux percentiles sont standard :

- **p50** : médiane — 50 % des requêtes sont traitées en moins de ce temps. Représente l'expérience de l'utilisateur médian.
- **p99** : 99e percentile — 1 % des requêtes prennent plus que ce temps. Représente les cas lents qui se produisent plusieurs fois par minute sous une charge modérée.

La règle empirique des SRE (*Site Reliability Engineers*) : si p99 > 3 × p50, la distribution a une queue longue qui signale un problème de ressources (contention GPU, garbage collection, fragmentation mémoire).

**Exemple de benchmark :**

```python
import time, statistics, numpy as np

def benchmark_endpoint(endpoint_fn: callable,
                         texts:        list[str],
                         n_warmup:     int = 10) -> dict:
    """
    Benchmark de latence sur une liste de textes.
    Retourne les percentiles et le débit.
    """
    # Warm-up : pas comptabilisé (chargement des poids en VRAM)
    for text in texts[:n_warmup]:
        endpoint_fn(text)

    # Mesure
    latencies = []
    t_start   = time.perf_counter()
    for text in texts:
        t0 = time.perf_counter()
        endpoint_fn(text)
        latencies.append((time.perf_counter() - t0) * 1000)
    t_total = time.perf_counter() - t_start

    latencies = sorted(latencies)
    return {
        "n_requests":  len(latencies),
        "p50_ms":      np.percentile(latencies, 50),
        "p90_ms":      np.percentile(latencies, 90),
        "p99_ms":      np.percentile(latencies, 99),
        "mean_ms":     statistics.mean(latencies),
        "std_ms":      statistics.stdev(latencies),
        "throughput":  round(len(latencies) / t_total, 1),   # req/s
    }

# Résultats typiques sur T4 (CamemBERT-NER, batch_size=1, seq=128) :
# FP16  : p50=50ms  p99=99ms   throughput=18 req/s
# INT8  : p50=28ms  p99=48ms   throughput=32 req/s
# ONNX  : p50=22ms  p99=42ms   throughput=40 req/s
```

### 4.2 Débit et mémoire GPU

Le **débit** (*throughput*) est le nombre de requêtes traitées par seconde. Il dépend du batch size, de la longueur des séquences, et de la quantisation. Pour un modèle de classification de tokens comme CamemBERT-NER :

| Configuration | Débit (req/s) | VRAM (Mo) |
|---|---|---|
| FP16, batch=1 | 18 | 580 |
| FP16, batch=8 | 55 | 810 |
| INT8, batch=1 | 32 | 310 |
| INT8, batch=8 | 90 | 450 |
| ONNX FP32, CPU | 5 | 440 (RAM) |
| ONNX INT8, CPU | 22 | 110 (RAM) |

La **mémoire GPU** inclut les poids du modèle, les activations de la couche courante, et les buffers d'inférence. Règle pratique : prévoir 2× l'empreinte des poids en VRAM pour l'inférence séquentielle, 3× pour un batch de 8.

---

## 5. Observabilité : logging, détection de drift, monitoring

### 5.1 Logging structuré des inférences

En production, chaque appel à l'endpoint doit être loggué de façon structurée. Le log d'inférence est à la fois un outil de débogage, un audit trail pour la traçabilité scientifique, et la source de données pour la détection de drift.

```python
import logging, json, datetime

logger = logging.getLogger("nlp_pipeline")

class InferenceLogger:
    """
    Logger structuré pour les inférences NER.
    Stocke : timestamp, version du modèle, entrée tronquée,
    sortie (entités), latence, confiance.
    """
    def log(self,
             request:    dict,
             response:   dict,
             latency_ms: float) -> None:
        entry = {
            "ts":            datetime.datetime.utcnow().isoformat(),
            "model_version": response.get("model_version", "unknown"),
            "line_id":       request.get("line_id", ""),
            "input_len":     len(request.get("text", "")),
            "n_entities":    len(response.get("ner_spans", [])),
            "entity_types":  [s["label"] for s in response.get("ner_spans", [])],
            "latency_ms":    latency_ms,
            "confidence":    request.get("confidence", 1.0),
        }
        logger.info(json.dumps(entry, ensure_ascii=False))
```

**Ce qu'il ne faut pas logger :** le texte complet des transcriptions de manuscrits peut contenir des informations personnelles (noms de personnes, dates de naissance médiévales ayant une valeur patrimoniale). Dans un contexte d'API patrimoniale, définir explicitement une politique de rétention des logs et les anonymiser si nécessaire.

### 5.2 Détection de drift conceptuel

Un **drift conceptuel** (*concept drift*) survient quand la distribution des données en entrée change significativement par rapport à la distribution sur laquelle le modèle a été entraîné. Dans le contexte des humanités numériques, ce risque est réel : un modèle entraîné sur des chartes normandes du XIVe siècle peut être déployé sur des registres paroissiaux gascons du XVIe siècle, dont la langue et les entités nommées diffèrent.

La détection de drift s'appuie sur la comparaison statistique entre la distribution baseline (corpus d'entraînement) et la distribution courante (requêtes récentes). La **divergence de Jensen-Shannon** est particulièrement adaptée car elle est symétrique et bornée dans $[0, 1]$ :

$$\text{JSD}(P \| Q) = \frac{1}{2} \text{KL}(P \| M) + \frac{1}{2} \text{KL}(Q \| M), \quad M = \frac{P+Q}{2}$$

```python
import numpy as np
from scipy.spatial.distance import jensenshannon

class DriftDetector:
    """
    Détecteur de drift basé sur la distribution des types d'entités
    et sur la confiance moyenne des prédictions.
    """
    def __init__(self, baseline_entity_dist: dict,
                  baseline_confidence:       float,
                  jsd_threshold:             float = 0.05,
                  confidence_drop_threshold: float = 0.10):
        self.baseline_dist       = baseline_entity_dist
        self.baseline_confidence = baseline_confidence
        self.jsd_threshold       = jsd_threshold
        self.conf_drop_threshold = confidence_drop_threshold
        self._recent_entities:  list[str]   = []
        self._recent_confidences: list[float] = []

    def update(self, entity_types: list[str],
                confidence:       float) -> None:
        """Enregistre les entités et la confiance d'une inférence récente."""
        self._recent_entities.extend(entity_types)
        self._recent_confidences.append(confidence)
        # Fenêtre glissante : garder les 1000 dernières entités
        if len(self._recent_entities) > 1000:
            self._recent_entities = self._recent_entities[-1000:]

    def check_drift(self) -> dict:
        """
        Compare la distribution courante avec la baseline.
        Retourne un dict avec les signaux de drift.
        """
        if len(self._recent_entities) < 100:
            return {"status": "insufficient_data"}

        # Distribution courante des types d'entités
        from collections import Counter
        current_counts = Counter(self._recent_entities)
        all_types      = set(self.baseline_dist) | set(current_counts)
        baseline_vec   = np.array([self.baseline_dist.get(t, 0)
                                    for t in all_types], dtype=float)
        current_vec    = np.array([current_counts.get(t, 0)
                                    for t in all_types], dtype=float)
        if current_vec.sum() > 0:
            current_vec /= current_vec.sum()
        if baseline_vec.sum() > 0:
            baseline_vec /= baseline_vec.sum()

        jsd            = jensenshannon(baseline_vec, current_vec) ** 2
        mean_conf      = float(np.mean(self._recent_confidences[-100:]))
        confidence_drop = self.baseline_confidence - mean_conf

        drift_detected = (jsd > self.jsd_threshold
                          or confidence_drop > self.conf_drop_threshold)

        return {
            "status":            "drift_detected" if drift_detected else "ok",
            "jsd":               round(float(jsd), 4),
            "jsd_threshold":     self.jsd_threshold,
            "mean_confidence":   round(mean_conf, 3),
            "confidence_drop":   round(confidence_drop, 3),
            "n_entities_recent": len(self._recent_entities),
            "drift_type":        (
                "entity_distribution" if jsd > self.jsd_threshold
                else "confidence_drop" if drift_detected
                else "none"
            ),
        }
```

### 5.3 Dashboard de monitoring

Un dashboard minimal pour l'endpoint NER médiéval doit exposer, en temps réel :

**Métriques de performance :** p50 et p99 de la latence (graphe temporel), débit courant (req/s), taux d'erreur HTTP 5xx.

**Métriques de qualité :** confiance moyenne des prédictions (signal de drift), distribution des types d'entités sur la fenêtre glissante (signal de drift), proportion de requêtes sans entité détectée (signal de dégradation).

**Métriques d'infrastructure :** utilisation GPU (%), VRAM utilisée (Mo), charge CPU, longueur de la file d'attente du batcher.

Ces métriques peuvent être exposées via Prometheus et visualisées dans Grafana, ou plus simplement via FastAPI et un endpoint `/metrics`.

---

## 6. Spécificités des humanités numériques : traçabilité vs throughput

### 6.1 Le profil de trafic d'une API patrimoniale

Une API de traitement de manuscrits médiévaux n'est pas soumise aux mêmes contraintes qu'une API commerciale. Un service de messagerie instantanée doit gérer des millions de requêtes par heure et une latence visible doit être inférieure à 100 ms. Une API patrimoniale utilisée par des chercheurs en histoire du droit traite quelques milliers de requêtes par semaine, avec une tolérance à la latence de l'ordre de la seconde.

Ce profil de trafic change radicalement les priorités :

**Ce qui importe :** la reproductibilité des résultats (deux appels identiques doivent produire la même réponse), la traçabilité des versions de modèle (savoir exactement quel modèle a produit quelle annotation sur quel corpus), la qualité des annotations (F1 > 0.80 est plus important que 40 req/s vs 18 req/s).

**Ce qui importe moins :** le throughput brut, la latence à la milliseconde, le scaling horizontal.

### 6.2 Versionnement des modèles et traçabilité scientifique

Dans un contexte de recherche, chaque annotation produite est un résultat scientifique potentiellement cité dans une publication. La traçabilité des versions de modèle est donc une exigence scientifique, pas seulement opérationnelle.

```python
from dataclasses import dataclass, field
from datetime import datetime
import hashlib, json

@dataclass
class ModelCard:
    """
    Fiche descriptive d'un modèle déployé.
    Doit accompagner tout export de données annotées.
    """
    model_name:       str
    base_model:       str           # ex. "almanach/camembert-base"
    training_split:   str           # SHA-256 du split de données (Chapitre 4)
    training_date:    str           = field(default_factory=lambda: datetime.utcnow().isoformat())
    lora_config:      dict          = field(default_factory=dict)
    metrics_test:     dict          = field(default_factory=dict)   # F1 par type sur le test set
    corpus_description: str         = ""
    known_limitations:  list[str]   = field(default_factory=list)
    bias_analysis:      str         = ""

    def to_dict(self) -> dict:
        return {
            "model_name":        self.model_name,
            "base_model":        self.base_model,
            "training_split":    self.training_split,
            "training_date":     self.training_date,
            "lora_config":       self.lora_config,
            "metrics_test":      self.metrics_test,
            "corpus_description":self.corpus_description,
            "known_limitations": self.known_limitations,
            "bias_analysis":     self.bias_analysis,
            "model_card_hash":   self._hash(),
        }

    def _hash(self) -> str:
        content = json.dumps({
            "model_name":     self.model_name,
            "training_split": self.training_split,
            "training_date":  self.training_date,
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def save(self, path: str = "MODEL_CARD.json") -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

# Exemple de model card pour le pipeline NER médiéval
model_card = ModelCard(
    model_name     = "camembert-ner-medieval-v1.0",
    base_model     = "almanach/camembert-base",
    training_split = "76aff95784fd4d4e",   # SHA-256 du split Jour 2
    lora_config    = {"r": 8, "alpha": 16,
                      "target_modules": ["query", "value"]},
    metrics_test   = {"f1_micro": 0.818, "f1_macro": 0.797,
                      "per_entity": {
                          "PER":   0.818, "LOC":   0.912,
                          "DATE":  0.889, "ORG":   0.610,
                          "TITLE": 0.756,
                      }},
    corpus_description = (
        "Corpus CREMMA médiéval, chartes normandes XIVe–XVe siècle. "
        "40 phrases annotées en BIO (5 types). "
        "Normalisation orthographique pipeline Jour 2 (CER ~5%)."
    ),
    known_limitations  = [
        "F1-ORG faible (0.61) : confusion LOC/ORG sur les institutions localisées.",
        "Prénoms féminins sous-représentés dans le corpus d'entraînement.",
        "Micro-toponymes ruraux non couverts par le gazetier.",
        "Hors-domaine : textes postérieurs à 1500 ou antérieurs à 1300.",
    ],
    bias_analysis = (
        "Corpus biaisé vers la noblesse masculine normande. "
        "F1-PER attendu plus faible sur personnes féminines (biais de représentation, "
        "cf. Chapitre 5 §7)."
    ),
)
```

### 6.3 Le choix entre API interne et API patrimoniale publique

Pour un projet de recherche, deux architectures de déploiement sont typiques.

**API interne :** le pipeline NLP tourne sur le serveur du laboratoire ou dans un environnement cloud privé, accessible uniquement depuis les outils internes (notebook Jupyter, interface d'annotation Recogito, scripts de traitement batch). La latence n'est pas critique, la disponibilité non plus. Un simple endpoint FastAPI avec un modèle chargé en mémoire est suffisant.

**API patrimoniale publique :** le pipeline est exposé à la communauté scientifique comme service interopérable (IIIF, API REST documentée OpenAPI). Les enjeux sont différents : documentation extensive, versionnement stable des endpoints, garantie de reproductibilité sur au moins 3 ans, gestion des accès, conformité RGPD. Dans ce cas, containeriser le service avec Docker et figer les versions de toutes les dépendances (`requirements.txt` avec hashes, `Dockerfile` reproducible) est une exigence minimale.

```dockerfile
# Dockerfile pour le pipeline NER médiéval
FROM python:3.11-slim

WORKDIR /app

# Dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Dépendances Python figées
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code et modèle
COPY src/          ./src/
COPY models/       ./models/
COPY MODEL_CARD.json .

# Variables d'environnement
ENV MODEL_PATH=/app/models/camembert_ner_lora
ENV MODEL_VERSION=1.0.0
ENV PORT=8000

EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.main:app",
     "--host", "0.0.0.0",
     "--port", "8000",
     "--workers", "1"]   # 1 worker = 1 modèle en VRAM
```

**Note sur le nombre de workers :** avec PyTorch, chaque worker (processus) charge le modèle indépendamment. Sur un serveur avec une seule T4 (16 Go), lancer 4 workers × 600 Mo = 2.4 Go pour les poids seuls — gérable. Mais avec un modèle de 4 Go, 2 workers suffisent. Préférer un seul worker avec un serveur asynchrone (uvicorn, asyncio) et du batching dynamique plutôt que plusieurs workers bloquants.

---

## 7. Synthèse : choisir la bonne configuration pour votre projet

Le tableau suivant résume les choix techniques recommandés selon le profil du projet :

| Critère | Projet de recherche (faible trafic) | API patrimoniale publique | Production commerciale |
|---|---|---|---|
| Quantisation | FP16 ou INT8 | INT8 | INT4 / AWQ |
| Distillation | Non nécessaire | Optionnelle | Recommandée |
| Serving | FastAPI simple | FastAPI + ONNX | vLLM / TGI |
| Batching | Statique (batch=4) | Dynamique | Continuous |
| Monitoring | Logs fichier | Prometheus + alertes | Prometheus + SLA |
| Versionnement | SHA-256 split + model card | Sémantique strict + changelog | CI/CD complet |
| Priorité 1 | Reproductibilité | Stabilité API | Throughput |
| Priorité 2 | Traçabilité | Documentation | Latence p99 |

**Pour le projet de fin de module** (pipeline NLP médiéval dockerisé) : INT8 avec ONNX Runtime ou bitsandbytes, FastAPI avec batching statique batch_size=4, logging structuré JSON, model card complète, Dockerfile reproductible. Ce profil couvre les exigences du livrable sans sur-ingénierie.

---

## Bibliographie de référence

### Quantisation

Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). **QLoRA: Efficient Finetuning of Quantized LLMs**. *NeurIPS 2023*. [arXiv:2305.14314](https://arxiv.org/abs/2305.14314) — NF4 et double quantisation.

Frantar, E., Ashkpour, S., Hoefler, T., & Alistarh, D. (2022). **GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers**. *ICLR 2023*. [arXiv:2210.17323](https://arxiv.org/abs/2210.17323)

Lin, J., Tang, J., Tang, H., Yang, S., Dang, X., & Han, S. (2023). **AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration**. [arXiv:2306.00978](https://arxiv.org/abs/2306.00978)

Dettmers, T., Lewis, M., Belkada, Y., & Zettlemoyer, L. (2022). **LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale**. *NeurIPS 2022*. [arXiv:2208.07339](https://arxiv.org/abs/2208.07339)

### Distillation

Hinton, G., Vinyals, O., & Dean, J. (2015). **Distilling the Knowledge in a Neural Network**. *NIPS Workshop 2015*. [arXiv:1503.02531](https://arxiv.org/abs/1503.02531) — Article fondateur.

Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). **DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter**. *EMC2 Workshop, NeurIPS 2019*. [arXiv:1910.01108](https://arxiv.org/abs/1910.01108)

Jiao, X., Yin, Y., Shang, L., Jiang, X., Chen, X., Li, L., Wang, F., & Liu, Q. (2019). **TinyBERT: Distilling BERT for Natural Language Understanding**. *EMNLP 2020 Findings*. [arXiv:1909.10351](https://arxiv.org/abs/1909.10351)

### Serving et optimisation d'inférence

Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J. E., Zhang, H., & Stoica, I. (2023). **Efficient Memory Management for Large Language Model Serving with PagedAttention**. *SOSP 2023*. [arXiv:2309.06180](https://arxiv.org/abs/2309.06180) — Fondement de vLLM.

ONNX Runtime (2023). [https://onnxruntime.ai](https://onnxruntime.ai) — Documentation officielle.

Hugging Face. **Text Generation Inference**. [GitHub: huggingface/text-generation-inference](https://github.com/huggingface/text-generation-inference)

### Monitoring et drift

Sculley, D., Holt, G., Golovin, D., Davydov, E., Phillips, T., Ebner, D., Chaudhary, V., Young, M., Crespo, J.-F., & Dennison, D. (2015). **Hidden Technical Debt in Machine Learning Systems**. *NeurIPS 2015*. — Sur les risques opérationnels des systèmes ML en production.

Widmer, G., & Kubat, M. (1996). **Learning in the presence of concept drift and hidden contexts**. *Machine Learning*, 23(1). — Fondements théoriques du concept drift.

### Humanités numériques et déploiement

Romary, L. (2015). **TEI and the challenge of servicing scholarly communities**. *Digital Scholarship in the Humanities*, 30(suppl. 1). — Sur les exigences spécifiques des API patrimoniales.

Jänicke, S., Franzini, G., Cheema, M. F., & Scheuermann, G. (2015). **On Close and Distant Reading in Digital Humanities: A Survey and Future Challenges**. *EuroVis 2015*. — Sur les différents profils d'utilisation des outils NLP en recherche humaniste.

---

*Support de cours rédigé pour le Master Data/IA · Module NLP · MD5 Volet 2 · 2026. Ce document accompagne le cours magistral du Jour 5 (09h00–11h30). Il est le prérequis théorique du TP Chapitre 10 (quantisation, benchmark, Docker, pytest) qui suit à 11h30.*
