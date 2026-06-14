# Chapitre 3 — Adapter les LLMs avec peu de ressources : PEFT, LoRA, QLoRA

**Module NLP · Master Data/IA · MD5 Volet 2 · 2026**  
Cours magistral — 3 heures

---

## Avant-propos : pourquoi ne pas faire de full fine-tuning

La réponse immédiate est calculatoire. CamemBERT-base possède 110 millions de paramètres. En float32 (4 octets par paramètre), stocker le modèle seul requiert 440 Mo. Mais pendant l'entraînement, il faut stocker en plus les gradients (440 Mo), et pour l'optimiseur Adam, deux moments supplémentaires — le moment du premier ordre $m_t$ et du second ordre $v_t$ (880 Mo). Total : environ 1,8 Go rien que pour l'optimiseur. En pratique, avec les activations intermédiaires des couches conservées pour la rétropropagation : 8 à 12 Go pour un modèle de 110 M paramètres. Hors de portée d'un GPU T4 (16 Go) avec un batch size raisonnable.

Pour T5-base (250 M paramètres) ou mT5-base (300 M) utilisé pour la normalisation orthographique du Jour 2, la facture double. Pour les modèles modernes — LLaMA-7B représente 7 milliards de paramètres, soit 28 Go en float32 — le full fine-tuning est simplement impossible sur du matériel académique standard.

La réponse plus profonde est que le full fine-tuning n'est souvent pas nécessaire. Les LLMs pré-entraînés sur de larges corpus ont déjà appris une représentation du langage très riche. Adapter ce modèle à une tâche spécifique — reconnaître des entités nommées médiévales, normaliser des graphies du XIVe siècle — ne requiert pas de modifier tous les paramètres : il suffit de modifier une petite fraction stratégiquement choisie. Cette intuition est le fondement des méthodes PEFT (*Parameter-Efficient Fine-Tuning*).

Ce chapitre a trois objectifs. Formaliser la mécanique de LoRA pour que les choix d'hyperparamètres du TP ne soient pas arbitraires. Expliquer QLoRA pour que vous compreniez ce que fait `bitsandbytes` quand vous chargez un modèle en 4 bits. Et positionner LoRA parmi les autres approches PEFT pour que vous sachiez pourquoi on ne fait pas de prefix tuning sur mT5.

---

## 1. Le paysage du fine-tuning : un spectre de compromis

### 1.1 Full fine-tuning : le cas de référence

En full fine-tuning, tous les paramètres $\theta$ du modèle pré-entraîné sont mis à jour par descente de gradient. Pour un modèle de paramètres $\theta_0$ (weights pré-entraînés), on minimise la loss sur la tâche cible :

$$\theta^* = \arg\min_\theta \mathcal{L}_{\text{tâche}}(\theta)$$

en initialisant depuis $\theta_0$ et en appliquant la rétropropagation sur tous les paramètres. C'est la méthode la plus expressive — tous les poids peuvent s'adapter — mais la plus coûteuse en mémoire, en calcul, et en risque de sur-apprentissage sur un corpus de taille limitée.

Le bilan mémoire d'Adam sur un modèle de $N$ paramètres en float32 est :

$$\text{Mémoire}_{\text{Adam}} = N \times (4 + 4 + 4 + 4) \text{ octets} = 16N \text{ octets}$$

soit 4 octets pour les paramètres, 4 pour les gradients, 4 pour $m_t$ (moment du premier ordre) et 4 pour $v_t$ (moment du second ordre). Pour CamemBERT-base ($N = 110 \times 10^6$) : $16 \times 110 \times 10^6 \approx 1{,}76 \text{ Go}$ rien pour l'optimiseur, avant même de compter les activations.

### 1.2 Les méthodes PEFT : taxonomie

Les approches PEFT partagent une philosophie commune : geler la majorité des poids du modèle pré-entraîné et n'entraîner qu'un petit nombre de paramètres additionnels ou sélectionnés. On peut les classer selon où et comment elles interviennent dans l'architecture.

**Ajout de paramètres** : des modules supplémentaires sont insérés dans l'architecture existante. Les *adapters* (Houlsby et al., 2019) insèrent de petits réseaux goulot d'étranglement entre les couches du Transformer. LoRA (Hu et al., 2022) ajoute des matrices de faible rang en parallèle des projections existantes. Le prefix tuning (Li & Lam, 2021) préfixe les clés et valeurs de l'attention avec des vecteurs entraînables.

**Sélection de paramètres** : une sous-ensemble des paramètres existants est sélectionné pour être mis à jour. BitFit (Zaken et al., 2022) n'entraîne que les biais du modèle — typiquement 0,1 % des paramètres — avec des résultats surprenants sur les tâches de classification.

**Reparamétrisation** : les paramètres entraînables sont exprimés dans un espace de dimension inférieure. LoRA appartient également à cette catégorie, ce qui en fait une méthode hybride.

**Prompting appris** : le prompt d'entrée est complété ou remplacé par des vecteurs continus entraînables. Le prompt tuning (Lester et al., 2021) n'entraîne que ces vecteurs de prompt, sans toucher à l'architecture.

Le tableau ci-dessous compare les principales approches sur les dimensions pertinentes pour votre module :

| Méthode | Paramètres entraînés | Mémoire | Inférence modifiée | Usage recommandé |
|---|---|---|---|---|
| Full fine-tuning | 100 % | Très élevée | Non | Corpus large, GPU puissant |
| LoRA | 0,1–1 % | Faible | Non (fusion possible) | Cas général, GPU limité |
| QLoRA | 0,1–1 % | Très faible | Non | Très grand modèle, GPU limité |
| Adapters | ~3 % | Modérée | Oui (latence +) | Tâches multiples sur un même modèle |
| Prefix tuning | 0,1 % | Très faible | Oui (séquence +) | Tâches de génération |
| Prompt tuning | < 0,01 % | Minimale | Oui (token +) | Modèles très grands, adaptation légère |
| BitFit | ~0,1 % | Minimale | Non | Classification, BERT-like |

---

## 2. LoRA : décomposition de rang faible

### 2.1 L'hypothèse fondamentale

LoRA (*Low-Rank Adaptation*) repose sur une observation empirique formulée par Hu et al. (2022) : lors du fine-tuning, les mises à jour des matrices de poids ont un *rang intrinsèque faible*. Autrement dit, les changements $\Delta W$ nécessaires pour adapter un modèle pré-entraîné à une tâche spécifique vivent dans un sous-espace de faible dimension par rapport à l'espace complet des paramètres.

Cette hypothèse s'appuie sur des travaux antérieurs de Li et al. (2018) montrant que le paysage de perte d'un réseau sur-paramétrisé possède une *dimension intrinsèque* bien inférieure au nombre total de paramètres — c'est-à-dire qu'on peut naviguer vers de bonnes solutions en se déplaçant dans un sous-espace de faible dimension.

Formellement, pour une matrice de poids pré-entraînée $W_0 \in \mathbb{R}^{d \times k}$, LoRA contraint sa mise à jour $\Delta W$ à être de rang $r \ll \min(d, k)$ :

$$\Delta W = BA$$

où $B \in \mathbb{R}^{d \times r}$ et $A \in \mathbb{R}^{r \times k}$ sont deux matrices entraînables, avec $r \ll \min(d, k)$.

Pendant le forward pass, la sortie de la couche adaptée est :

$$h = W_0 x + \Delta W x = W_0 x + B A x$$

Le modèle original $W_0$ est gelé ($\nabla_{W_0} \mathcal{L} = 0$) ; seules les matrices $A$ et $B$ reçoivent des gradients.

### 2.2 Initialisation et mise à l'échelle

L'initialisation est conçue pour que le début de l'entraînement soit neutre : $A$ est initialisée avec une distribution gaussienne de moyenne nulle, et $B$ est initialisée à zéro. Ainsi $\Delta W = BA = 0$ au démarrage — la perturbation LoRA n'affecte pas le comportement du modèle à l'initialisation.

La sortie est mise à l'échelle par un facteur $\frac{\alpha}{r}$ :

$$h = W_0 x + \frac{\alpha}{r} B A x$$

où $\alpha$ est un hyperparamètre (typiquement fixé à $2r$ ou à $r$, soit un ratio $\alpha/r = 1$ ou $2$). Ce facteur évite que la magnitude de $\Delta W$ dépende trop de $r$ : doubler $r$ sans ajuster $\alpha$ doublerait la contribution de LoRA, rendant les comparaisons entre valeurs de $r$ difficiles à interpréter.

**Choix pratique :** dans la plupart des implémentations (bibliothèque PEFT de HuggingFace), $\alpha$ est fixé indépendamment de $r$. Un réglage courant est $\alpha = 16$ avec $r = 8$, soit un ratio de 2. Mais l'essentiel est la cohérence : si vous changez $r$ dans une ablation, gardez $\alpha$ constant pour mesurer l'effet de $r$ seul.

### 2.3 Bilan paramétrique

Pour une matrice $W_0 \in \mathbb{R}^{d \times k}$, le full fine-tuning requiert $d \times k$ paramètres entraînables. LoRA de rang $r$ en requiert $r \times (d + k)$.

Pour la projection query de CamemBERT-base ($d = k = 768$, $r = 8$) :

$$\text{Full FT} : 768 \times 768 = 589\,824 \text{ paramètres}$$
$$\text{LoRA}_{r=8} : 8 \times (768 + 768) = 12\,288 \text{ paramètres}$$

soit un ratio de $589\,824 / 12\,288 \approx 48$. En adaptant les projections Q et V de toutes les couches de CamemBERT (12 couches, 2 matrices par couche) :

$$\text{Paramètres LoRA totaux} = 12 \times 2 \times 12\,288 = 295\,000 \approx 0{,}27\% \text{ de } 110M$$

C'est ce ratio — environ 0,3 % — qui permet de réduire l'empreinte mémoire de l'optimiseur d'un facteur 300.

### 2.4 Pourquoi Q et V, pas K

Rappel du Chapitre 1 : dans le mécanisme d'attention,

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

$W^Q$ projette chaque token vers l'espace des requêtes — il contrôle *ce que chaque token cherche* dans le contexte. $W^V$ projette chaque token vers l'espace des valeurs — il contrôle *ce qui est propagé* quand l'attention est forte. $W^K$ projette chaque token vers l'espace des clés — il contrôle *comment les tokens se laissent trouver*.

L'intuition derrière l'adaptation préférentielle de Q et V (et non K) est la suivante. Adapter $W^Q$ redirige l'attention vers des patterns pertinents pour la tâche cible : pour la NER médiévale, le modèle doit apprendre à "regarder" différemment autour des noms propres. Adapter $W^V$ change ce qui est transmis en aval quand un token est sélectionné : c'est le vecteur de valeur qui alimente les couches supérieures. $W^K$ joue un rôle plus passif : il expose les tokens à la comparaison, mais le résultat de cette comparaison est déjà capturé par les modifications de $W^Q$. Voita et al. (2019) ont montré empiriquement que certaines têtes d'attention sont facilement élagables — et que les clés sont les composantes les plus stables.

Ce choix n'est pas universel. Pour des tâches de génération longue, adapter également les projections de sortie $W^O$ peut apporter un gain marginal. Certaines configurations PEFT adaptent toutes les projections (`target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]`). L'ablation sur votre corpus répondra à cette question pour votre cas.

### 2.5 La fusion au moment de l'inférence

Un avantage pratique de LoRA est que les matrices $A$ et $B$ peuvent être fusionnées avec $W_0$ *après* l'entraînement, sans coût d'inférence supplémentaire :

$$W_{\text{merged}} = W_0 + \frac{\alpha}{r} B A$$

Le modèle résultant est structurellement identique au modèle original — même nombre de paramètres, même architecture. La latence d'inférence est inchangée. C'est une différence fondamentale avec les adapters, qui insèrent des couches supplémentaires et augmentent la latence de manière permanente.

En pratique avec PEFT :

```python
from peft import PeftModel

# Charger le modèle de base et les poids LoRA
model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-base")
model = PeftModel.from_pretrained(model, "chemin/vers/lora_checkpoint")

# Fusion : les matrices LoRA sont absorbées dans W_0
model = model.merge_and_unload()
# model est maintenant un modèle standard, sans overhead PEFT
```

---

## 3. Hyperparamètres LoRA : ce que chaque curseur fait

### 3.1 Le rang $r$

Le rang $r$ contrôle la *capacité* de l'adaptation LoRA : plus $r$ est grand, plus le sous-espace de mise à jour est riche, plus le modèle peut s'éloigner du pré-entraînement. C'est un compromis entre expressivité et régularisation implicite.

En pratique, les valeurs courantes vont de $r = 4$ à $r = 64$. Au-delà de $r = 64$, on approche un full fine-tuning partiel et le bénéfice en paramètres devient marginal. Hu et al. (2022) rapportent que $r = 4$ ou $r = 8$ suffisent pour la plupart des tâches NLP de classification et de génération, ce qui confirme empiriquement l'hypothèse du faible rang intrinsèque.

Votre ablation du TP (r=8 vs r=16) vous permettra de vérifier si cette conclusion tient sur votre corpus médiéval. L'hypothèse de travail est : pour une tâche de normalisation orthographique avec ~3000 paires d'entraînement, r=8 devrait suffire car les modifications nécessaires (règles graphiques systématiques) sont peu nombreuses et régulières.

### 3.2 $\alpha$ et le ratio $\alpha/r$

Comme expliqué en section 2.2, $\alpha$ met à l'échelle la contribution de LoRA. Ce qui compte dans la pratique est le ratio $\alpha/r$ :

- $\alpha/r = 1$ : contribution LoRA de même ordre de grandeur que les poids pré-entraînés, adaptation conservative.
- $\alpha/r = 2$ : ratio le plus courant dans les implémentations de référence.
- $\alpha/r > 4$ : adaptation agressive, risque de sur-apprentissage sur petits corpus.

Une recommandation pratique : fixer $\alpha = 2r$ pour les premières expériences, puis explorer si les résultats sont décevants.

### 3.3 Dropout LoRA

Un taux de dropout ($p = 0.05$ à $0.1$) est appliqué sur la sortie de la matrice $A$ avant multiplication par $B$. Ce dropout est une régularisation spécifique à LoRA : il force les matrices $A$ et $B$ à ne pas sur-spécialiser. Pour des corpus de taille réduite (typiquement < 5000 exemples), un dropout de 0.1 est recommandé. Pour des corpus plus larges, 0.05 ou 0 sont utilisables.

### 3.4 `target_modules` : quelles couches adapter

`target_modules` spécifie les noms des matrices à adapter. La valeur dépend de l'architecture :

```python
# CamemBERT / RoBERTa (encoder-only)
lora_config_camembert = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query", "value"],   # noms spécifiques à RoBERTa
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.TOKEN_CLS,        # classification de tokens (NER)
)

# mT5 / T5 (encoder-decoder)
lora_config_mt5 = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q", "v"],           # noms spécifiques à T5
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM,     # transduction de séquences
)
```

La différence de nommage (`"query"` vs `"q"`) reflète les conventions de chaque architecture. La bibliothèque PEFT inspecte les noms des sous-modules du modèle pour localiser les couches cibles — vous pouvez les lister avec `{name for name, _ in model.named_modules()}`.

### 3.5 Récapitulatif des décisions d'hyperparamètres

| Hyperparamètre | Valeur de départ | Quand l'ajuster |
|---|---|---|
| $r$ | 8 | Augmenter si underfitting constaté ; ne pas dépasser 64 |
| $\alpha$ | $2r$ | Garder ce ratio constant dans les ablations |
| `lora_dropout` | 0.1 | Réduire si corpus > 10 000 exemples |
| `target_modules` | `["q", "v"]` | Ajouter `"o"` si tâche de génération longue |
| `bias` | `"none"` | `"lora_only"` si gain marginal attendu sur les biais |

---

## 4. QLoRA : quantisation et LoRA combinés

### 4.1 Le problème de la quantisation naïve

La quantisation consiste à représenter les poids du modèle en précision réduite. Passer de float32 (4 octets) à int8 (1 octet) divise par 4 l'empreinte mémoire des poids. Mais la quantisation naïve détériore la qualité du modèle : les plages dynamiques des activations sont très variables selon les couches, et un simple mapping linéaire vers {-128, …, 127} introduit des erreurs de quantisation inacceptables sur les couches sensibles.

### 4.2 La quantisation NF4 (NormalFloat4)

QLoRA (Dettmers et al., 2023) introduit deux innovations techniques qui rendent la quantisation à 4 bits viables pour le fine-tuning.

La première innovation est le type de données **NF4** (*NormalFloat4*). La distribution des poids d'un LLM pré-entraîné est approximativement gaussienne de moyenne nulle. NF4 exploite cette propriété en définissant 16 niveaux de quantisation qui ne sont pas uniformément espacés, mais positionnés de façon à minimiser l'erreur de quantisation sous une distribution normale. Formellement, les niveaux $q_i$ sont définis comme les quantiles d'une distribution gaussienne standard :

$$q_i = Q_{\mathcal{N}(0,1)}\!\left(\frac{i + 0.5}{2^k}\right), \quad i = 0, \ldots, 2^k - 1$$

pour $k = 4$ bits. Cela donne une représentation optimale pour les poids gaussiens, avec une erreur de quantisation $\sim 2\times$ inférieure à la quantisation int4 uniforme.

La deuxième innovation est la **double quantisation** (*double quantization*) : les constantes de quantisation elles-mêmes (une par bloc de poids) sont quantisées en float8, réduisant leur empreinte de $\sim 0.5$ bits par paramètre.

### 4.3 La pagination mémoire avec `bitsandbytes`

QLoRA utilise la **pagination de la mémoire unifiée** NVIDIA pour gérer les pics de mémoire lors de la rétropropagation. Les pages mémoire CPU et GPU sont gérées par le driver NVIDIA comme un espace unifié : quand la mémoire GPU est saturée, les pages les moins utilisées sont automatiquement déplacées vers la RAM CPU. Cela permet d'entraîner des modèles qui dépasseraient sinon la capacité GPU, au prix d'une légère latence sur les transferts.

### 4.4 Le bilan mémoire de QLoRA

Pour un modèle de $N$ paramètres :

| Composante | Float32 | Float16 | QLoRA (NF4 + LoRA) |
|---|---|---|---|
| Poids du modèle | $4N$ octets | $2N$ octets | $\sim 0.5N$ octets |
| Optimiseur (Adam) | $8N$ octets | $8N$ octets | $\sim 8N_{\text{LoRA}}$ octets |
| Activations | Variable | Variable | Variable |

Pour LLaMA-7B ($N = 7 \times 10^9$) avec $r = 64$ ($N_{\text{LoRA}} \approx 160 M$) :
- Poids en NF4 : $0{,}5 \times 7 \times 10^9 \approx 3{,}5 \text{ Go}$
- Optimiseur sur $N_{\text{LoRA}}$ seulement : $8 \times 160 \times 10^6 \approx 1{,}3 \text{ Go}$
- Total estimé : **< 6 Go** — entraînable sur un T4 de 16 Go.

En pratique avec `bitsandbytes` :

```python
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import torch

# Configuration de quantisation NF4
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # NormalFloat4
    bnb_4bit_compute_dtype=torch.float16,# calculs en float16
    bnb_4bit_use_double_quant=True,      # double quantisation
)

# Chargement du modèle en 4 bits
model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/mt5-base",
    quantization_config=bnb_config,
    device_map="auto",                   # placement automatique sur GPU(s)
)

# Préparation pour l'entraînement k-bit (cast des layer norms en float32)
model = prepare_model_for_kbit_training(model)

# Ajout de LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q", "v"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Sortie typique : trainable params: 2,949,120 || all params: 302,432,256
#                  trainable%: 0.9754
```

### 4.5 Quand choisir QLoRA vs LoRA

QLoRA est nécessaire quand le modèle de base ne tient pas en mémoire GPU même en float16. Pour les modèles de ce module (CamemBERT 110 M, mT5-base 300 M), LoRA standard en float16 suffit. QLoRA prend tout son sens pour mT5-large (1,2 B), mT5-XL (3,7 B), ou tout modèle > 1 B sur un T4.

---

## 5. Les autres méthodes PEFT

### 5.1 Adapters (Houlsby et al., 2019)

Les adapters insèrent de petits modules entre les couches du Transformer. Un adapter est un réseau goulot d'étranglement : projection descendante vers une dimension $d_{\text{bottleneck}} \ll d_{\text{model}}$, activation non-linéaire, projection remontante vers $d_{\text{model}}$, connexion résiduelle.

$$\text{Adapter}(h) = h + W_{\text{up}} \cdot f(W_{\text{down}} \cdot h)$$

avec $W_{\text{down}} \in \mathbb{R}^{d_{\text{bottleneck}} \times d_{\text{model}}}$ et $W_{\text{up}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{bottleneck}}}$.

Les adapters ont deux avantages structurels sur LoRA. Premièrement, ils sont particulièrement adaptés au *multi-task learning* : on peut entraîner un seul modèle de base partagé et des adapters séparés par tâche, en swappant uniquement les modules légers à l'inférence. Deuxièmement, ils peuvent être insérés à n'importe quelle couche, pas seulement dans les projections d'attention. Leur inconvénient est une latence d'inférence augmentée : les modules supplémentaires ajoutent des opérations matricielles sur le chemin critique.

**Pour votre module :** les adapters sont moins pertinents que LoRA car vous n'avez pas de contrainte multi-tâche immédiate. Mais si au Jour 4 vous souhaitez adapter un même mT5 à la fois à la normalisation et à la résolution de coréférence, les adapters deviennent une option sérieuse.

### 5.2 Prefix Tuning (Li & Lam, 2021)

Le prefix tuning préfixe les séquences de clés et valeurs dans chaque couche d'attention avec des vecteurs entraînables $P_K, P_V \in \mathbb{R}^{l \times d_{\text{model}}}$, où $l$ est la longueur du préfixe (typiquement 10 à 100 tokens) :

$$\text{Attention}(Q, [P_K;\, K], [P_V;\, V])$$

Seuls les vecteurs du préfixe sont entraînés ; le modèle de base est entièrement gelé. Le préfixe "conditionne" chaque couche d'attention vers le comportement souhaité pour la tâche.

Le prefix tuning a montré des performances compétitives avec le fine-tuning complet sur les tâches de génération, mais est plus difficile à optimiser (le gradient doit traverser l'opération d'attention pour atteindre $P_K$ et $P_V$). Il est moins stable que LoRA pour les tâches de classification et séquence-à-séquence courtes.

### 5.3 Prompt Tuning (Lester et al., 2021)

Le prompt tuning est une simplification du prefix tuning : au lieu de préfixer chaque couche, on préfixe uniquement la couche d'embedding d'entrée avec un vecteur de prompt entraînable $p \in \mathbb{R}^{l \times d_{\text{model}}}$. Le modèle traite ces vecteurs comme des tokens supplémentaires en tête de séquence.

L'avantage est la simplicité extrême : aucune modification de l'architecture, seulement des paramètres d'entrée. L'inconvénient est la faible expressivité : le gradient ne peut influencer le comportement du modèle qu'indirectement, via les représentations de ces tokens virtuels.

Le prompt tuning atteint ses meilleures performances sur les très grands modèles (> 10 B paramètres), où le modèle a déjà suffisamment de capacité pour "interpréter" un prompt conditionneur. Sur CamemBERT-base (110 M), ses performances sont généralement inférieures à LoRA.

### 5.4 (IA)³ — Infused Adapter by Inhibiting and Amplifying Inner Activations

(IA)³ (Liu et al., 2022) est une méthode particulièrement économe : elle apprend trois vecteurs de mise à l'échelle $l_k, l_v, l_{ff}$ pour chaque couche, qui sont multipliés élément par élément avec les clés, les valeurs et les activations du FFN. Le nombre total de paramètres entraînés est infime — environ 0,01 % — mais les performances sont compétitives avec LoRA sur plusieurs benchmarks de few-shot.

L'intuition est que l'inhibition et l'amplification sélective de certaines dimensions des activations est suffisante pour reconfigurer le comportement du modèle pour une nouvelle tâche. C'est la méthode PEFT la plus légère disponible, utile quand la mémoire est extrêmement contrainte ou quand on cherche à minimiser l'overfitting sur un corpus minuscule.

---

## 6. Entraînement LoRA en pratique : recette pour mT5 et CamemBERT

### 6.1 Configuration complète pour mT5 (normalisation)

La tâche de normalisation orthographique est une transduction séquence-à-séquence : entrée *"au duc de norm~die"* → sortie *"au duc de normandie"*. C'est naturellement une tâche encoder-decoder, d'où le choix de mT5. La formulation concrète passe par le préfixe de tâche T5 :

```python
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                           Seq2SeqTrainer, Seq2SeqTrainingArguments,
                           DataCollatorForSeq2Seq)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset
import torch

MODEL_NAME = "google/mt5-base"
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)

# Préfixe de tâche T5 : indiquer au décodeur ce qu'on attend
PREFIX = "normalise moyen français: "

def preprocess(examples):
    inputs  = [PREFIX + t for t in examples["source"]]
    targets = examples["target"]
    model_inputs = tokenizer(inputs,  max_length=128, truncation=True, padding=False)
    labels       = tokenizer(targets, max_length=128, truncation=True, padding=False)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Construction du dataset (paires brut → normalisé)
train_data = Dataset.from_dict({
    "source": ["au duc de norm~die", "li roys signa l acte", "q~ feist il"],
    "target": ["au duc de normandie", "le roi signa l acte",  "que feist il"],
})
train_data = train_data.map(preprocess, batched=True, remove_columns=["source","target"])

# Configuration LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q", "v"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM,
)

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Arguments d'entraînement
training_args = Seq2SeqTrainingArguments(
    output_dir="./lora_mt5_normalisation",
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=3e-4,               # plus élevé que pour full FT : LoRA converge vite
    lr_scheduler_type="cosine",       # voir section 7
    warmup_ratio=0.1,                 # 10 % des steps pour le warmup
    weight_decay=0.01,
    fp16=True,                        # mixed precision
    predict_with_generate=True,       # nécessaire pour Seq2Seq
    generation_max_length=128,
    save_strategy="epoch",
    eval_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    data_collator=data_collator,
)
trainer.train()
```

### 6.2 Configuration complète pour CamemBERT (NER)

La NER est une classification de tokens : pour chaque token, prédire son label BIO (B-PER, I-PER, B-LOC, …, O). C'est une tâche encoder-only, d'où CamemBERT. Les labels des tokens `[CLS]`, `[SEP]` et les sous-tokens de continuation (produits par BPE) sont conventionnellement ignorés via la valeur spéciale $-100$.

```python
from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                           TrainingArguments, Trainer,
                           DataCollatorForTokenClassification)
from peft import get_peft_model, LoraConfig, TaskType

MODEL_NAME = "almanach/camembert-base"
LABELS     = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC",
               "B-DATE", "I-DATE", "B-ORG", "I-ORG", "B-TITLE", "I-TITLE"]
label2id   = {l: i for i, l in enumerate(LABELS)}
id2label   = {i: l for l, i in label2id.items()}

tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_and_align(examples):
    """
    Tokenise et aligne les labels BIO sur les sous-tokens BPE.
    Le premier sous-token d'un mot reçoit le label original ;
    les suivants reçoivent -100 (ignorés dans la loss).
    """
    tokenized = tokenizer(
        examples["tokens"], is_split_into_words=True,
        truncation=True, max_length=256,
    )
    all_labels = []
    for i, word_labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        aligned  = []
        prev_wid = None
        for wid in word_ids:
            if wid is None:
                aligned.append(-100)      # [CLS] et [SEP]
            elif wid != prev_wid:
                aligned.append(word_labels[wid])   # premier sous-token
            else:
                aligned.append(-100)      # sous-tokens suivants
            prev_wid = wid
        all_labels.append(aligned)
    tokenized["labels"] = all_labels
    return tokenized

# Configuration LoRA pour CamemBERT
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query", "value"],  # nommage RoBERTa/CamemBERT
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.TOKEN_CLS,
)

model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABELS),
    id2label=id2label,
    label2id=label2id,
)
model = get_peft_model(model, lora_config)
```

### 6.3 Le journal d'expériences

Le journal d'expériences est un fichier JSONL (*JSON Lines*) : une ligne JSON par run, avec tous les hyperparamètres, la date, le hash du split, et les métriques de validation. C'est l'artefact qui rend vos ablations interprétables.

```python
import json, hashlib, datetime

def log_experiment(config: dict, metrics: dict, split_hash: str,
                   journal_path: str = "experiments/journal.jsonl"):
    """Enregistre une expérience dans le journal JSONL."""
    entry = {
        "timestamp":  datetime.datetime.now().isoformat(),
        "split_hash": split_hash,
        "config":     config,
        "metrics":    metrics,
    }
    with open(journal_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# Exemple d'appel après un run LoRA
log_experiment(
    config={
        "model":          "google/mt5-base",
        "method":         "lora",
        "r":              8,
        "alpha":          16,
        "target_modules": ["q", "v"],
        "learning_rate":  3e-4,
        "epochs":         10,
        "batch_size":     8,
    },
    metrics={
        "eval_loss":    0.312,
        "token_accuracy": 0.847,
        "cer_normalized": 0.043,
    },
    split_hash="10532c040c7639e3a05ff88a5b3773f5...",
)
```

---

## 7. Scheduler cosine, warmup et arrêt prématuré

### 7.1 Pourquoi un scheduler

Le taux d'apprentissage est le paramètre le plus sensible de l'entraînement LoRA. Un learning rate trop élevé en début d'entraînement peut détruire les représentations pré-entraînées avant que les matrices LoRA aient convergé. Un learning rate trop faible en fin d'entraînement empêche l'exploration locale des dernières epochs.

Le scheduler cosine décroît le learning rate selon un cosinus de $\eta_{\text{max}}$ à $\eta_{\text{min}} \approx 0$ :

$$\eta_t = \eta_{\text{min}} + \frac{1}{2}(\eta_{\text{max}} - \eta_{\text{min}}) \left(1 + \cos\!\left(\frac{\pi t}{T}\right)\right)$$

où $t$ est le step courant et $T$ le nombre total de steps. La décroissance est douce et continue — le modèle "explore" moins aggressivement au fur et à mesure de la convergence.

### 7.2 Le warmup linéaire

Le warmup consiste à démarrer avec un learning rate très bas ($\sim 0$) et à l'augmenter linéairement jusqu'à $\eta_{\text{max}}$ sur les premiers `warmup_steps` steps. Pour LoRA, `warmup_ratio=0.1` (10 % des steps totaux) est un réglage robuste.

L'intuition : au début de l'entraînement, les matrices LoRA $A$ et $B$ sont aléatoires (ou nulles pour $B$). Un grand learning rate immédiat causerait des mises à jour de grande amplitude qui perturbent le modèle avant que les matrices LoRA ne soient "calibrées" sur la distribution de la tâche. Le warmup laisse les matrices converger doucement vers un régime stable.

### 7.3 L'arrêt prématuré (*early stopping*)

L'arrêt prématuré interrompt l'entraînement quand la métrique de validation ne s'améliore plus pendant `patience` epochs consécutives. Pour LoRA avec un corpus de taille limitée, c'est une précaution indispensable : le modèle peut sur-apprendre les particularités du corpus d'entraînement après quelques epochs, et continuer à entraîner ne fait que dégrader la généralisation.

```python
from transformers import EarlyStoppingCallback

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    data_collator=data_collator,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=3,        # tolérance de 3 epochs sans amélioration
            early_stopping_threshold=0.001,   # amélioration minimale significative
        )
    ],
)
```

**Interaction avec le scheduler cosine :** l'arrêt prématuré et le scheduler cosine peuvent interagir de façon contre-intuitive. Si l'arrêt intervient au step $t^* < T$, le learning rate à ce moment peut être encore élevé (milieu de la courbe cosinus). Certaines implémentations utilisent donc un scheduler à base de plateau (`ReduceLROnPlateau`) plutôt que cosinus quand l'arrêt prématuré est actif — le plateau s'adapte à la progression réelle de la validation.

---

## 8. Connexion aux décisions du module

### 8.1 Lecture du tableau d'ablation (Jour 2, TP)

Vous produirez un tableau d'ablation avec les colonnes suivantes :

| Configuration | CER val | Token acc. | Paramètres | Temps/epoch |
|---|---|---|---|---|
| Règles seules | — | — | 0 | < 1 s |
| LoRA r=8, Q+V | — | — | ~295 K | — |
| LoRA r=16, Q+V | — | — | ~590 K | — |
| LoRA r=8, Q+K+V+O | — | — | ~590 K | — |
| Full FT (si mémoire dispo.) | — | — | 300 M | — |

Pour interpréter ce tableau, il faut comprendre ce que chaque ligne teste :

LoRA r=8 vs r=16 teste l'hypothèse du faible rang : si r=16 fait significativement mieux que r=8, les mises à jour nécessaires sont dans un espace de dimension > 8 — ce qui suggère que la tâche est plus complexe qu'attendu, ou que votre corpus de normalisation contient des cas ambigus que r=8 ne peut pas capturer. Si r=8 et r=16 donnent des résultats similaires, le principe PEFT est validé.

LoRA r=8 Q+V vs Q+K+V+O teste si l'adaptation des projections de sortie et des clés apporte un gain sur votre tâche spécifique. Sur la normalisation, l'intuition est que K est peu utile (même raisonnement qu'au Chapitre 1) ; $W^O$ pourrait aider si la normalisation requiert des reformulations syntaxiques au-delà de la substitution graphique.

### 8.2 Interaction avec les scores de confiance HTR

Le syllabus mentionne explicitement : "les scores de confiance HTR alimentent les poids de perte : lignes incertaines pondérées moins fortement à l'entraînement." Concrètement, cela se traduit par une loss pondérée dans le Trainer :

```python
class WeightedSeq2SeqTrainer(Seq2SeqTrainer):
    """
    Trainer qui pondère la loss par la confiance HTR de chaque ligne.
    Chaque exemple du dataset doit contenir un champ "sample_weight"
    (float entre 0 et 1, typiquement = confidence globale de la ligne).
    """
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        weights = inputs.pop("sample_weight", None)
        outputs = model(**inputs)
        loss    = outputs.loss  # loss non pondérée (scalaire moyenné)

        if weights is not None:
            # Recalcul de la loss par exemple pour pouvoir pondérer
            logits = outputs.logits
            labels = inputs["labels"]
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
            per_token_loss = loss_fct(
                logits.view(-1, logits.size(-1)), labels.view(-1)
            ).view(labels.size())  # (batch, seq_len)
            # Moyenne par séquence
            per_seq_loss = per_token_loss.mean(dim=-1)  # (batch,)
            # Pondération et moyenne
            loss = (per_seq_loss * weights.to(per_seq_loss.device)).mean()

        return (loss, outputs) if return_outputs else loss
```

Ce point est non négociable dans le cahier des charges : un pipeline qui ignore `char_confidences` pour pondérer la loss traite de la même façon une transcription fiable à 95 % et une ligne bruitée à 65 %. Le biais introduit sur le corpus d'entraînement se retrouvera dans les erreurs du modèle en production.

---

## Bibliographie de référence

### Articles fondateurs PEFT

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). **LoRA: Low-Rank Adaptation of Large Language Models**. *ICLR 2022*. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)

Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). **QLoRA: Efficient Finetuning of Quantized LLMs**. *NeurIPS 2023*. [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)

Houlsby, N., Giurgiu, A., Jastrzebski, S., Morrone, B., de Laroussilhe, Q., Gesmundo, A., Attariyan, M., & Gelly, S. (2019). **Parameter-Efficient Transfer Learning for NLP** (Adapters). *ICML 2019*. [arXiv:1902.00751](https://arxiv.org/abs/1902.00751)

Li, X. L., & Liang, P. (2021). **Prefix-Tuning: Optimizing Continuous Prompts for Generation**. *ACL 2021*. [arXiv:2101.00190](https://arxiv.org/abs/2101.00190)

Lester, B., Al-Rfou, R., & Constant, N. (2021). **The Power of Scale for Parameter-Efficient Prompt Tuning**. *EMNLP 2021*. [arXiv:2104.08691](https://arxiv.org/abs/2104.08691)

Liu, H., Tam, D., Muqeeth, M., Mohta, J., Huang, T., Bansal, M., & Raffel, C. (2022). **Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning** ((IA)³). *NeurIPS 2022*. [arXiv:2205.05638](https://arxiv.org/abs/2205.05638)

Zaken, E. B., Ravfogel, S., & Goldberg, Y. (2022). **BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models**. *ACL 2022*. [arXiv:2106.10199](https://arxiv.org/abs/2106.10199)

### Fondements théoriques

Li, C., Farkhoor, H., Liu, R., & Yosinski, J. (2018). **Measuring the Intrinsic Dimension of Objective Landscapes**. *ICLR 2018*. [arXiv:1804.08838](https://arxiv.org/abs/1804.08838)

Aghajanyan, A., Zettlemoyer, L., & Gupta, S. (2021). **Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning**. *ACL 2021*. [arXiv:2012.13255](https://arxiv.org/abs/2012.13255)

### Quantisation

Dettmers, T., Lewis, M., Belkada, Y., & Zettlemoyer, L. (2022). **LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale**. *NeurIPS 2022*. [arXiv:2208.07339](https://arxiv.org/abs/2208.07339)

Frantar, E., Ashkboos, S., Hoefler, T., & Alistarh, D. (2023). **GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers**. *ICLR 2023*. [arXiv:2210.17323](https://arxiv.org/abs/2210.17323)

### Interprétabilité et sélection des couches

Voita, E., Talbot, D., Moiseev, F., Sennrich, R., & Titov, I. (2019). **Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned**. *ACL 2019*. [arXiv:1905.09418](https://arxiv.org/abs/1905.09418)

### Bibliothèques

Mangrulkar, S., Gugger, S., Debut, L., Belkada, Y., Paul, S., & Bossan, B. (2022). **PEFT: State-of-the-art Parameter-Efficient Fine-Tuning methods**. [GitHub HuggingFace/peft](https://github.com/huggingface/peft)

Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., ... & Rush, A. M. (2020). **Transformers: State-of-the-Art Natural Language Processing**. *EMNLP 2020*. [arXiv:1910.03771](https://arxiv.org/abs/1910.03771)

### Surveys PEFT

Ding, N., Qin, Y., Yang, G., Wei, F., Yang, Z., Su, Y., ... & Sun, M. (2023). **Parameter-efficient fine-tuning of large-scale pre-trained language models**. *Nature Machine Intelligence*, 5(3). [arXiv:2203.06904](https://arxiv.org/abs/2203.06904)

He, J., Zhou, C., Ma, X., Berg-Kirkpatrick, T., & Neubig, G. (2022). **Towards a Unified View of Parameter-Efficient Transfer Learning**. *ICLR 2022*. [arXiv:2110.04366](https://arxiv.org/abs/2110.04366)

### Manuels et ressources pédagogiques

Jurafsky, D., & Martin, J. H. (2024). *Speech and Language Processing* (3e éd., brouillon). Chapitre 11. [Disponible en ligne](https://web.stanford.edu/~jurafsky/slp3/)

Tunstall, L., von Werra, L., & Wolf, T. (2022). *Natural Language Processing with Transformers*. O'Reilly Media. Chapitres 3 et 8.

---

*Support de cours rédigé pour le Master Data/IA · Module NLP · MD5 Volet 2 · 2026. Ce document accompagne le cours magistral du Jour 2 (09h00–12h00). Il est le prérequis théorique du TP Chapitre 4 (normalisation orthographique LoRA mT5) de l'après-midi.*
