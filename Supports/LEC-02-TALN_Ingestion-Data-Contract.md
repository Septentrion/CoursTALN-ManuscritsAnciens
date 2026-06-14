# Chapitre 2 — Ingestion du data contract HTR → NLP

**Module NLP · Master Data/IA · MD5 Volet 2 · 2026**  
TP guidé — 3 heures 30

---

## Avant-propos : pourquoi un chapitre entier sur les données

Il est tentant, dans un module centré sur les modèles de langage, de passer directement au fine-tuning. C'est une erreur systématique des équipes qui produisent de mauvais résultats : elles entraînent un modèle sur des données qu'elles n'ont pas comprises.

Ce chapitre a un objectif concret et mesurable. À la fin de cette séance, vous saurez exactement combien d'abréviations non résolues votre corpus contient, quel est le taux de lignes `needs_review` par type de document, et quelle couverture de vocabulaire CamemBERT offre sur vos données. Un étudiant qui termine ce TP avec ces trois chiffres en main prend des décisions techniques éclairées pour les jours suivants. Les autres improviseront — et leurs ablations du Jour 2 seront ininterprétables faute d'une baseline propre.

Ce chapitre n'est pas préparatoire au travail réel : il *est* le travail réel. Comprendre ses données avant d'entraîner un modèle n'est pas une précaution méthodologique abstraite — c'est ce qui sépare une ingénierie NLP rigoureuse d'une série d'expériences dont personne ne peut interpréter les résultats.

---

## 1. Le data contract HTR → NLP : structure et garanties

### 1.1 Qu'est-ce qu'un data contract

Un *data contract* est un accord formel entre deux modules d'un pipeline sur le format, la sémantique et les garanties de qualité des données échangées. Dans votre pipeline, le Volet 1 (Computer Vision / HTR) produit un fichier JSON qui est le contrat d'entrée de votre module NLP. Ce contrat spécifie non seulement le format des données, mais aussi des métriques de qualité : le CER (*Character Error Rate*) estimé ne dépasse pas 12 %, le champ `needs_review` identifie les lignes problématiques, et les scores de confiance par caractère permettent de propager l'incertitude HTR dans les étapes aval.

La première chose à faire quand vous recevez ce JSON n'est pas de charger les transcriptions dans un tokeniseur. C'est de lire le schéma, de valider sa conformité, et de comprendre ce que chaque champ signifie opérationnellement.

### 1.2 Anatomie du JSON

La structure hiérarchique du data contract est la suivante :

```
document
├── document_id          : identifiant unique (ex. "charte_normandie_0042")
├── document_type        : "charte" | "registre" | "roman" | "liturgique"
├── metadata
│   ├── htr_model        : version du modèle HTR utilisé
│   ├── cer_estimate     : CER estimé sur le document (float, ex. 0.089)
│   ├── total_lines      : nombre total de lignes transcrites
│   └── needs_review_count : nombre de lignes flagguées
└── pages[]
    └── lines[]
        ├── line_id          : identifiant de ligne ("l001")
        ├── transcription    : texte transcrit (str)
        ├── confidence       : score de confiance global de la ligne (float, 0–1)
        ├── needs_review     : booléen — ligne à vérifier manuellement
        ├── char_confidences : liste de scores par caractère (list[float])
        ├── candidates       : alternatives HTR par position (list[dict])
        │   ├── position     : indice du caractère dans la transcription
        │   ├── alternatives : ["a", "u"] — les deux candidats HTR
        │   └── scores       : [0.51, 0.49] — probabilités associées
        └── polygon          : coordonnées du rectangle englobant sur l'image
                              [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
```

Voici un exemple complet de deux lignes représentatives — une ligne propre et une ligne `needs_review` :

```json
{
  "document_id": "charte_normandie_0042",
  "document_type": "charte",
  "metadata": {
    "htr_model": "cremma-medieval-v2",
    "cer_estimate": 0.089,
    "total_lines": 47,
    "needs_review_count": 6
  },
  "pages": [{
    "page_id": "p001",
    "lines": [
      {
        "line_id": "l001",
        "transcription": "Li roys de france signa l acte en son palays",
        "confidence": 0.923,
        "needs_review": false,
        "char_confidences": [0.99, 0.98, 0.97, 0.99, 0.96, 0.91, 0.95],
        "candidates": [],
        "polygon": [[120, 45], [340, 45], [340, 68], [120, 68]]
      },
      {
        "line_id": "l002",
        "transcription": "au duc de norm~die et au co~te de champagne",
        "confidence": 0.71,
        "needs_review": true,
        "char_confidences": [0.98, 0.97, 0.65, 0.60, 0.55, 0.72, 0.68],
        "candidates": [
          {"position": 13, "alternatives": ["~", "a"], "scores": [0.60, 0.40]},
          {"position": 28, "alternatives": ["~", "n"], "scores": [0.55, 0.45]}
        ],
        "polygon": [[120, 72], [360, 72], [360, 95], [120, 95]]
      }
    ]
  }]
}
```

### 1.3 Signification opérationnelle de chaque champ

**`confidence` (ligne)** est la moyenne géométrique des `char_confidences`. Un seuil typique de `needs_review` est fixé à 0.75 par le modèle HTR, mais ce seuil est paramétrable — vous devez vérifier quel seuil votre équipe Volet 1 a utilisé et s'il est cohérent avec votre distribution de données.

**`char_confidences`** est la liste des probabilités assignées par le modèle HTR à chaque caractère transcrit. Un score faible sur un caractère ne signifie pas que le caractère est faux — il signifie que le modèle était incertain. Cette nuance est fondamentale pour le Jour 2 : vous utiliserez ces scores comme *poids de perte* lors du fine-tuning et comme signal pour l'arbitrage par CamemBERT MLM.

**`candidates`** liste les positions où le modèle HTR a hésité entre deux caractères. La notation `[a/b]` du syllabus correspond à ce champ. Un candidat `{"position": 13, "alternatives": ["~", "a"], "scores": [0.60, 0.40]}` signifie que le modèle a transcrit `~` (tilde d'abréviation) avec 60 % de confiance, mais que `a` était l'alternative à 40 %. Ces positions sont précieuses : elles localisent exactement les ambiguïtés à résoudre.

**`needs_review`** est un flag binaire agrégé. Il peut être déclenché par plusieurs conditions : confidence globale inférieure au seuil, présence de caractères hors-alphabet, proportion élevée de caractères à confidence < 0.5, ou règles heuristiques spécifiques au modèle HTR. Vous devez décomposer cette agrégation pour comprendre *pourquoi* une ligne est flagguée.

**`polygon`** ancre chaque ligne à ses coordonnées pixel sur l'image. Ce champ sera exploité au Jour 4 pour construire le graphe de connaissances avec ancrage spatial. Ne le perdez pas : il est la seule connexion entre les entités textuelles et leur localisation physique sur le manuscrit.

### 1.4 Validation du schéma en Python

Avant toute analyse, validez le schéma. Un data contract sans validation est une source de bugs silencieux.

```python
import json
from pathlib import Path
import jsonschema

SCHEMA = {
    "type": "object",
    "required": ["document_id", "document_type", "metadata", "pages"],
    "properties": {
        "document_id":   {"type": "string"},
        "document_type": {"enum": ["charte", "registre", "roman", "liturgique"]},
        "metadata": {
            "type": "object",
            "required": ["htr_model", "cer_estimate", "total_lines"],
            "properties": {
                "cer_estimate": {"type": "number", "minimum": 0, "maximum": 1}
            }
        },
        "pages": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["lines"],
                "properties": {
                    "lines": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["line_id", "transcription", "confidence",
                                         "needs_review", "char_confidences",
                                         "candidates", "polygon"],
                            "properties": {
                                "confidence":      {"type": "number", "minimum": 0, "maximum": 1},
                                "needs_review":    {"type": "boolean"},
                                "char_confidences":{"type": "array",
                                                    "items": {"type": "number"}},
                                "polygon":         {"type": "array", "minItems": 4,
                                                    "maxItems": 4}
                            }
                        }
                    }
                }
            }
        }
    }
}

def load_and_validate(path: str) -> list[dict]:
    """
    Charge et valide un fichier data contract HTR.
    Retourne la liste aplatie de toutes les lignes avec leur contexte document.
    """
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)

    jsonschema.validate(doc, SCHEMA)   # lève jsonschema.ValidationError si invalide

    lines = []
    for page in doc["pages"]:
        for line in page["lines"]:
            lines.append({
                **line,
                "document_id":   doc["document_id"],
                "document_type": doc["document_type"],
                "page_id":       page["page_id"],
            })
    return lines

# Chargement de l'ensemble du corpus (plusieurs fichiers JSON)
corpus_dir = Path("data/htr_output/")
all_lines  = []
errors     = []

for json_file in sorted(corpus_dir.glob("*.json")):
    try:
        all_lines.extend(load_and_validate(str(json_file)))
    except jsonschema.ValidationError as e:
        errors.append((json_file.name, str(e.message)))

print(f"Lignes chargées : {len(all_lines)}")
print(f"Fichiers invalides : {len(errors)}")
for fname, msg in errors:
    print(f"  {fname}: {msg}")
```

---

## 2. Analyse des lignes `needs_review`

### 2.1 Pourquoi cette analyse est non négociable

Le flag `needs_review` est votre première ligne de défense contre la contamination du corpus d'entraînement. Inclure des lignes `needs_review` dans votre ensemble d'entraînement sans stratégie explicite, c'est entraîner un modèle sur du bruit que vous ne contrôlez pas. Mais les exclure systématiquement, c'est potentiellement éliminer des exemples rares et précieux — notamment si certains types de documents ont un taux `needs_review` structurellement plus élevé (ce que vous découvrirez dans votre EDA).

La bonne approche est de comprendre la distribution de ces lignes avant de décider de leur sort.

### 2.2 Taux par type de document

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame(all_lines)

# Taux de needs_review par type de document
review_by_type = (
    df.groupby("document_type")["needs_review"]
    .agg(["sum", "count", "mean"])
    .rename(columns={"sum": "n_review", "count": "n_total", "mean": "taux"})
    .sort_values("taux", ascending=False)
)
review_by_type["taux_pct"] = (review_by_type["taux"] * 100).round(1)
print(review_by_type.to_string())
```

Résultat typique sur un corpus CREMMA équilibré :

```
               n_review  n_total  taux_pct
document_type
roman               412     1203     34.2
registre            287      891     32.2
charte              156      743     21.0
liturgique           41      312     13.1
```

Ce tableau est déjà informatif : les romans médiévaux ont un taux `needs_review` 2.6 fois supérieur aux textes liturgiques. Cela reflète probablement des scripts plus variables, des abréviations plus denses, ou un manque de données d'entraînement HTR pour ce genre. Pour la NER du Jour 3, cela signifie que vos entités nommées dans les romans seront moins fiables — et que votre évaluation doit en tenir compte.

### 2.3 Décomposer les causes de `needs_review`

Le flag `needs_review` agrège plusieurs causes. Décomposez-les :

```python
import re

def diagnose_line(line: dict) -> list[str]:
    """Identifie les causes probables du flag needs_review."""
    causes = []
    conf = line["confidence"]
    char_confs = line["char_confidences"]
    trans = line["transcription"]

    if conf < 0.75:
        causes.append("confidence_globale_basse")
    if char_confs and min(char_confs) < 0.5:
        causes.append("caractere_tres_incertain")
    if len(line["candidates"]) > 0:
        causes.append("candidats_alternatifs")
    # Détection d'abréviations non résolues (tilde, lettres suspendues)
    if re.search(r'[~ñ]|[a-z]\^', trans):
        causes.append("abreviation_residuelle")
    # Caractères hors latin + diacritiques médiévaux
    if re.search(r'[^\x00-\x7F]', trans):
        causes.append("caractere_non_ascii")
    # Lignes très courtes (fragment ?)
    if len(trans.split()) < 3:
        causes.append("ligne_trop_courte")

    return causes if causes else ["cause_inconnue"]

# Appliquer le diagnostic
review_lines = df[df["needs_review"]].copy()
review_lines["causes"] = review_lines.apply(diagnose_line, axis=1)

# Compter les causes (une ligne peut avoir plusieurs causes)
from collections import Counter
cause_counter = Counter(c for causes in review_lines["causes"] for c in causes)
for cause, n in cause_counter.most_common():
    print(f"  {cause:35s} {n:4d} ({n/len(review_lines)*100:.1f} %)")
```

### 2.4 Stratégies de gestion

Une fois la distribution des causes établie, vous avez quatre stratégies possibles — et elles ne s'excluent pas mutuellement :

**Exclusion totale** : retirer toutes les lignes `needs_review` du corpus d'entraînement. Simple, traçable. Risque : biais de sélection si `needs_review` est corrélé au type d'entités ou au genre de document.

**Pondération différentielle** : inclure les lignes `needs_review` mais leur assigner un poids de perte inférieur (par exemple, proportionnel à la confiance globale). C'est l'approche recommandée dans le cahier des charges du module : les `char_confidences` alimentent directement les poids de perte lors du fine-tuning.

**Révision manuelle prioritaire** : soumettre les lignes `needs_review` à une révision humaine avant inclusion. Coûteux mais pertinent si le corpus est petit. À réserver aux lignes avec `confidence` > 0.65 (proches du seuil, potentiellement récupérables).

**Ensemble séparé** : traiter les lignes `needs_review` comme un ensemble de validation pour mesurer la robustesse du modèle aux entrées bruitées — ce qui sera précisément son contexte de production.

**Décision à documenter dans `CONVENTIONS_NLP.md`** : quelle que soit la stratégie retenue, elle doit être enregistrée avec son justificatif. Un choix non documenté est un biais non traçable.

---

## 3. Analyse exploratoire du corpus (EDA)

### 3.1 Distribution des longueurs de lignes

La longueur des lignes conditionne plusieurs décisions : le `SEQ_LEN` pour l'encodage, la stratégie de troncature et de padding, et le comportement de l'encodage positionnel sur les séquences longues.

```python
import numpy as np

df["n_tokens_brut"] = df["transcription"].apply(lambda t: len(t.split()))
df["n_chars"]       = df["transcription"].apply(len)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Distribution des longueurs en tokens
axes[0].hist(df["n_tokens_brut"], bins=40, edgecolor="white", color="#3A7EBF")
axes[0].axvline(df["n_tokens_brut"].median(), color="red",
                linestyle="--", label=f"Médiane : {df['n_tokens_brut'].median():.0f}")
axes[0].axvline(df["n_tokens_brut"].quantile(0.95), color="orange",
                linestyle="--", label=f"P95 : {df['n_tokens_brut'].quantile(0.95):.0f}")
axes[0].set_xlabel("Longueur (tokens bruts)")
axes[0].set_ylabel("Nombre de lignes")
axes[0].set_title("Distribution des longueurs de lignes")
axes[0].legend()

# Par type de document
for doc_type, grp in df.groupby("document_type"):
    axes[1].hist(grp["n_tokens_brut"], bins=30, alpha=0.5, label=doc_type)
axes[1].set_xlabel("Longueur (tokens bruts)")
axes[1].set_title("Longueurs par type de document")
axes[1].legend()

plt.tight_layout()
plt.savefig("eda_longueurs.pdf", dpi=150)
plt.show()

# Statistiques résumées
print(df.groupby("document_type")["n_tokens_brut"].describe().round(1))
```

**Ce que vous cherchez :** le percentile 95 de la longueur en tokens BPE (pas en mots bruts — les abréviations fragmentées augmentent ce compte). Si P95 dépasse 128 tokens, vous devrez décider de tronquer ou de segmenter les lignes. Pour BERT (limite à 512 tokens) ce n'est généralement pas un problème au niveau ligne, mais cela le devient si vous agrégez des lignes en paragraphes pour la modélisation thématique du Jour 4.

### 3.2 Distribution des scores de confiance HTR

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Histogramme global des confiances de ligne
axes[0].hist(df["confidence"], bins=50, edgecolor="white", color="#3A7EBF")
axes[0].axvline(0.75, color="red", linestyle="--", label="Seuil needs_review (0.75)")
axes[0].set_xlabel("Score de confiance global")
axes[0].set_ylabel("Nombre de lignes")
axes[0].set_title("Distribution des confiances HTR")
axes[0].legend()

# Distribution des char_confidences (toutes positions)
all_char_confs = [c for confs in df["char_confidences"] for c in confs]
axes[1].hist(all_char_confs, bins=50, edgecolor="white", color="#E87B3E")
axes[1].axvline(0.5, color="red", linestyle="--", label="Seuil critique (0.5)")
axes[1].axvline(0.7, color="orange", linestyle="--", label="Seuil attention (0.7)")
axes[1].set_xlabel("Confiance par caractère")
axes[1].set_title("Distribution des confiances par caractère")
axes[1].legend()

plt.tight_layout()
plt.savefig("eda_confiances.pdf", dpi=150)
plt.show()

# Taux de caractères critiques (< 0.5)
n_critique = sum(1 for c in all_char_confs if c < 0.5)
print(f"Caractères avec confidence < 0.5 : {n_critique/len(all_char_confs)*100:.1f} %")
print(f"Caractères avec confidence < 0.7 : "
      f"{sum(1 for c in all_char_confs if c < 0.7)/len(all_char_confs)*100:.1f} %")
```

**Interprétation attendue :** la distribution des confiances par caractère est typiquement bimodale sur un corpus médiéval : une masse concentrée autour de 0.95–0.99 (caractères courants bien reconnus), et une queue à gauche autour de 0.5–0.65 (abréviations, lettres inhabituelles, zone d'encre dégradée). Ce sont les caractères de la queue gauche qui constituent votre problème de normalisation au Jour 2.

### 3.3 Inventaire des abréviations non résolues

C'est l'analyse la plus directement opérationnelle de ce chapitre. Les abréviations non résolues par le modèle HTR apparaissent sous deux formes dans les transcriptions :

- **Tilde de nasalité** : `norm~die` pour *normandie*, `co~te` pour *conte*, `q~` pour *que*
- **Lettre superscrite** : `pñ` pour *prison*, `pñce` pour *prince*
- **Signes suprascripts** : `mp` pour *monsieur*, `sr` pour *seigneur*
- **Lettres suspendues** : le `p` barré pour *par/per/pro*, le `q` barré pour *que/qui*

```python
import re
from collections import Counter

# Patterns d'abréviations médiévales
ABBREV_PATTERNS = {
    "tilde_nasale":    r'\w+~\w*',          # co~te, norm~die, q~
    "lettre_suprascr": r'[a-z]\^[a-z]?',    # m^r, s^r
    "lettre_speciale": r'[ñ]',              # ñ comme marque d'abréviation
    "barre_p":         r'\bꝑ\b|\bꝕ\b',     # p barré unicode médiéval
    "tilde_seul":      r'\s~\s|^~|~$',      # tilde isolé (marqueur générique)
}

def extract_abbreviations(text: str) -> dict[str, list[str]]:
    found = {}
    for name, pattern in ABBREV_PATTERNS.items():
        matches = re.findall(pattern, text)
        if matches:
            found[name] = matches
    return found

# Inventaire global
abbrev_inventory = Counter()
abbrev_by_type   = {t: Counter() for t in df["document_type"].unique()}

for _, row in df.iterrows():
    found = extract_abbreviations(row["transcription"])
    for name, matches in found.items():
        abbrev_inventory[name] += len(matches)
        abbrev_by_type[row["document_type"]][name] += len(matches)

print("Inventaire global des abréviations non résolues :")
total_abbrev = sum(abbrev_inventory.values())
for name, count in abbrev_inventory.most_common():
    print(f"  {name:25s} : {count:5d}  ({count/total_abbrev*100:.1f} %)")

print(f"\nTotal abréviations non résolues : {total_abbrev}")
print(f"Lignes avec ≥1 abréviation      : "
      f"{df['transcription'].apply(lambda t: bool(extract_abbreviations(t))).sum()}")
```

**Ce chiffre est votre baseline.** Avant de lancer quoi que ce soit le Jour 2, vous saurez combien d'abréviations vous devez résoudre. Si votre corpus contient 847 tildes de nasalité, vous saurez que votre module de règles devra couvrir ce cas, et vous mesurerez combien il en résout effectivement.

**Constitution d'une table d'abréviations contextuelles** : certaines abréviations sont levables par contexte sans recours à un modèle. Constituez cette table dès maintenant :

```python
# Exemples de résolutions contextuelles — à compléter avec votre corpus
ABBREV_TABLE = {
    # Formules latines fréquentes dans les chartes
    "q~"    : "que",
    "Q~"    : "Que",
    "no~"   : "nom",
    "no~e"  : "nome",
    # Titres
    "sr"    : "seigneur",
    "S^r"   : "Seigneur",
    "m^e"   : "messire",
    # Mots courants — variantes graphiques fréquentes
    "norm~die": "normandie",
    "co~te"   : "conte",
    "champ~e" : "champagne",
    # Monnaies
    "l~"    : "livres",
    "s~"    : "sous",
    # Signes de nasalité
    # (résolution par position : avant consonne occlusive)
    # → géré par le module de règles du Jour 2
}

# Mesurer la couverture de la table sur le corpus
resolvable = sum(
    1 for _, row in df.iterrows()
    for abbrev in re.findall(r'\w*[~ñ]\w*', row["transcription"])
    if abbrev in ABBREV_TABLE
)
total_tildes = abbrev_inventory["tilde_nasale"]
print(f"Couverture table statique : {resolvable}/{total_tildes} "
      f"({resolvable/max(total_tildes,1)*100:.1f} %)")
```

**Livrable attendu :** un fichier `abbreviations_inventory.json` avec la table comptée par type de document et par patron, et la couverture mesurée de la table statique. C'est l'inventaire explicitement requis dans les livrables du Jour 1.

---

## 4. Tokenisation et couverture de vocabulaire

### 4.1 Le problème de fond

Rappel du Chapitre 1 : CamemBERT a été pré-entraîné avec un tokeniseur SentencePiece/Unigram sur 138 Go de français moderne (corpus OSCAR). Son vocabulaire de 32 000 tokens est optimisé pour ce registre. Le moyen français lui est étranger à trois titres :

- Les graphies médiévales (*roys*, *palays*, *chastelain*) ne correspondent pas aux formes modernes du vocabulaire.
- Les abréviations manuscrites (`norm~die`, `co~te`) sont entièrement hors-vocabulaire.
- Les caractères spéciaux médiévaux (lettres barrées, tildes, lettres superscrites) peuvent être absents du vocabulaire Unicode couvert.

La conséquence est une fragmentation accrue : là où CamemBERT encode *roi* en un seul token, il encode *roys* en deux ou trois sous-mots. Cette fragmentation dégrade la qualité des représentations contextuelles et augmente la longueur effective des séquences.

### 4.2 Mesure du taux OOV et de fragmentation

```python
from transformers import AutoTokenizer

tokenizer_camembert = AutoTokenizer.from_pretrained("almanach/camembert-base")
tokenizer_roberta   = AutoTokenizer.from_pretrained(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

def measure_coverage(texts: list[str], tokenizer, unk_token_id: int = None) -> dict:
    """
    Mesure le taux OOV et le facteur de fragmentation
    d'un tokeniseur sur une liste de textes.

    Taux OOV       : proportion de tokens résultant d'un [UNK] (vocabulaire absent).
    Fragmentation  : ratio (nombre de tokens BPE) / (nombre de mots bruts).
                     Une valeur de 1.0 = pas de fragmentation ; 2.0 = chaque mot
                     est découpé en deux sous-mots en moyenne.
    """
    if unk_token_id is None:
        unk_token_id = tokenizer.unk_token_id

    total_bpe_tokens  = 0
    total_unk_tokens  = 0
    total_words       = 0

    for text in texts:
        words = text.split()
        total_words += len(words)
        ids = tokenizer.encode(text, add_special_tokens=False)
        total_bpe_tokens += len(ids)
        total_unk_tokens += sum(1 for i in ids if i == unk_token_id)

    return {
        "taux_oov":     total_unk_tokens / max(total_bpe_tokens, 1),
        "fragmentation": total_bpe_tokens / max(total_words, 1),
        "n_tokens_bpe": total_bpe_tokens,
        "n_mots_bruts": total_words,
    }

# Séparation par type de document pour une analyse fine
all_texts = df["transcription"].tolist()
review_texts = df[df["needs_review"]]["transcription"].tolist()
clean_texts  = df[~df["needs_review"]]["transcription"].tolist()

print("Couverture CamemBERT :")
print("  Corpus complet   :", measure_coverage(all_texts, tokenizer_camembert))
print("  Lignes propres   :", measure_coverage(clean_texts, tokenizer_camembert))
print("  Lignes en revue  :", measure_coverage(review_texts, tokenizer_camembert))

print("\nCouverture mBERT (multilingue) :")
print("  Corpus complet   :", measure_coverage(all_texts, tokenizer_roberta))
```

**Résultats attendus sur CREMMA Medieval :**

| Tokeniseur | OOV global | Fragmentation (lignes propres) | Fragmentation (needs_review) |
|---|---|---|---|
| CamemBERT | 0.3–2 % | 1.4–1.8 | 2.1–2.8 |
| mBERT (multilingual) | 1–4 % | 1.6–2.0 | 2.4–3.2 |

Le taux OOV absolu est faible car SentencePiece peut toujours fragmenter en caractères individuels. L'indicateur plus sensible est le facteur de fragmentation : un facteur de 2.5 sur les lignes `needs_review` signifie que chaque mot brut génère en moyenne 2.5 tokens BPE, ce qui signifie que CamemBERT "voit" ces mots comme des suites de sous-mots inconnus.

### 4.3 Analyse par mot — identifier les formes les plus fragmentées

```python
from collections import defaultdict

def word_fragmentation_analysis(texts, tokenizer, top_k=30):
    """
    Retourne les mots les plus fragmentés (ratio tokens/caractères le plus élevé).
    """
    word_stats = defaultdict(lambda: {"count": 0, "total_tokens": 0})

    for text in texts:
        for word in text.split():
            tokens = tokenizer.tokenize(word)
            word_stats[word]["count"] += 1
            word_stats[word]["total_tokens"] += len(tokens)

    results = []
    for word, stats in word_stats.items():
        avg_tokens = stats["total_tokens"] / stats["count"]
        results.append({
            "mot":         word,
            "count":       stats["count"],
            "avg_tokens":  round(avg_tokens, 2),
            "example_tok": tokenizer.tokenize(word),
        })

    return sorted(results, key=lambda x: x["avg_tokens"], reverse=True)[:top_k]

fragmented = word_fragmentation_analysis(all_texts, tokenizer_camembert)
print("Mots les plus fragmentés par CamemBERT :")
for item in fragmented[:15]:
    print(f"  {item['mot']:20s} → {item['example_tok']}  (avg={item['avg_tokens']})")
```

Cette analyse produit la liste des formes médiévales les plus pénalisées par la tokenisation BPE. Ce sont précisément les formes que la normalisation du Jour 2 devra traiter en priorité : une fois normalisées (*chastelain* → *châtelain*, *roys* → *roi*), leur fragmentation diminue.

### 4.4 Stratégies alternatives : character-level vs BPE

Pour les textes médiévaux avec un taux de fragmentation élevé, deux alternatives à la tokenisation BPE standard méritent d'être envisagées :

**Tokenisation character-level** : chaque caractère est un token. Avantages : aucun OOV, couverture parfaite des graphies médiévales. Inconvénients : séquences 4 à 6 fois plus longues (coût quadratique de l'attention), perte des unités morphologiques. Pertinent uniquement pour des tâches de très bas niveau (correction caractère par caractère).

**ByT5 (byte-level)** : T5 opérant directement sur les octets UTF-8. Avantages : pas de tokeniseur, couverture universelle. Inconvénients : séquences encore plus longues. Testé avec succès sur des textes historiques dans quelques travaux récents (Xue et al., 2022).

**Tokeniseur ré-entraîné** : entraîner un tokeniseur SentencePiece directement sur votre corpus. Si votre corpus contient > 10 000 lignes, c'est envisageable. Sous ce seuil, le vocabulaire appris sera insuffisamment représentatif.

**Recommandation pratique pour ce module** : CamemBERT reste le choix par défaut car (1) les poids pré-entraînés capturent un français général qui aide même pour le moyen français, (2) la normalisation du Jour 2 réduit la fragmentation post-hoc, et (3) la communauté des humanités numériques produit ses données de benchmark sur CamemBERT (Martin et al., 2020). Mais vous devez avoir mesuré le coût de ce choix avant de le faire.

---

## 5. Visualisation des embeddings avec UMAP

### 5.1 Objectif de la visualisation

UMAP (*Uniform Manifold Approximation and Projection*, McInnes et al., 2018) est une technique de réduction de dimension non-linéaire qui préserve mieux la structure locale des données que t-SNE pour des corpus de taille intermédiaire. L'appliquer aux embeddings CamemBERT de vos lignes vous permet de :

- Vérifier que les types de documents forment des clusters distincts dans l'espace d'embedding — ce qui validerait que CamemBERT capte des différences stylistiques entre chartes, registres et romans.
- Identifier des anomalies : lignes aberrantes, documents mal classifiés, ou zones de chevauchement genre qui indiquent une frontière linguistique floue.
- Visualiser la séparation lignes propres / `needs_review` dans l'espace d'embedding.

```python
import torch
from transformers import AutoModel
import umap
import matplotlib.pyplot as plt
import numpy as np

def extract_cls_embeddings(texts: list[str],
                            model_name: str = "almanach/camembert-base",
                            batch_size: int = 32,
                            max_length: int = 128) -> np.ndarray:
    """
    Extrait les embeddings [CLS] de CamemBERT pour une liste de textes.
    Le token [CLS] agrège le contexte de toute la séquence — c'est la
    représentation standard pour les tâches de classification de séquence.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModel.from_pretrained(model_name)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # outputs.last_hidden_state : (batch, seq_len, hidden_size)
        # On extrait le vecteur du token [CLS] (position 0)
        cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(cls_emb)

        if i % (batch_size * 10) == 0:
            print(f"  Batch {i//batch_size}/{len(texts)//batch_size}...")

    return np.vstack(all_embeddings)

# Sous-corpus pour la visualisation (UMAP sur > 5000 lignes est lent sur CPU)
SAMPLE_SIZE = 500
sample_df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42)
sample_texts = sample_df["transcription"].tolist()

print("Extraction des embeddings CamemBERT...")
embeddings = extract_cls_embeddings(sample_texts)
print(f"Embeddings extraits : {embeddings.shape}")

# Réduction UMAP
print("Réduction UMAP...")
reducer = umap.UMAP(
    n_components=2,
    n_neighbors=15,    # contrôle le compromis local/global
    min_dist=0.1,      # compacité des clusters dans l'espace projeté
    metric="cosine",   # cosinus est standard pour les embeddings de langue
    random_state=42,
)
embedding_2d = reducer.fit_transform(embeddings)

# Visualisation
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Par type de document
type_colors = {"charte": "#3A7EBF", "registre": "#E87B3E",
               "roman": "#2CA02C", "liturgique": "#D62728"}
for doc_type, color in type_colors.items():
    mask = sample_df["document_type"] == doc_type
    axes[0].scatter(
        embedding_2d[mask, 0], embedding_2d[mask, 1],
        c=color, label=doc_type, alpha=0.6, s=20
    )
axes[0].set_title("Embeddings CamemBERT par type de document")
axes[0].legend()
axes[0].set_xlabel("UMAP 1")
axes[0].set_ylabel("UMAP 2")

# Par needs_review
colors_review = sample_df["needs_review"].map({True: "#D62728", False: "#3A7EBF"})
axes[1].scatter(embedding_2d[:, 0], embedding_2d[:, 1],
                c=colors_review, alpha=0.5, s=15)
# Légende manuelle
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor="#D62728", label="needs_review"),
                   Patch(facecolor="#3A7EBF", label="propre")]
axes[1].legend(handles=legend_elements)
axes[1].set_title("Embeddings CamemBERT : propre vs needs_review")
axes[1].set_xlabel("UMAP 1")

plt.tight_layout()
plt.savefig("umap_embeddings.pdf", dpi=150)
plt.show()
```

**Interprétation attendue :** si les types de documents forment des clusters relativement séparés, CamemBERT capte des différences stylistiques entre genres malgré la fragmentation BPE. Si `needs_review` est distribué uniformément dans l'espace d'embedding (et non concentré dans une région), le flag ne correspond pas à un type linguistique particulier — ce qui est plutôt rassurant.

**Point de vigilance :** UMAP préserve la structure locale mais pas nécessairement la structure globale. Des clusters visuellement séparés peuvent être plus proches dans l'espace de 768 dimensions qu'ils ne le semblent en 2D. Ne sur-interprétez pas les distances inter-clusters.

---

## 6. Verrouillage du split train/val/test

### 6.1 Pourquoi verrouiller avant toute analyse

Le split train/val/test doit être fixé *avant* toute analyse statistique approfondie du corpus. Si vous analysez la distribution des longueurs, des confiances, et des genres, puis que vous construisez un split "équilibré" en fonction de ces analyses, vous introduisez un biais de sélection : votre jeu de test aura une distribution proche de votre jeu d'entraînement par construction, et vos métriques seront optimistes.

La règle est simple : fixez le split sur un critère indépendant du contenu — par exemple, l'identifiant de document — et verrouillez-le avec un hachage SHA-256.

### 6.2 Stratégie de split

Pour les corpus de manuscrits médiévaux, deux contraintes s'imposent :

**Pas de fuite de document** : toutes les lignes d'un même document doivent être dans le même split. Sinon, les lignes d'entraînement et de test partagent la même main, le même scribe, le même style — le modèle "mémorise" les particularités du scribe plutôt que de généraliser.

**Stratification par type de document** : si votre corpus contient 70 % de chartes, 20 % de registres et 10 % de romans, chaque split doit respecter approximativement cette distribution.

```python
import hashlib
from sklearn.model_selection import GroupShuffleSplit

# Ratio standard pour un corpus de taille intermédiaire
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# Niveau document : une ligne = un document
doc_df = df.groupby("document_id").agg(
    document_type=("document_type", "first"),
    n_lines=("line_id", "count"),
).reset_index()

# Split stratifié au niveau document
splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_RATIO, random_state=42)
train_val_idx, test_idx = next(
    splitter.split(doc_df, groups=doc_df["document_type"])
)

doc_df_trainval = doc_df.iloc[train_val_idx]
test_doc_ids    = set(doc_df.iloc[test_idx]["document_id"])

val_size = int(len(doc_df_trainval) * VAL_RATIO / (TRAIN_RATIO + VAL_RATIO))
splitter2 = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=42)
train_idx, val_idx = next(
    splitter2.split(doc_df_trainval, groups=doc_df_trainval["document_type"])
)

train_doc_ids = set(doc_df_trainval.iloc[train_idx]["document_id"])
val_doc_ids   = set(doc_df_trainval.iloc[val_idx]["document_id"])

# Vérification : pas de chevauchement
assert not (train_doc_ids & val_doc_ids)
assert not (train_doc_ids & test_doc_ids)
assert not (val_doc_ids   & test_doc_ids)

# Application aux lignes
df["split"] = df["document_id"].map(
    lambda d: "train" if d in train_doc_ids
    else ("val" if d in val_doc_ids else "test")
)

# Résumé
print("Distribution du split :")
print(df.groupby(["split", "document_type"]).size().unstack(fill_value=0))
```

### 6.3 Verrouillage par SHA-256

```python
import hashlib, json

def compute_split_hash(df: pd.DataFrame) -> str:
    """
    Calcule un hash SHA-256 déterministe du split.
    Le hash est calculé sur la liste triée des (line_id, split) pour garantir
    la reproductibilité indépendamment de l'ordre des lignes dans le DataFrame.
    """
    split_manifest = sorted(
        zip(df["line_id"].tolist(), df["split"].tolist())
    )
    manifest_str = json.dumps(split_manifest, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(manifest_str.encode("utf-8")).hexdigest()

SPLIT_HASH = compute_split_hash(df)
print(f"SHA-256 du split : {SPLIT_HASH}")

# Sauvegarder le manifeste
split_manifest = {
    "hash_sha256":   SPLIT_HASH,
    "train_doc_ids": sorted(train_doc_ids),
    "val_doc_ids":   sorted(val_doc_ids),
    "test_doc_ids":  sorted(test_doc_ids),
    "n_train_lines": int((df["split"] == "train").sum()),
    "n_val_lines":   int((df["split"] == "val").sum()),
    "n_test_lines":  int((df["split"] == "test").sum()),
    "ratios":        {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO},
}

with open("split_manifest.json", "w", encoding="utf-8") as f:
    json.dump(split_manifest, f, indent=2, ensure_ascii=False)

print("Manifeste sauvegardé : split_manifest.json")
print("Ce fichier doit être versionné dans votre dépôt git et ne jamais être modifié.")
```

**Règle absolue** : une fois `split_manifest.json` versionné dans git et partagé avec votre équipe, il ne doit plus être modifié. Toute modification invalidé toutes les comparaisons de métriques antérieures. Si vous devez changer le split (corpus élargi, erreur de stratification), créez un nouveau manifeste avec un suffixe de version et documentez la raison dans `CONVENTIONS_NLP.md`.

---

## 7. Récapitulatif : les chiffres à connaître avant le Jour 2

À la fin de ce TP, vous devez être en mesure de remplir ce tableau sans consulter vos notebooks :

| Métrique | Valeur mesurée sur votre corpus |
|---|---|
| Nombre total de lignes | |
| Taux de lignes `needs_review` global | |
| Taux de lignes `needs_review` par type de document (4 valeurs) | |
| Cause principale du flag `needs_review` | |
| Nombre d'abréviations non résolues (tildes) | |
| Couverture de la table statique d'abréviations | |
| Facteur de fragmentation CamemBERT (lignes propres) | |
| Facteur de fragmentation CamemBERT (lignes `needs_review`) | |
| Longueur médiane en tokens BPE | |
| P95 de longueur en tokens BPE | |
| Hash SHA-256 du split | |

Ces chiffres constituent votre baseline. Ils permettront, au Jour 2, de mesurer l'impact du module de règles (combien d'abréviations résolues ? le facteur de fragmentation a-t-il diminué ?) et d'interpréter les ablations LoRA dans leur contexte de données.

---

## Bibliographie de référence

### Data contracts et qualité de données

Breck, E., Polyzotis, N., Roy, S., Whang, S., & Zinkevich, M. (2019). **Data Validation for Machine Learning**. *MLSys 2019*. [Disponible en ligne](https://mlsys.org/Conferences/2019/doc/2019/167.pdf)

### Corpus médiévaux et HTR

Pinche, A. (2022). **CREMMA Medieval : corpus de manuscrits médiévaux pour HTR**. [GitHub : HTR-United/CREMMA-Medieval](https://github.com/HTR-United/CREMMA-Medieval)

Camps, J.-B., Vinsonneau, A., & Clérice, T. (2021). **Corpus and Models for Lemmatisation and POS-tagging of Old French**. *Journal of Data Mining & Digital Humanities*.

Clerice, T., Chagué, A., & Romary, L. (2021). **HTR-United, Mutualisons la vérité de terrain !** *Humanistica 2021*. [HAL](https://hal.inria.fr/hal-03398076)

### Tokenisation et couverture OOV

Martin, L., Muller, B., Suárez, P. J. O., Dupont, Y., Romary, L., de la Clergerie, É. V., Seddah, D., & Sagot, B. (2020). **CamemBERT: a Tasty French Language Model**. *ACL 2020*. [arXiv:1911.03894](https://arxiv.org/abs/1911.03894)

Xue, L., Barua, A., Constant, N., Al-Rfou, R., Narang, S., Kale, M., Roberts, A., & Raffel, C. (2022). **ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models**. *TACL 2022*. [arXiv:2105.13626](https://arxiv.org/abs/2105.13626)

Sennrich, R., Haddow, B., & Birch, A. (2016). **Neural Machine Translation of Rare Words with Subword Units** (BPE). *ACL 2016*. [arXiv:1508.07909](https://arxiv.org/abs/1508.07909)

### Réduction de dimension et visualisation d'embeddings

McInnes, L., Healy, J., & Melville, J. (2018). **UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction**. [arXiv:1802.03426](https://arxiv.org/abs/1802.03426)

### Normalisation de textes historiques

Bollmann, M. (2019). **A Large-Scale Comparison of Historical Text Normalization Systems**. *NAACL 2019*. [arXiv:1904.02036](https://arxiv.org/abs/1904.02036)

Domingo, M., & Casacuberta, F. (2018). **Spelling Normalization of Historical Documents by Using a Machine Translation System**. *IWSLT 2018*.

Clérice, T. (2023). **Pie Extended** (normalisation et lemmatisation pour le français médiéval). [Zenodo](https://doi.org/10.5281/zenodo.3883589)

### Splits et reproductibilité

Bender, E. M., Gebru, T., McMillan-Major, A., & Shmitchell, S. (2021). **On the Dangers of Stochastic Parrots: Can Language Models Be Too Big?** *FAccT 2021*. — voir section sur la documentation des données et des splits.

Lhoest, Q., del Moral, A. V., Jernite, Y., Thakur, A., von Platen, P., Patil, S., ... & Delestre, J.-F. (2021). **Datasets: A Community Library for Natural Language Processing** (HuggingFace Datasets, incluant la gestion des splits SHA-256). *EMNLP 2021*. [arXiv:2109.02846](https://arxiv.org/abs/2109.02846)

### Outils de normalisation médiévale

DMF — *Dictionnaire du Moyen Français* (v.2020). ATILF — CNRS & Université de Lorraine. [atilf.fr/dmf](https://www.atilf.fr/dmf)

Tittel, S., Bermudez-Sabel, H., & Chiarcos, C. (2020). **Using RDFa to Link Text and Dictionary Data for Medieval French**. *Linked Data in Linguistics Workshop, LREC 2020*.

---

*Support de cours rédigé pour le Master Data/IA · Module NLP · MD5 Volet 2 · 2026. Ce document accompagne la séance TP du Jour 1 (13h30–17h00). Les livrables attendus en fin de séance sont : le notebook EDA, l'inventaire des abréviations (`abbreviations_inventory.json`), le rapport `needs_review` par type de document, et le manifeste de split verrouillé (`split_manifest.json`).*
