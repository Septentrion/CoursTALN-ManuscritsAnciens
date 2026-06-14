# Chapitre 6 — Pipeline NER bout en bout sur le corpus médiéval

**Module NLP · Master Data/IA · MD5 Volet 2 · 2026**  
TP guidé — 3 heures 30

---

## Avant-propos : de la théorie à la production

Le Chapitre 5 a posé les fondements : schémas BIO/BIOES, architecture CamemBERT+CRF, kappa de Cohen, propagation d'erreurs HTR. Ce chapitre en est la concrétisation. Vous allez construire un pipeline complet qui prend en entrée les transcriptions normalisées du Jour 2 et produit en sortie un corpus annoté en entités nommées, étiquettes POS, et lemmes, exporté dans les formats standards des humanités numériques (CoNLL-2003, CoNLL-U, data contract enrichi).

Quatre décisions structurent ce TP. Premièrement, les données d'entraînement sont les transcriptions normalisées, pas les transcriptions brutes — la justification quantitative est dans la simulation CER→F1 du Chapitre 5 : passer de CER 12 % à CER 5 % récupère environ 13 points de F1-NER. Deuxièmement, le modèle de fondation est CamemBERT, adapté par LoRA (Chapitre 3) plutôt qu'en full fine-tuning, pour des raisons de mémoire et de risque de sur-apprentissage sur un corpus annoté de taille limitée. Troisièmement, l'annotation POS et les lemmes sont produits par pie-extended et Stanza `frm`, pas déduits du modèle NER — les deux tâches sont résolues par des outils distincts adaptés à chacune. Quatrièmement, tout est exporté dans le data contract enrichi, qui est l'artefact de sortie du Volet 2 : la connexion avec les polygones d'image du Volet 1 est maintenue à travers le champ `polygon_ref`.

---

## 1. Stratégie de données : constituer le corpus d'entraînement NER

### 1.1 Les entrées disponibles

À l'issue du Jour 2, vous disposez de trois artefacts qui constituent ensemble vos données d'entraînement potentielles :

Le **split manifest** (`split_manifest.json`) avec son hash SHA-256 identifie exactement quelles lignes appartiennent à chaque partition (train / val / test). Ce manifest garantit que les métriques que vous calculerez ce soir sont comparables à celles du tableau d'ablation du Jour 2.

Le **corpus normalisé** — les sorties du pipeline règles + DMF + mT5-LoRA — est le texte sur lequel le modèle NER sera entraîné. Si vous n'avez pas atteint l'Étape 9 du TP Chapitre 4, utilisez les sorties du pipeline de règles seul : CER ≈ 4–5 %, ce qui est suffisant.

Le **journal d'expériences** (`experiments/journal.jsonl`) permet de tracer exactement quel modèle de normalisation a produit les données sur lesquelles le modèle NER est entraîné. Ce lien de traçabilité est non négociable dans un projet de recherche.

### 1.2 Annotation NER : bootstrap depuis un gazetier

Le corpus normalisé n'est pas encore annoté en entités. Annoter 200 lignes à la main prend deux à trois heures pour un expert. Pour amorcer l'annotation, on utilise une approche de *weak supervision* par gazetier (*gazetteer*) : une liste d'entités connues est utilisée pour annoter automatiquement le corpus, et un humain corrige les erreurs.

```python
# Gazetier minimal pour le corpus médiéval
GAZETTEER = {
    'PER':   {'jean', 'pierre', 'guillaume', 'thomas', 'robert', 'isabelle',
              'marguerite', 'charles', 'philippe', 'louis'},
    'LOC':   {'normandie', 'paris', 'france', 'bourgogne', 'champagne',
              'gisors', 'rouen', 'chartres', 'louvre', 'vincenne'},
    'DATE':  {'janvier', 'février', 'mars', 'avril', 'mai', 'juin',
              'juillet', 'août', 'septembre', 'octobre', 'novembre', 'décembre',
              'pâques', 'noël', 'pentecôte'},
    'ORG':   {'parlement', 'chapitre', 'abbaye', 'prévôté', 'cour'},
    'TITLE': {'seigneur', 'sénéchal', 'bailli', 'prévôt', 'châtelain',
              'monseigneur', 'messire', 'évêque', 'roi', 'duc', 'comte',
              'chevalier'},
}

def annotate_with_gazetteer(tokens: list[str],
                             gazetteer: dict,
                             scheme: str = "BIO") -> list[str]:
    """
    Annote une séquence de tokens par lookup dans le gazetier.
    Produit une annotation BIO ou BIOES.
    Stratégie : fenêtre glissante 1-gramme uniquement (pas d'entités multi-tokens).
    Les conflits (un token dans plusieurs catégories) sont résolus
    par priorité décroissante : TITLE > PER > ORG > LOC > DATE.
    """
    priority = ['TITLE', 'PER', 'ORG', 'LOC', 'DATE']
    labels   = ['O'] * len(tokens)

    for i, token in enumerate(tokens):
        tok_low = token.lower()
        for ent_type in priority:
            if tok_low in gazetteer.get(ent_type, set()):
                labels[i] = f'S-{ent_type}' if scheme == 'BIOES' else f'B-{ent_type}'
                break   # priorité respectée, passer au token suivant

    return labels
```

**Limite de cette approche :** le gazetier ne couvre pas les entités multi-tokens (*"jean de normandie"*) ni les variantes graphiques non normalisées (*"Jehan"*, *"Guillaumes"*). C'est précisément pour corriger ces cas que l'annotation humaine est indispensable. La weak supervision par gazetier fournit un pré-annotation rapide qui réduit le coût de l'annotation humaine d'environ 40 % ; elle ne la remplace pas.

### 1.3 Format CoNLL-2003

CoNLL-2003 est le format d'interchange standard pour la NER. Il est tabulé, une ligne par token, avec les colonnes séparées par des espaces. Les phrases sont séparées par des lignes vides. Un en-tête `-DOCSTART-` ouvre chaque document.

```
-DOCSTART- -X- O O

li          DET    _  O
sénéchal    NOUN   _  B-TITLE
jean        PROPN  _  B-PER
de          ADP    _  I-PER
normandie   PROPN  _  I-PER
porta       VERB   _  O
les         DET    _  O
lettres     NOUN   _  O

le          DET    _  O
roi         NOUN   _  B-TITLE
signa       VERB   _  O
l           DET    _  O
acte        NOUN   _  O
```

Les colonnes sont : forme, POS, chunk tag (souvent `_` en NER), étiquette NER. Cette structure à quatre colonnes est celle qu'attend `seqeval` et la grande majorité des frameworks de NER.

```python
def write_conll2003(sentences:  list[list[tuple]],
                    output_path: str) -> None:
    """
    Écrit un corpus au format CoNLL-2003.

    Paramètre sentences
    -------------------
    list de phrases, chaque phrase étant une liste de tuples
    (forme, pos, ner) où ner est une étiquette BIO ou BIOES.

    Exemple d'une phrase :
        [("li","DET","O"), ("sénéchal","NOUN","B-TITLE"), ...]
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('-DOCSTART- -X- O O\n\n')
        for sentence in sentences:
            for forme, pos, ner in sentence:
                f.write(f'{forme}\t{pos}\t_\t{ner}\n')
            f.write('\n')   # ligne vide entre phrases
```

---

## 2. Fine-tuning CamemBERT-NER avec LoRA

### 2.1 Architecture complète

L'architecture que vous construisez est CamemBERT + tête de classification linéaire, avec adaptateurs LoRA sur les projections Q et V des 12 couches d'attention. La tête de classification projette les représentations contextuelles de dimension 768 vers le nombre d'étiquettes ($T = 11$ pour BIO, $T = 21$ pour BIOES) :

$$\hat{y}_t = \text{softmax}(W_c \, h_t + b_c), \quad W_c \in \mathbb{R}^{T \times 768}$$

La tête ajoute $768 \times 11 + 11 = 8\,459$ paramètres, soit 0.008 % du modèle — elle est entraînable en full fine-tuning sans impact mémoire significatif. Ce sont les adaptateurs LoRA qui portent l'adaptation des représentations contextuelles.

**Bilan mémoire pour CamemBERT-base (110M params) avec LoRA r=8, Q+V :**

| Composante | Paramètres | Mémoire (float32) |
|---|---|---|
| CamemBERT gelé (float16) | 110 000 000 | 220 Mo |
| LoRA Q+V, r=8, 12 couches | 159 744 | 0.6 Mo (grad) |
| Adam sur LoRA | 159 744 | 1.3 Mo (moments) |
| Tête de classification | 8 459 | 0.03 Mo |
| Activations (batch=16, seq=128) | — | ~500 Mo |
| **Total GPU** | | **≈ 722 Mo ≈ 0.7 Go** |

Ce budget mémoire est confortable sur un GPU T4 (16 Go). La différence avec le full fine-tuning de CamemBERT (qui requiert ~1.8 Go pour l'optimiseur seul, plus les activations) est dramatique pour les modèles plus grands, moins visible ici — mais le risque de sur-apprentissage sur un corpus annoté limité justifie indépendamment le choix LoRA.

### 2.2 Configuration LoRA pour la NER

La configuration LoRA pour CamemBERT-NER suit les mêmes principes que pour mT5 au Chapitre 3, avec une adaptation au modèle encoder-only :

```python
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForTokenClassification

LABEL2ID = {
    'O':       0,
    'B-PER':   1, 'I-PER':   2,
    'B-LOC':   3, 'I-LOC':   4,
    'B-DATE':  5, 'I-DATE':  6,
    'B-ORG':   7, 'I-ORG':   8,
    'B-TITLE': 9, 'I-TITLE': 10,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

def build_camembert_ner(model_name: str = "almanach/camembert-base",
                         r:          int = 8,
                         alpha:      int = 16) -> tuple:
    """
    Construit CamemBERT pour la classification de tokens (NER)
    avec adaptateurs LoRA sur les projections Q et V.

    Retourne (model, lora_config)
    """
    # Modèle de base avec tête de classification NER
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels = len(LABEL2ID),
        id2label   = ID2LABEL,
        label2id   = LABEL2ID,
    )

    # Configuration LoRA
    # Pour CamemBERT (architecture RoBERTa), les projections Q et V
    # sont nommées "query" et "value" (pas "q" et "v" comme dans T5)
    lora_config = LoraConfig(
        r               = r,
        lora_alpha      = alpha,
        target_modules  = ["query", "value"],
        lora_dropout    = 0.1,
        bias            = "none",
        task_type       = TaskType.TOKEN_CLS,  # ← différence avec SEQ_2_SEQ_LM
        modules_to_save = ["classifier"],      # ← tête de classification entraînable en full
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, lora_config
```

**Remarque sur `modules_to_save` :** ce paramètre indique à PEFT que la tête de classification (`classifier`) doit être sauvegardée et chargée avec le checkpoint LoRA, même si elle n'est pas un adaptateur LoRA. Sans cette ligne, la tête de classification serait perdue lors du chargement du checkpoint.

**Remarque sur `target_modules` :** les noms des sous-modules varient selon l'architecture. Pour CamemBERT (héritant de RoBERTa), ce sont `"query"` et `"value"`. Pour T5/mT5 (Chapitre 3), c'était `"q"` et `"v"`. Toujours vérifier avec `model.named_modules()` avant de configurer LoRA.

### 2.3 Fonction de perte pondérée par la confiance HTR

Le Chapitre 3 a introduit le `WeightedSeq2SeqTrainer`. L'équivalent pour la classification de tokens pondère la loss par la confiance de la ligne HTR — les lignes moins fiables contribuent moins à l'entraînement :

```python
import torch
from transformers import Trainer

class WeightedTokenClassificationTrainer(Trainer):
    """
    Trainer pour la NER qui pondère la loss par la confiance HTR.
    Chaque exemple doit contenir un champ "sample_weight" (float entre 0 et 1).
    """
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        weights = inputs.pop("sample_weight", None)
        outputs = model(**inputs)
        logits  = outputs.logits                        # (batch, seq_len, n_labels)
        labels  = inputs["labels"]                     # (batch, seq_len)

        loss_fct = torch.nn.CrossEntropyLoss(
            reduction    = "none",
            ignore_index = -100,    # -100 pour les subwords non-premiers et padding
        )
        # Loss par token, forme (batch, seq_len)
        per_token_loss = loss_fct(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        ).view(labels.size())

        # Moyenne par séquence (en ignorant les -100)
        valid_mask    = labels != -100
        per_seq_loss  = (per_token_loss * valid_mask).sum(dim=1) / \
                        valid_mask.sum(dim=1).clamp(min=1)

        if weights is not None:
            loss = (per_seq_loss * weights.to(per_seq_loss.device)).mean()
        else:
            loss = per_seq_loss.mean()

        return (loss, outputs) if return_outputs else loss
```

Cette pondération est le dernier maillon de la chaîne qui commence au data contract du Volet 1 : les scores `char_confidences` alimentent la colonne `confidence` du corpus, qui devient `sample_weight` ici. Un pipeline qui ignore cette information traite de façon identique une ligne transcrite à 95 % de confiance et une ligne à 62 %.

### 2.4 Paramètres d'entraînement recommandés

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir              = "./camembert_ner_lora",
    num_train_epochs        = 20,           # NER converge plus lentement que norm.
    per_device_train_batch_size = 16,
    per_device_eval_batch_size  = 16,
    learning_rate           = 2e-4,         # légèrement plus bas que pour seq2seq
    lr_scheduler_type       = "cosine",
    warmup_ratio            = 0.1,
    fp16                    = True,         # float16 sur GPU
    eval_strategy           = "epoch",
    save_strategy           = "epoch",
    load_best_model_at_end  = True,
    metric_for_best_model   = "f1",
    greater_is_better       = True,
    logging_steps           = 10,
    report_to               = "none",
)
```

**Choix des hyperparamètres :** `num_train_epochs=20` peut sembler élevé par rapport aux 10 epochs du Chapitre 4. La NER sur un petit corpus converge plus lentement parce que les entités rares sont vues peu souvent — l'early stopping (patience=5) arrêtera l'entraînement si le F1 de validation stagne. Le learning rate de `2e-4` est légèrement plus conservateur que pour la normalisation : les gradients des adaptateurs LoRA pour la classification de tokens sont plus sensibles aux oscillations que pour la génération.

---

## 3. Annotation POS et lemmatisation avec pie-extended et Stanza

### 3.1 pie-extended en pratique

```python
# Installation : pip install pie-extended
# Téléchargement du modèle medieval-fr : pie-extended download medieval-fr

from pie_extended.pipeline import ExtensiblePipeline
from pie_extended.tagger import ExtensibleTagger

def annotate_pos_lemma_pie(texts: list[str],
                            model: str = "medieval-fr") -> list[list[dict]]:
    """
    Annote les textes normalisés avec pie-extended (POS + lemmatisation).

    Paramètre
    ---------
    texts : list[str]  lignes normalisées (une phrase par élément)

    Retourne
    --------
    list de listes de dicts, un dict par token :
        {"form": str, "lemma": str, "pos": str, "morph": str}
    """
    tagger = ExtensibleTagger.from_pretrained(model)
    results = []
    for text in texts:
        tokens = text.split()
        annotation = tagger.tag_tokens(tokens)
        sentence_result = []
        for token, ann in zip(tokens, annotation):
            sentence_result.append({
                "form":  token,
                "lemma": ann.get("lemma", token),
                "pos":   ann.get("pos",   "X"),
                "morph": ann.get("morph", "_"),
            })
        results.append(sentence_result)
    return results
```

**Schéma Cattex vs UPOS :** pie-extended produit des étiquettes en schéma Cattex (ex. `NOMcom`, `VERcjg`, `DArti`). Pour l'export CoNLL-U (qui utilise UPOS), une table de correspondance est nécessaire :

```python
CATTEX_TO_UPOS = {
    'NOMcom':  'NOUN',  'NOMpro':  'PROPN',
    'VERcjg':  'VERB',  'VERinf':  'VERB',  'VERppa': 'VERB',
    'DArti':   'DET',   'PROper':  'PRON',  'PROrel': 'PRON',
    'ADJqua':  'ADJ',   'ADVgen':  'ADV',
    'PRE':     'ADP',   'CONcoo':  'CCONJ', 'CONsub': 'SCONJ',
    'PONfrt':  'PUNCT', 'PONfbl':  'PUNCT',
    'NUMcar':  'NUM',   'ETR':     'X',
}

def cattex_to_upos(cattex_tag: str) -> str:
    """Convertit une étiquette Cattex en UPOS Universal Dependencies."""
    return CATTEX_TO_UPOS.get(cattex_tag, 'X')
```

### 3.2 Stanza pour les dépendances syntaxiques

pie-extended produit le POS et le lemme mais pas les arcs de dépendance. Pour les dépendances syntaxiques (nécessaires pour l'export CoNLL-U complet), Stanza `frm` est plus adapté :

```python
import stanza

def annotate_dependencies_stanza(text: str,
                                  lang: str = "frm") -> list[dict]:
    """
    Annote les dépendances syntaxiques avec Stanza.
    Complète l'annotation pie-extended avec les colonnes head et deprel.
    """
    nlp = stanza.Pipeline(
        lang                  = lang,
        processors            = "tokenize,mwt,pos,lemma,depparse",
        tokenize_pretokenized = True,
        use_gpu               = True,
    )
    doc = nlp(text)
    tokens = []
    for sent in doc.sentences:
        for word in sent.words:
            tokens.append({
                "id":     word.id,
                "form":   word.text,
                "lemma":  word.lemma,
                "upos":   word.upos,
                "xpos":   word.xpos,
                "feats":  word.feats or "_",
                "head":   word.head,
                "deprel": word.deprel,
            })
    return tokens
```

### 3.3 Export CoNLL-U complet

CoNLL-U est le format d'échange universel pour les corpus annotés en UD. Chaque ligne contient 10 champs tabulation-séparés :

```
ID  FORM      LEMMA     UPOS  XPOS    FEATS                    HEAD  DEPREL  DEPS  MISC
1   li        le        DET   DArti   Definite=Def|Gender=Masc  2     det     _     polygon_ref=bbox_230
2   sénéchal  sénéchal  NOUN  NOMcom  Gender=Masc|Number=Sing   3     nsubj   _     _
3   porta     porter    VERB  VERcjg  Mood=Ind|Tense=Past        0     root    _     _
```

La colonne MISC (champ 10) est l'endroit standard pour les métadonnées non-UD. C'est là que le `polygon_ref` du data contract Volet 1 est encodé : chaque token est ainsi lié à sa position physique dans le manuscrit numérisé.

```python
def export_conllu(sentences:   list[list[dict]],
                  ner_spans:   list[list[dict]],
                  output_path: str,
                  polygon_refs: dict | None = None) -> None:
    """
    Exporte un corpus annoté au format CoNLL-U.

    Paramètres
    ----------
    sentences   : list de phrases, chaque phrase = list de dicts token UD
    ner_spans   : list de listes de spans NER pour chaque phrase
    output_path : chemin du fichier de sortie
    polygon_refs: dict {token_global_id: polygon_ref_str} (facultatif)
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for sent_id, (sentence, spans) in enumerate(zip(sentences, ner_spans)):
            f.write(f'# sent_id = {sent_id + 1}\n')
            # Reconstruire le texte de la phrase
            text = ' '.join(t['form'] for t in sentence)
            f.write(f'# text = {text}\n')
            # Encoder les NER spans en commentaire (extension non-UD)
            if spans:
                ner_str = '; '.join(f"{s['text']}:{s['label']}" for s in spans)
                f.write(f'# ner_spans = {ner_str}\n')

            for token in sentence:
                misc = '_'
                if polygon_refs and token.get('global_id') in polygon_refs:
                    misc = f"polygon_ref={polygon_refs[token['global_id']]}"
                f.write(
                    f"{token['id']}\t{token['form']}\t{token.get('lemma','_')}\t"
                    f"{token.get('upos','X')}\t{token.get('xpos','_')}\t"
                    f"{token.get('feats','_')}\t{token.get('head',0)}\t"
                    f"{token.get('deprel','_')}\t_\t{misc}\n"
                )
            f.write('\n')
```

---

## 4. Évaluation avec seqeval

### 4.1 F1 micro et macro par type d'entité

`seqeval` est la bibliothèque de référence pour l'évaluation NER. Elle évalue au niveau *span* (entité entière), pas au niveau token : une entité *"jean de normandie"* doit être reconnue dans son intégralité (bons tokens, bon type, bonnes frontières) pour compter comme vrai positif. Une entité reconnue avec le bon type mais une frontière incorrecte est comptée comme faux positif *et* faux négatif.

```python
from seqeval.metrics import (classification_report,
                              f1_score, precision_score, recall_score)
from seqeval.scheme import IOB2   # pour le schéma BIO

def evaluate_ner(y_true: list[list[str]],
                 y_pred: list[list[str]]) -> dict:
    """
    Évalue un modèle NER avec seqeval.

    Paramètres
    ----------
    y_true : list de séquences d'étiquettes de référence (une liste par phrase)
    y_pred : list de séquences d'étiquettes prédites

    Retourne
    --------
    dict avec f1_micro, f1_macro, precision, recall,
    et le rapport complet par type d'entité.
    """
    report = classification_report(
        y_true, y_pred,
        scheme  = IOB2,
        output_dict = True,
        zero_division = 0,
    )
    return {
        "f1_micro":   f1_score(y_true, y_pred, average="micro",  scheme=IOB2),
        "f1_macro":   f1_score(y_true, y_pred, average="macro",  scheme=IOB2),
        "precision":  precision_score(y_true, y_pred, average="micro", scheme=IOB2),
        "recall":     recall_score(y_true, y_pred, average="micro",    scheme=IOB2),
        "per_entity": report,
    }
```

**Résultats attendus (valeurs indicatives sur corpus médiéval annoté, ~200 lignes) :**

| Entité | Précision | Rappel | F1 |
|---|---|---|---|
| PER | 0.849 | 0.789 | 0.818 |
| LOC | 0.925 | 0.899 | 0.912 |
| DATE | 0.903 | 0.875 | 0.889 |
| ORG | 0.667 | 0.562 | 0.610 |
| TITLE | 0.738 | 0.775 | 0.756 |
| **micro** | **0.836** | **0.800** | **0.818** |
| **macro** | — | — | **0.797** |

**F1 micro vs macro :** le F1 micro pèse chaque entité en proportion de sa fréquence dans le corpus — LOC et PER, les plus fréquentes, dominent le score. Le F1 macro donne le même poids à chaque type — ORG et TITLE, moins fréquentes et plus difficiles, tirent le score vers le bas. Pour un rapport scientifique, toujours reporter les deux.

**Pourquoi ORG est le plus faible :** les organisations médiévales (prévôtés, chapitres, cours) partagent des surfaces lexicales avec des LOC (*"la cour de Paris"*) et des TITLE (*"le chapitre"* sans nom propre). La frontière ontologique est ambiguë — c'est une source de désaccord inter-annotateurs avant d'être une limite du modèle.

### 4.2 Rapport de classification complet

```python
def print_ner_report(y_true: list[list[str]],
                     y_pred: list[list[str]]) -> None:
    """Affiche le rapport de classification NER avec seqeval."""
    print(classification_report(y_true, y_pred, scheme=IOB2, zero_division=0))

# Exemple de sortie attendue :
#               precision    recall  f1-score   support
#
#         DATE       0.90      0.88      0.89        32
#          LOC       0.93      0.90      0.91        69
#          ORG       0.67      0.56      0.61        32
#          PER       0.85      0.79      0.82        57
#        TITLE       0.74      0.78      0.76        40
#
#    micro avg       0.84      0.80      0.82       230
#    macro avg       0.82      0.78      0.80       230
# weighted avg       0.83      0.80      0.82       230
```

---

## 5. Analyse des erreurs

### 5.1 Matrice de confusion par type d'entité

L'analyse d'erreurs commence par une matrice de confusion entre types d'entités. Contrairement à la classification standard, la matrice NER inclut une ligne/colonne `O` pour les entités manquées et les faux positifs hors entité :

```python
import pandas as pd
from collections import defaultdict

def ner_confusion_matrix(y_true: list[list[str]],
                          y_pred: list[list[str]]) -> pd.DataFrame:
    """
    Construit une matrice de confusion au niveau span pour la NER.
    Lignes = type réel, colonnes = type prédit.
    La diagonale correspond aux vrais positifs.
    """
    from seqeval.metrics.sequence_labeling import get_entities
    confusion = defaultdict(lambda: defaultdict(int))

    for true_seq, pred_seq in zip(y_true, y_pred):
        true_entities = {(s, e, t) for t, s, e in get_entities(true_seq)}
        pred_entities = {(s, e, t) for t, s, e in get_entities(pred_seq)}

        # Vrais positifs : span et type corrects
        for span in true_entities & pred_entities:
            confusion[span[2]][span[2]] += 1

        # Erreurs de type : bon span, mauvais type
        true_spans = {(s, e): t for s, e, t in true_entities}
        pred_spans = {(s, e): t for s, e, t in pred_entities}
        for span, true_type in true_spans.items():
            pred_type = pred_spans.get(span, 'O')
            if pred_type != true_type:
                confusion[true_type][pred_type] += 1

        # Entités manquées (faux négatifs sur O)
        for span in true_entities - pred_entities:
            if span not in {(s,e,t) for s,e,t in pred_entities
                            if (s,e) == (span[0],span[1])}:
                confusion[span[2]]['O'] += 1

    types = sorted(set(k for d in confusion.values() for k in d) |
                   set(confusion.keys()))
    df = pd.DataFrame(
        [[confusion[t_true][t_pred] for t_pred in types] for t_true in types],
        index   = types,
        columns = types,
    )
    return df
```

### 5.2 Les cinq sources d'erreurs principales

**Confusion PER / LOC :** la source d'erreur la plus documentée dans les corpus médiévaux. *"Jean de Normandie"* contient un LOC (*"Normandie"*) imbriqué dans une PER. Le modèle BIO ne peut pas représenter cette imbrication : il doit choisir entre annoter l'ensemble comme PER ou annoter *"Normandie"* comme LOC à l'intérieur. La convention standard est d'annoter l'entité de niveau le plus haut (PER pour *"Jean de Normandie"*), mais les annotateurs novices l'inversent souvent.

**Confusion TITLE / PER :** *"le seigneur"* sans antécédent explicite désigne une personne par sa fonction. Le modèle aura tendance à annoter *"seigneur"* comme TITLE plutôt que de reconnaître son usage référentiel. Ce cas est ambigu même pour un expert — il justifie une règle explicite dans les consignes d'annotation.

**Entités multi-tokens avec abréviations résiduelles :** si le pipeline de normalisation n'a pas résolu *"Guill^me"* en *"Guillaume"*, l'entité PER sera fragmentée entre une forme non résolue et un contexte normalisé. Le modèle ne reconnaît pas le token abrégé comme faisant partie d'une PER.

**Lieux non répertoriés :** les micro-toponymes médiévaux (*"le bois de la Croix-aux-Moines"*, *"le marais de Pontoise"*) ne figurent pas dans les gazetiers ni dans les données pré-entraînées de CamemBERT. Ils constituent des *unseen entities* que seul le contexte syntaxique (*"le [LOC] de [LOC]"*) peut signaler.

**Abréviations résiduelles manquées :** les abréviations non résolues par le pipeline du Jour 2 — celles qui n'étaient ni dans `ABBREV_TABLE` ni dans le cache DMF — apparaissent dans le texte normalisé comme des tokens non standard (`normndie`, `seignr`). Le modèle NER ne les reconnaît pas comme entités. Cela illustre l'importance du pipeline hybride règles+neural du Chapitre 4 : chaque abréviation non résolue est une entité potentiellement manquée.

### 5.3 IAA comparée au modèle

Le kappa inter-annotateurs calculé à l'Étape 5 du TP sert de borne supérieure naturelle pour l'évaluation du modèle. Si le modèle atteint un F1 proche du kappa inter-annotateurs, il se comporte aussi bien qu'un annotateur humain — ce qui est le seuil au-delà duquel il devient difficile d'améliorer davantage sans réviser les consignes d'annotation elles-mêmes.

Un kappa de 0.76 sur vos 100 lignes annotées en croisé correspond, en termes de F1, à environ 0.82–0.85. Si votre modèle atteint F1 = 0.82 sur le jeu de test, la marge d'amélioration résiduelle est liée aux cas ambigus sur lesquels les humains eux-mêmes divergent — pas à un déficit du modèle.

---

## 6. Intégration dans le data contract NLP

### 6.1 Structure du data contract enrichi

Le data contract du Volet 1 stockait les métadonnées HTR (transcription, confidence, polygon). Le data contract NLP du Volet 2 enrichit chaque ligne avec les annotations produites ce jour :

```json
{
  "line_id":        "charte_1346_fol12_l03",
  "transcription":  "li sénéchal jean de normandie porta les lettres",
  "confidence":     0.87,
  "normalized":     "li sénéchal jean de normandie porta les lettres",
  "ner_spans": [
    {"start": 3,  "end": 11, "label": "TITLE", "text": "sénéchal"},
    {"start": 12, "end": 29, "label": "PER",   "text": "jean de normandie"}
  ],
  "pos_tags":  ["DET", "NOUN", "PROPN", "ADP", "PROPN", "VERB", "DET", "NOUN"],
  "lemmas":    ["le", "sénéchal", "jean", "de", "normandie", "porter", "le", "lettre"],
  "polygon_ref": "fol12_bbox_230_415_890_440"
}
```

**Invariants à vérifier :** les listes `pos_tags` et `lemmas` ont exactement la même longueur que `normalized.split()`. Les offsets `start` / `end` dans `ner_spans` sont des offsets caractères dans la chaîne `normalized` (pas des indices de tokens). Le champ `polygon_ref` reprend la valeur du data contract Volet 1 — il n'est pas recalculé ici.

```python
def build_enriched_contract(line_id:       str,
                             transcription: str,
                             confidence:    float,
                             normalized:    str,
                             ner_spans:     list[dict],
                             pos_tags:      list[str],
                             lemmas:        list[str],
                             polygon_ref:   str) -> dict:
    """
    Construit un enregistrement de data contract enrichi.
    Vérifie les invariants avant de retourner.
    """
    tokens = normalized.split()
    assert len(pos_tags) == len(tokens), \
        f"pos_tags ({len(pos_tags)}) != tokens ({len(tokens)})"
    assert len(lemmas) == len(tokens), \
        f"lemmas ({len(lemmas)}) != tokens ({len(tokens)})"
    for span in ner_spans:
        extracted = normalized[span['start']:span['end']]
        assert extracted == span['text'], \
            f"Offset incohérent : [{span['start']}:{span['end']}] = {extracted!r}, attendu {span['text']!r}"

    return {
        "line_id":       line_id,
        "transcription": transcription,
        "confidence":    confidence,
        "normalized":    normalized,
        "ner_spans":     ner_spans,
        "pos_tags":      pos_tags,
        "lemmas":        lemmas,
        "polygon_ref":   polygon_ref,
    }
```

### 6.2 Écriture et validation du corpus enrichi

```python
import json
from pathlib import Path

def write_enriched_corpus(records:     list[dict],
                           output_path: str) -> None:
    """
    Écrit le corpus enrichi en JSONL (une ligne JSON par enregistrement).
    Format choisi pour sa lisibilité et sa compatibilité avec les pipelines
    de traitement de données (Spark, HuggingFace datasets, etc.).
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"{len(records)} enregistrements écrits dans {output_path}")

def validate_corpus(records: list[dict]) -> dict:
    """Valide les invariants sur l'ensemble du corpus enrichi."""
    errors = []
    for i, r in enumerate(records):
        tokens = r['normalized'].split()
        if len(r['pos_tags']) != len(tokens):
            errors.append(f"[{i}] pos_tags length mismatch")
        if len(r['lemmas']) != len(tokens):
            errors.append(f"[{i}] lemmas length mismatch")
        for span in r.get('ner_spans', []):
            extracted = r['normalized'][span['start']:span['end']]
            if extracted != span['text']:
                errors.append(f"[{i}] span offset error: {span}")
    return {"n_records": len(records), "n_errors": len(errors), "errors": errors[:10]}
```

### 6.3 Lien avec le Chapitre 7

Le data contract enrichi produit ce jour est l'entrée directe du Chapitre 7. Les `ner_spans` alimenteront l'extraction de relations et la construction du graphe de connaissances. Les `lemmas` permettront d'indexer le corpus par forme canonique. Les `polygon_ref` assureront la traçabilité jusqu'à l'image source — condition nécessaire pour que les humanistes puissent vérifier chaque annotation dans le manuscrit original.

---

## 7. Point éthique : analyser les biais de représentation

### 7.1 Quoi mesurer

Le Chapitre 5 a décrit les sources de biais structurels dans les corpus médiévaux (section 7). Ce chapitre fournit les outils pour les mesurer concrètement sur votre propre corpus annoté.

```python
from collections import Counter

def analyze_ner_representation(records: list[dict]) -> dict:
    """
    Analyse la distribution des entités nommées dans le corpus annoté.
    Identifie les déséquilibres de représentation.
    """
    entity_texts  = {t: [] for t in ['PER','LOC','DATE','ORG','TITLE']}
    entity_counts = Counter()

    for record in records:
        for span in record.get('ner_spans', []):
            label = span['label']
            text  = span['text'].lower()
            entity_texts[label].append(text)
            entity_counts[label] += 1

    # Distribution par type
    total = sum(entity_counts.values())
    print("Distribution des entités :")
    for ent_type, count in entity_counts.most_common():
        print(f"  {ent_type:6s} : {count:4d} ({count/total*100:.1f}%)")

    # Pour PER : ratio prénoms masculins / féminins
    # (approximation par listes de prénoms connus)
    prenoms_fem = {'isabelle','marguerite','marie','jeanne','aliénor',
                   'blanche','mathilde','agnes','catherine','anne'}
    per_texts   = entity_texts.get('PER', [])
    n_fem = sum(1 for t in per_texts if any(p in t for p in prenoms_fem))
    n_total_per = len(per_texts)
    print(f"\nEntités PER féminines : {n_fem}/{n_total_per} "
          f"({n_fem/max(n_total_per,1)*100:.1f}%)")

    # Concentration géographique : top-5 LOC
    loc_freq = Counter(entity_texts.get('LOC', []))
    print("\nTop-5 LOC :")
    for loc, count in loc_freq.most_common(5):
        print(f"  {loc:20s} : {count}")

    return {"entity_counts": dict(entity_counts), "top_loc": dict(loc_freq.most_common(10))}
```

### 7.2 Interpréter les résultats

Si votre corpus produit plus de 85 % d'entités PER masculines, ce n'est pas le reflet de la société médiévale — c'est le reflet du corpus de chartes, qui documente les transactions de la noblesse masculine. Un modèle entraîné sur ce corpus sera mécaniquement plus précis sur les personnages masculins.

Si les cinq premiers LOC couvrent plus de 50 % des occurrences de lieux, le modèle reconnaîtra bien Paris, Normandie, et Rouen, mais échouera sur les micro-toponymes ruraux — précisément les lieux qui intéressent les historiens de la vie locale.

Ces constats ne disqualifient pas le modèle. Ils définissent son domaine de validité, que vous devez documenter dans la *model card* du checkpoint. Un utilisateur qui applique votre modèle NER à un corpus de registres paroissiaux gascons du XVe siècle doit savoir qu'il sort du domaine d'entraînement.

---

## Bibliographie de référence

### NER et fine-tuning

Lample, G., Ballesteros, M., Subramanian, S., Kawakami, K., & Dyer, C. (2016). **Neural Architectures for Named Entity Recognition**. *NAACL 2016*. [arXiv:1603.01360](https://arxiv.org/abs/1603.01360)

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**. *NAACL 2019*. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)

Martin, L., Muller, B., Suárez, P. J. O., Dupont, Y., Romary, L., de la Clergerie, É., Seddah, D., & Sagot, B. (2020). **CamemBERT: a Tasty French Language Model**. *ACL 2020*. [arXiv:1911.03894](https://arxiv.org/abs/1911.03894)

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). **LoRA: Low-Rank Adaptation of Large Language Models**. *ICLR 2022*. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)

### Annotation et formats

Tjong Kim Sang, E. F., & De Meulder, F. (2003). **Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition**. *CoNLL 2003*. — Référence fondatrice du format CoNLL-2003.

Nivre, J., et al. (2020). **Universal Dependencies v2: An Evergrowing Multilingual Treebank Collection**. *LREC 2020*. — Format CoNLL-U et schéma UD.

### Outils spécialisés

Camps, J.-B., Clérice, T., & Duval, F. (2021). **Corpus and Models for Lemmatisation and POS-tagging of Old French**. *Journal of Data Mining & Digital Humanities*. — Fondement de pie-extended.

Qi, P., Zhang, Y., Zhang, Y., Bolton, J., & Manning, C. D. (2020). **Stanza: A Python NLP Toolkit for Many Human Languages**. *ACL 2020 (Demos)*. [arXiv:2003.07082](https://arxiv.org/abs/2003.07082)

Nakayama, H., Kubo, T., Kamura, J., Taniguchi, Y., & Liang, X. (2018). **seqeval: A Python framework for sequence labeling evaluation**. [GitHub: chakki-works/seqeval](https://github.com/chakki-works/seqeval)

### Données et biais

Bender, E. M., Gebru, T., McMillan-Major, A., & Shmitchell, S. (2021). **On the Dangers of Stochastic Parrots**. *FAccT 2021*. — Sur les biais des corpus et leurs effets sur les modèles.

Hamdi, A., Linhares Pontes, E., Boros, E., Nguyen, T. T., Plafourcade, G., Cabrera-Diego, L. A., & Moreno, J.-G. (2021). **A Multilingual Dataset for NER, Entity Linking and Stance Detection in Historical Newspapers**. *ACL-IJCNLP 2021*.

### Data contracts et interopérabilité

Ide, N., & Pustejovsky, J. (2010). **What Does Interoperability Mean, Anyway? Toward an Operational Definition of Interoperability for Language Technology**. *SLTC 2010*. — Sur la standardisation des formats d'annotation NLP.

---

*Support de cours rédigé pour le Master Data/IA · Module NLP · MD5 Volet 2 · 2026. Ce document accompagne le TP guidé du Jour 3 (13h30–17h00). Les livrables attendus en fin de séance sont : le modèle NER évalué (F1 par catégorie), le corpus annoté CoNLL-2003, la mesure IAA sur 100 lignes, et le rapport d'analyse d'erreurs.*
