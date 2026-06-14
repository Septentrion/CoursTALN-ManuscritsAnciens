"""
TP Guidé — Pipeline NER bout en bout sur le corpus médiéval
          Segmentation · POS · Lemmatisation · NER · IAA · seqeval
Chapitres 5 & 6 — Module NLP · Master Data/IA · MD5 Volet 2 · 2026
─────────────────────────────────────────────────────────────────────
Pipeline implémenté :

  Texte normalisé (sortie Jour 2)
       ↓
  [Étape 1]  Tokenisation par règles + TBA
       ↓
  [Étape 2]  Annotation NER par gazetier (baseline weak supervision)
       ↓
  [Étape 3]  Export CoNLL-2003
       ↓
  [Étape 4]  IAA — simulation de deux annotateurs + kappa de Cohen
       ↓
  [Étape 5]  Évaluation seqeval (F1 micro/macro par type d'entité)
       ↓
  [Étape 6]  Analyse des erreurs (matrice de confusion, exemples)
       ↓
  [Étape 7]  Data contract enrichi + export CoNLL-U
       ↓
  [Étape 8]  Bilan mémoire CamemBERT-LoRA + alignement subwords
       ↓
  [Étape 9]  Fine-tuning CamemBERT-NER avec LoRA (CPU/GPU)
       ↓
  [Étape 10] Tableau d'ablation NER

Durée estimée : 3 h 30
Livrables :
  - corpus_ner.conll          (CoNLL-2003, POS + NER)
  - corpus_ner.conllu         (CoNLL-U, POS + lemmes + dépendances)
  - enriched_corpus.jsonl     (data contract enrichi)
  - iaa_report.json           (kappa + annotations des deux annotateurs)
  - error_report.json         (matrice de confusion + exemples)
  - experiments/journal.jsonl (mis à jour)

Instructions générales
──────────────────────
Les cellules  # TODO  contiennent des squelettes à compléter.
Les cellules  # FOURNI  sont à exécuter telles quelles.
Même convention que les TPs précédents : ne pas modifier les signatures.
"""

# %% [markdown]
# # TP — Pipeline NER bout en bout sur le corpus médiéval
#
# Ce TP prend en entrée les transcriptions normalisées du Jour 2 et produit
# un corpus entièrement annoté : entités nommées (NER), étiquettes
# morphosyntaxiques (POS), lemmes, et un data contract enrichi reliant
# chaque annotation à sa position physique dans le manuscrit.
#
# Le pipeline suit l'ordre naturel d'un projet de traitement de corpus
# historique : règles d'abord (gazetier, tokenisation), mesure d'accord
# (IAA), évaluation rigoureuse (seqeval), puis fine-tuning neural
# (CamemBERT-LoRA) pour les cas que les règles ne couvrent pas.

# %% [markdown]
# ## Partie 0 — Imports et corpus de travail

# %% [FOURNI]
import re, json, math, hashlib, random, os, datetime, collections
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from seqeval.metrics import (classification_report, f1_score,
                              precision_score, recall_score)
from seqeval.scheme import IOB2

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

print("Environnement prêt.")

# %% [markdown]
# ### Corpus de travail
#
# Quarante phrases médiévales annotées manuellement en BIO (5 types :
# PER, LOC, DATE, ORG, TITLE) avec POS Universal Dependencies et lemmes.
# Chaque phrase est un tuple `(tokens, ner_labels, pos_tags, lemmas)`.
#
# Ce corpus synthétique remplace les transcriptions CREMMA en TP de démonstration.
# En TP réel, chargez vos paires normalisées depuis `enriched_corpus.jsonl`
# produit au Jour 2.

# %% [FOURNI]
CORPUS_NER = [
    (["li","sénéchal","jean","de","normandie","porta","les","lettres"],
     ["O","B-TITLE","B-PER","I-PER","I-PER","O","O","O"],
     ["DET","NOUN","PROPN","ADP","PROPN","VERB","DET","NOUN"],
     ["le","sénéchal","jean","de","normandie","porter","le","lettre"]),
    (["le","roi","philippe","signa","l","acte","en","mars"],
     ["O","B-TITLE","B-PER","O","O","O","O","B-DATE"],
     ["DET","NOUN","PROPN","VERB","DET","NOUN","ADP","NOUN"],
     ["le","roi","philippe","signer","le","acte","en","mars"]),
    (["l","abbaye","de","saint","denis","reçut","les","terres"],
     ["O","B-ORG","I-ORG","I-ORG","I-ORG","O","O","O"],
     ["DET","NOUN","ADP","PROPN","PROPN","VERB","DET","NOUN"],
     ["le","abbaye","de","saint","denis","recevoir","le","terre"]),
    (["messire","guillaume","de","rouen","était","bailli"],
     ["B-TITLE","B-PER","I-PER","I-PER","O","B-TITLE"],
     ["NOUN","PROPN","ADP","PROPN","VERB","NOUN"],
     ["messire","guillaume","de","rouen","être","bailli"]),
    (["le","château","de","gisors","fut","assiégé","en","juillet"],
     ["O","O","O","B-LOC","O","O","O","B-DATE"],
     ["DET","NOUN","ADP","PROPN","VERB","VERB","ADP","NOUN"],
     ["le","château","de","gisors","être","assiéger","en","juillet"]),
    (["le","prévôt","de","paris","rendit","son","jugement"],
     ["O","B-TITLE","O","B-LOC","O","O","O"],
     ["DET","NOUN","ADP","PROPN","VERB","DET","NOUN"],
     ["le","prévôt","de","paris","rendre","son","jugement"]),
    (["isabelle","de","champagne","donna","ses","terres","au","roi"],
     ["B-PER","I-PER","I-PER","O","O","O","O","B-TITLE"],
     ["PROPN","ADP","PROPN","VERB","DET","NOUN","ADP","NOUN"],
     ["isabelle","de","champagne","donner","son","terre","à","roi"]),
    (["le","chapitre","de","chartres","tint","son","assemblée"],
     ["O","B-ORG","I-ORG","I-ORG","O","O","O"],
     ["DET","NOUN","ADP","PROPN","VERB","DET","NOUN"],
     ["le","chapitre","de","chartres","tenir","son","assemblée"]),
    (["à","pâques","le","comte","de","bourgogne","revint"],
     ["O","B-DATE","O","B-TITLE","O","B-LOC","O"],
     ["ADP","NOUN","DET","NOUN","ADP","PROPN","VERB"],
     ["à","pâques","le","comte","de","bourgogne","revenir"]),
    (["le","bailli","thomas","leva","les","taxes","en","normandie"],
     ["O","B-TITLE","B-PER","O","O","O","O","B-LOC"],
     ["DET","NOUN","PROPN","VERB","DET","NOUN","ADP","PROPN"],
     ["le","bailli","thomas","lever","le","taxe","en","normandie"]),
    (["le","duc","charles","de","france","régna","vingt","ans"],
     ["O","B-TITLE","B-PER","I-PER","I-PER","O","O","O"],
     ["DET","NOUN","PROPN","ADP","PROPN","VERB","NUM","NOUN"],
     ["le","duc","charles","de","france","régner","vingt","an"]),
    (["la","cour","du","parlement","de","paris","se","réunit"],
     ["O","B-ORG","I-ORG","I-ORG","I-ORG","I-ORG","O","O"],
     ["DET","NOUN","ADP","NOUN","ADP","PROPN","PRON","VERB"],
     ["le","cour","de","parlement","de","paris","se","réunir"]),
    (["monseigneur","pierre","porta","les","missives","au","roi"],
     ["B-TITLE","B-PER","O","O","O","O","B-TITLE"],
     ["NOUN","PROPN","VERB","DET","NOUN","ADP","NOUN"],
     ["monseigneur","pierre","porter","le","missive","à","roi"]),
    (["la","prévôté","de","rouen","fut","supprimée"],
     ["O","B-ORG","I-ORG","I-ORG","O","O"],
     ["DET","NOUN","ADP","PROPN","VERB","VERB"],
     ["le","prévôté","de","rouen","être","supprimer"]),
    (["en","l","an","mil","trois","cent","quarante","six"],
     ["O","O","O","B-DATE","I-DATE","I-DATE","I-DATE","I-DATE"],
     ["ADP","DET","NOUN","NUM","NUM","NUM","NUM","NUM"],
     ["en","le","an","mil","trois","cent","quarante","six"]),
    (["le","seigneur","de","vincenne","arma","ses","chevaliers"],
     ["O","B-TITLE","O","B-LOC","O","O","O"],
     ["DET","NOUN","ADP","PROPN","VERB","DET","NOUN"],
     ["le","seigneur","de","vincenne","armer","son","chevalier"]),
    (["marguerite","de","flandre","hérita","du","comté"],
     ["B-PER","I-PER","I-PER","O","O","O"],
     ["PROPN","ADP","PROPN","VERB","ADP","NOUN"],
     ["marguerite","de","flandre","hériter","de","comté"]),
    (["le","châtelain","du","louvre","reçut","l","ordre"],
     ["O","B-TITLE","O","B-LOC","O","O","O"],
     ["DET","NOUN","ADP","PROPN","VERB","DET","NOUN"],
     ["le","châtelain","de","louvre","recevoir","le","ordre"]),
    (["après","noël","le","roi","convoqua","les","barons"],
     ["O","B-DATE","O","B-TITLE","O","O","O"],
     ["ADP","NOUN","DET","NOUN","VERB","DET","NOUN"],
     ["après","noël","le","roi","convoquer","le","baron"]),
    (["jean","de","gisors","chevalier","signa","l","acte"],
     ["B-PER","I-PER","I-PER","B-TITLE","O","O","O"],
     ["PROPN","ADP","PROPN","NOUN","VERB","DET","NOUN"],
     ["jean","de","gisors","chevalier","signer","le","acte"]),
    (["la","dame","alice","de","bretagne","vendit","ses","biens"],
     ["O","O","B-PER","I-PER","I-PER","O","O","O"],
     ["DET","NOUN","PROPN","ADP","PROPN","VERB","DET","NOUN"],
     ["le","dame","alice","de","bretagne","vendre","son","bien"]),
    (["le","évêque","de","paris","prêcha","en","carême"],
     ["O","B-TITLE","O","B-LOC","O","O","B-DATE"],
     ["DET","NOUN","ADP","PROPN","VERB","ADP","NOUN"],
     ["le","évêque","de","paris","prêcher","en","carême"]),
    (["le","prudhomme","robert","témoigna","devant","la","cour"],
     ["O","B-TITLE","B-PER","O","O","O","O"],
     ["DET","NOUN","PROPN","VERB","ADP","DET","NOUN"],
     ["le","prudhomme","robert","témoigner","devant","le","cour"]),
    (["la","pentecôte","marqua","le","début","du","procès"],
     ["O","B-DATE","O","O","O","O","O"],
     ["DET","NOUN","VERB","DET","NOUN","ADP","NOUN"],
     ["le","pentecôte","marquer","le","début","de","procès"]),
    (["le","comte","de","champagne","et","de","brie","signa"],
     ["O","B-TITLE","O","B-LOC","O","O","B-LOC","O"],
     ["DET","NOUN","ADP","PROPN","CCONJ","ADP","PROPN","VERB"],
     ["le","comte","de","champagne","et","de","brie","signer"]),
    (["l","abbé","de","cluny","reçut","les","moines"],
     ["O","B-TITLE","O","B-LOC","O","O","O"],
     ["DET","NOUN","ADP","PROPN","VERB","DET","NOUN"],
     ["le","abbé","de","cluny","recevoir","le","moine"]),
    (["thomas","de","coucy","fut","fait","prisonnier"],
     ["B-PER","I-PER","I-PER","O","O","O"],
     ["PROPN","ADP","PROPN","VERB","VERB","NOUN"],
     ["thomas","de","coucy","être","faire","prisonnier"]),
    (["en","septembre","la","ville","de","rouen","capitula"],
     ["O","B-DATE","O","O","O","B-LOC","O"],
     ["ADP","NOUN","DET","NOUN","ADP","PROPN","VERB"],
     ["en","septembre","le","ville","de","rouen","capituler"]),
    (["le","roi","jean","second","régna","de","france"],
     ["O","B-TITLE","B-PER","I-PER","O","O","B-LOC"],
     ["DET","NOUN","PROPN","ADJ","VERB","ADP","PROPN"],
     ["le","roi","jean","second","régner","de","france"]),
    (["les","moines","de","saint","maur","copiaient","les","chartes"],
     ["O","O","O","B-ORG","I-ORG","O","O","O"],
     ["DET","NOUN","ADP","PROPN","PROPN","VERB","DET","NOUN"],
     ["le","moine","de","saint","maur","copier","le","charte"]),
    (["la","prévôté","des","marchands","siégeait","à","paris"],
     ["O","B-ORG","I-ORG","I-ORG","O","O","B-LOC"],
     ["DET","NOUN","ADP","NOUN","VERB","ADP","PROPN"],
     ["le","prévôté","de","marchand","siéger","à","paris"]),
    (["robert","de","clermont","seigneur","de","bourbon","mourut"],
     ["B-PER","I-PER","I-PER","B-TITLE","O","B-LOC","O"],
     ["PROPN","ADP","PROPN","NOUN","ADP","PROPN","VERB"],
     ["robert","de","clermont","seigneur","de","bourbon","mourir"]),
    (["en","l","an","de","grâce","le","roi","signa"],
     ["O","O","B-DATE","I-DATE","I-DATE","O","B-TITLE","O"],
     ["ADP","DET","NOUN","ADP","NOUN","DET","NOUN","VERB"],
     ["en","le","an","de","grâce","le","roi","signer"]),
    (["le","connétable","de","france","leva","l","armée"],
     ["O","B-TITLE","O","B-LOC","O","O","O"],
     ["DET","NOUN","ADP","PROPN","VERB","DET","NOUN"],
     ["le","connétable","de","france","lever","le","armée"]),
    (["à","la","saint","jean","les","vassaux","rendirent","hommage"],
     ["O","O","B-DATE","I-DATE","O","O","O","O"],
     ["ADP","DET","NOUN","PROPN","DET","NOUN","VERB","NOUN"],
     ["à","le","saint","jean","le","vassal","rendre","hommage"]),
    (["marie","d","artois","dame","de","conches","hérita"],
     ["B-PER","I-PER","I-PER","B-TITLE","O","B-LOC","O"],
     ["PROPN","ADP","PROPN","NOUN","ADP","PROPN","VERB"],
     ["marie","de","artois","dame","de","conches","hériter"]),
    (["le","tribunal","du","châtelet","de","paris","siégea"],
     ["O","B-ORG","I-ORG","I-ORG","I-ORG","I-ORG","O"],
     ["DET","NOUN","ADP","PROPN","ADP","PROPN","VERB"],
     ["le","tribunal","de","châtelet","de","paris","siéger"]),
    (["en","mars","le","sénéchal","gilles","fut","destitué"],
     ["O","B-DATE","O","B-TITLE","B-PER","O","O"],
     ["ADP","NOUN","DET","NOUN","PROPN","VERB","VERB"],
     ["en","mars","le","sénéchal","gilles","être","destituer"]),
    (["les","chevaliers","de","bourgogne","prirent","rouen"],
     ["O","O","O","B-LOC","O","B-LOC"],
     ["DET","NOUN","ADP","PROPN","VERB","PROPN"],
     ["le","chevalier","de","bourgogne","prendre","rouen"]),
    (["le","parlement","de","paris","condamna","l","accusé"],
     ["O","B-ORG","I-ORG","I-ORG","O","O","O"],
     ["DET","NOUN","ADP","PROPN","VERB","DET","NOUN"],
     ["le","parlement","de","paris","condamner","le","accusé"]),
]

# Vérification silencieuse
for i, (tok, ner, pos, lem) in enumerate(CORPUS_NER):
    assert len(tok) == len(ner) == len(pos) == len(lem), f"Incohérence phrase {i}"

_n_tokens = sum(len(t) for t,_,_,_ in CORPUS_NER)
print(f"Corpus chargé : {len(CORPUS_NER)} phrases · {_n_tokens} tokens")
print(f"Distribution NER : { {k: sum(l.count('B-'+k)+l.count('I-'+k) for _,l,_,_ in CORPUS_NER) for k in ['PER','LOC','DATE','ORG','TITLE']} }")

# %% [markdown]
# ## Étape 1 — Tokenisation par règles et évaluation TBA
#
# Avant toute annotation, les transcriptions HTR doivent être correctement
# segmentées. Les modèles HTR produisent trois types d'erreurs de frontière :
# fusions de mots (*"leseigneur"*), coupures spurieuses (*"nor mandie"*),
# et tirets résiduels de coupure de ligne (*"nor- mandie"*).
#
# **À vous de jouer.** Complétez `tokenize_medieval` et `token_boundary_accuracy`.

# %% [FOURNI]
MERGE_ERRORS = [
    ("leseigneur",    "le seigneur"),
    ("dunormand",     "du normand"),
    ("delafrance",    "de la france"),
    ("auroi",         "au roi"),
    ("auxchevaliers", "aux chevaliers"),
]

SPLIT_ERRORS = [
    ("nor- mandie",  "normandie"),
    ("sei- gneur",   "seigneur"),
    ("cha- telet",   "chatelet"),
]

# %% [TODO]
def tokenize_medieval(text: str) -> list[str]:
    """
    Tokenise une transcription médiévale par règles.

    Opérations dans l'ordre :
    1. Normalisation des espaces multiples :
       re.sub(r'\\s+', ' ', text.strip())
    2. Résolution des tirets résiduels de coupure de ligne :
       re.sub(r'-\\s+', '', text)
       (supprime le tiret et l'espace qui le suit)
    3. Séparation des fusions clitique+nom courantes :
       re.sub(r'\\bleseigneur\\b',    'le seigneur', text)
       re.sub(r'\\bdunormand\\b',     'du normand',  text)
       re.sub(r'\\bdelafrance\\b',    'de la france', text)
       re.sub(r'\\bauroi\\b',         'au roi',       text)
       re.sub(r'\\bauxchevaliers\\b', 'aux chevaliers', text)
    4. Retourner text.split()

    Paramètre
    ---------
    text : str  transcription (ligne HTR normalisée ou brute)

    Retourne
    --------
    list[str]  liste de tokens
    """
    # TODO
    pass   # ← remplacer


def token_boundary_accuracy(hyp: list[str], ref: list[str]) -> float:
    """
    Calcule la Token Boundary Accuracy (TBA).

    Définition : fraction des frontières de mots de la référence
    correctement identifiées dans l'hypothèse.

    Algorithme
    ----------
    Calculer les positions de frontières droites (somme cumulée des
    longueurs de tokens) pour hyp et ref, puis :
        TBA = |hyp_bounds ∩ ref_bounds| / |ref_bounds|

    Exemple :
        hyp = ["nor", "mandie"]    → positions {3, 10}
        ref = ["normandie"]        → positions {9}
        intersection = {}  →  TBA = 0.0

    Cas limite : retourner 1.0 si ref est vide.
    """
    # TODO
    pass   # ← remplacer

# %% [markdown]
# **Validation 1**

# %% [FOURNI — validation]
assert tokenize_medieval("leseigneur porta les lettres") == \
    ["le","seigneur","porta","les","lettres"], "fusion non résolue"
assert tokenize_medieval("nor- mandie") == ["normandie"], "tiret résiduel non géré"
assert tokenize_medieval("auroi  signa") == ["au","roi","signa"], "espaces multiples"

_tba_perfect = token_boundary_accuracy(["normandie"], ["normandie"])
_tba_split   = token_boundary_accuracy(["nor","mandie"], ["normandie"])
_tba_fused   = token_boundary_accuracy(["leseigneur"], ["le","seigneur"])
assert _tba_perfect == 1.0,  "TBA parfaite attendue"
assert _tba_split   == 0.0,  "TBA nulle pour coupure spurieuse sur 1 token"
assert 0.0 <= _tba_fused <= 1.0
print("Validation 1 : OK")

# %% [FOURNI]
# Évaluation de la tokenisation sur les erreurs synthétiques
print("\nSimulation d'erreurs de segmentation :")
print(f"{'Entrée HTR':30s}  {'Tokens produits':35s}  TBA")
for fused, ref_str in MERGE_ERRORS:
    hyp = tokenize_medieval(fused)
    ref = ref_str.split()
    tba = token_boundary_accuracy(hyp, ref)
    print(f"  {fused:28s}  {str(hyp):33s}  {tba:.2f}")

# %% [markdown]
# **Question 1** : La TBA est-elle une métrique suffisante pour évaluer la
# qualité de la segmentation dans le contexte NER ? Que se passe-t-il si
# une fusion de mots produit *"leseigneur"* alors que *"seigneur"* est une
# entité TITLE dans la référence ? Quelle métrique complémentaire utiliser ?

# %% [markdown]
# ## Étape 2 — Annotation NER par gazetier (baseline)
#
# Avant de fine-tuner un modèle, on établit une baseline par *weak supervision* :
# un gazetier (liste d'entités connues) annote automatiquement le corpus.
# Ce gazetier couvre les entités fréquentes ; les entités rares et les
# entités multi-tokens complexes restent non annotées ou mal annotées.
#
# **À vous de jouer.** Complétez `annotate_with_gazetteer`.

# %% [FOURNI]
GAZETTEER = {
    'PER':   {'jean','pierre','guillaume','thomas','robert','isabelle',
              'marguerite','charles','philippe','louis','alice','marie',
              'gilles','blanche','mathilde'},
    'LOC':   {'normandie','paris','france','bourgogne','champagne','brie',
              'gisors','rouen','chartres','louvre','vincenne','flandre',
              'bretagne','artois','conches','cluny','coucy','bourbon','clermont'},
    'DATE':  {'janvier','février','mars','avril','mai','juin','juillet',
              'août','septembre','octobre','novembre','décembre',
              'pâques','noël','pentecôte','carême'},
    'ORG':   {'parlement','chapitre','abbaye','prévôté','cour','tribunal'},
    'TITLE': {'seigneur','sénéchal','bailli','prévôt','châtelain','connétable',
              'monseigneur','messire','évêque','roi','duc','comte','comtesse',
              'chevalier','prudhomme','abbé','dame'},
}

PRIORITY = ['TITLE', 'PER', 'ORG', 'LOC', 'DATE']

# %% [TODO]
def annotate_with_gazetteer(tokens:    list[str],
                             gazetteer: dict = GAZETTEER,
                             scheme:    str  = "BIO") -> list[str]:
    """
    Annote une séquence de tokens par lookup dans le gazetier.

    Stratégie
    ---------
    Pour chaque token (en minuscules) :
    1. Parcourir PRIORITY dans l'ordre.
    2. Si token.lower() est dans gazetteer[ent_type], assigner 'B-{ent_type}'.
    3. Sinon, assigner 'O'.

    Note : cette version ne gère que les entités mono-token.
    Les entités multi-tokens (ex. "jean de normandie") ne sont PAS
    reconnues ici — c'est précisément ce que le modèle neural doit apprendre.

    Paramètres
    ----------
    tokens    : list[str]  tokens de la phrase
    gazetteer : dict       {type: set(formes)}
    scheme    : str        "BIO" uniquement dans ce TP

    Retourne
    --------
    list[str]  étiquettes BIO, une par token
    """
    labels = ['O'] * len(tokens)
    for i, token in enumerate(tokens):
        tok_low = token.lower()
        # TODO : boucle sur PRIORITY, assigner 'B-{ent_type}' si trouvé
        pass   # ← remplacer
    return labels

# %% [markdown]
# **Validation 2**

# %% [FOURNI — validation]
_toks = ["le","sénéchal","jean","de","normandie","porta","les","lettres"]
_labs = annotate_with_gazetteer(_toks)
assert _labs[1] == "B-TITLE", f"sénéchal doit être B-TITLE, obtenu {_labs[1]}"
assert _labs[2] == "B-PER",   f"jean doit être B-PER, obtenu {_labs[2]}"
assert _labs[4] == "B-LOC",   f"normandie doit être B-LOC, obtenu {_labs[4]}"
assert _labs[3] == "O",       f"'de' doit être O (non dans le gazetier)"
# Priorité TITLE > PER : "roi" est dans TITLE et pourrait être dans PER
_toks2 = ["le","roi","philippe"]
_labs2 = annotate_with_gazetteer(_toks2)
assert _labs2[1] == "B-TITLE", "roi doit être TITLE (priorité TITLE > PER)"
print("Validation 2 : OK")

# %% [FOURNI]
# Évaluation seqeval de la baseline gazetier
_y_true_all = [list(ner) for _, ner, _, _ in CORPUS_NER]
_y_pred_gaz = [annotate_with_gazetteer(t) for t, _, _, _ in CORPUS_NER]
print("\nBaseline gazetier (seqeval) :")
print(classification_report(_y_true_all, _y_pred_gaz, scheme=IOB2, zero_division=0))

# %% [markdown]
# **Question 2** : La baseline gazetier a un rappel faible sur les entités
# PER multi-tokens (*"jean de normandie"*). Comment le schéma BIO
# représente-t-il cette entité, et pourquoi le gazetier mono-token ne peut-il
# pas la reconnaître ? Quel serait le rappel théorique maximal d'un gazetier
# si l'on ajoutait toutes les entités multi-tokens du corpus ?

# %% [markdown]
# ## Étape 3 — Export CoNLL-2003
#
# CoNLL-2003 est le format tabulé standard pour les corpus NER.
# Chaque ligne contient : forme, POS, chunk tag (`_`), étiquette NER.
# Les phrases sont séparées par des lignes vides.
#
# **À vous de jouer.** Complétez `write_conll2003` et `read_conll2003`.

# %% [TODO]
def write_conll2003(sentences: list[list[tuple]], output_path: str) -> None:
    """
    Écrit un corpus au format CoNLL-2003.

    Paramètre sentences
    -------------------
    list de phrases, chaque phrase = list de (forme, pos, ner).

    Format d'une ligne :
        forme\\tPOS\\t_\\tNER\\n
    Séparateur de phrases : ligne vide.
    En-tête du document : '-DOCSTART- -X- O O\\n\\n'

    Indice
    ------
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    """
    # TODO
    pass   # ← remplacer


def read_conll2003(path: str) -> list[list[tuple]]:
    """
    Lit un corpus CoNLL-2003 et retourne les phrases.

    Retourne
    --------
    list[list[tuple(forme, pos, ner)]]

    Algorithme
    ----------
    - Ignorer les lignes '-DOCSTART-' et les lignes vides (elles délimitent les phrases).
    - Pour chaque ligne non vide : parser les 4 champs tab-séparés.
    - Retourner (champ 0, champ 1, champ 3) — ignorer champ 2 (chunk tag).
    """
    # TODO
    pass   # ← remplacer

# %% [markdown]
# **Validation 3**

# %% [FOURNI — validation]
_sentences_exp = [
    [(tok, pos, ner)
     for tok, ner, pos, _ in zip(t, n, p, l)]
    for t, n, p, l in CORPUS_NER
]
write_conll2003(_sentences_exp, "corpus_ner.conll")
_reloaded = read_conll2003("corpus_ner.conll")
assert len(_reloaded) == len(CORPUS_NER), \
    f"{len(_reloaded)} phrases rechargées, {len(CORPUS_NER)} attendues"
assert _reloaded[0][0] == ("li",  "DET",  "O"),       "Premier token incorrect"
assert _reloaded[0][1] == ("sénéchal", "NOUN", "B-TITLE"), "Deuxième token incorrect"
print("Validation 3 : OK")
print(f"Livrable : corpus_ner.conll ({len(_reloaded)} phrases)")

# %% [markdown]
# ## Étape 4 — Mesure de l'accord inter-annotateurs (IAA)
#
# Avant d'évaluer un modèle, il faut une référence fiable.
# Le kappa de Cohen mesure l'accord entre deux annotateurs au-delà du hasard.
#
# En l'absence de vrais annotateurs, nous simulons deux annotateurs artificiels
# avec des taux d'erreur paramétrables. Les erreurs suivent une distribution
# réaliste : confusions entre types proches (TITLE↔PER, ORG↔LOC),
# entités manquées (→O), et erreurs de frontière (B↔I).
#
# **À vous de jouer.** Complétez `cohen_kappa` et `simulate_annotator`.

# %% [FOURNI]
ALL_LABELS = ['O','B-PER','I-PER','B-LOC','I-LOC','B-DATE','I-DATE',
              'B-ORG','I-ORG','B-TITLE','I-TITLE']

# Confusions réalistes entre types proches
CONFUSION_PAIRS = {
    'B-TITLE': 'B-PER',  'B-PER':  'B-TITLE',
    'B-ORG':   'B-LOC',  'B-LOC':  'B-ORG',
    'I-PER':   'O',       'B-DATE': 'O',
    'I-ORG':   'B-ORG',
}

# %% [TODO]
def cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    """
    Calcule le kappa de Cohen entre deux séquences d'étiquettes.

    Formule :
        κ = (p_o - p_e) / (1 - p_e)

    où :
        p_o = proportion d'accord observé = somme(a==b) / n
        p_e = accord attendu par chance
            = Σ_k  (freq_a[k] / n) * (freq_b[k] / n)

    Paramètres
    ----------
    labels_a, labels_b : list[str]  séquences d'étiquettes de même longueur

    Retourne
    --------
    float  kappa (entre -1 et 1 ; > 0.61 = accord substantiel)

    Cas limite : retourner 0.0 si p_e >= 1.0.
    """
    assert len(labels_a) == len(labels_b), "Séquences de longueurs différentes"
    n = len(labels_a)
    # TODO : calculer p_o, p_e, retourner kappa
    pass   # ← remplacer


def simulate_annotator(labels:          list[str],
                        error_rate:      float,
                        rng:             random.Random,
                        confusion_pairs: dict = CONFUSION_PAIRS) -> list[str]:
    """
    Simule un annotateur humain imparfait.

    Pour chaque étiquette, avec probabilité error_rate, l'annotateur commet
    une erreur selon la distribution suivante :
    - 50 % : confusion de type (via confusion_pairs si disponible)
    - 30 % : étiquette remplacée par 'O' (entité manquée / fausse alarme)
    - 20 % : erreur de frontière :
             B-X → I-X, ou I-X → B-X (inversion Begin/Inside)

    Paramètres
    ----------
    labels          : list[str]   étiquettes de référence
    error_rate      : float       probabilité d'erreur par token (ex. 0.08)
    rng             : random.Random  générateur aléatoire (pour reproductibilité)
    confusion_pairs : dict         confusions entre types proches

    Retourne
    --------
    list[str]  étiquettes annotées (avec erreurs aléatoires)

    Indice
    ------
    Tirer un float entre 0 et 1 avec rng.random() pour décider si erreur.
    Pour le type d'erreur : rng.random() < 0.5 → confusion, < 0.8 → O, sinon frontière.
    """
    result = []
    for label in labels:
        if rng.random() < error_rate:
            r = rng.random()
            if r < 0.5 and label in confusion_pairs:
                # TODO : confusion de type
                pass
            elif r < 0.8:
                # TODO : entité manquée → O
                pass
            else:
                # TODO : erreur de frontière (B↔I)
                pass
        else:
            result.append(label)
    return result

# %% [TODO]
def compute_iaa(corpus:        list[tuple],
                error_rate_a:  float = 0.08,
                error_rate_b:  float = 0.12,
                seed:          int   = 42) -> dict:
    """
    Simule deux annotateurs sur le corpus et calcule le kappa inter-annotateurs.

    Algorithme
    ----------
    1. Créer deux RNG distincts : random.Random(seed) et random.Random(seed+1).
    2. Pour chaque phrase (tokens, ner, pos, lemmas) du corpus :
       - gold = ner (annotations de référence)
       - ann_a = simulate_annotator(gold, error_rate_a, rng_a)
       - ann_b = simulate_annotator(gold, error_rate_b, rng_b)
       - Étendre all_gold, all_a, all_b avec les étiquettes produites.
    3. Calculer :
       - kappa_inter  = cohen_kappa(all_a, all_b)
       - kappa_a_gold = cohen_kappa(all_a, all_gold)
       - kappa_b_gold = cohen_kappa(all_b, all_gold)
       - agreement_observed = somme(a==b) / n

    Retourne
    --------
    dict avec :
        "kappa_inter", "kappa_a_gold", "kappa_b_gold",
        "agreement_observed", "n_tokens",
        "annotator_a" (liste plate), "annotator_b" (liste plate), "gold"
    """
    rng_a = random.Random(seed)
    rng_b = random.Random(seed + 1)
    all_gold, all_a, all_b = [], [], []
    for tokens, ner, _, _ in corpus:
        # TODO
        pass   # ← remplacer
    # TODO : calculer et retourner le dict de métriques
    pass   # ← remplacer

# %% [markdown]
# **Validation 4**

# %% [FOURNI — validation]
_iaa = compute_iaa(CORPUS_NER, error_rate_a=0.08, error_rate_b=0.12)
assert _iaa is not None,              "compute_iaa doit retourner un dict"
assert 0.5 < _iaa["kappa_inter"] < 1.0, \
    f"kappa attendu entre 0.5 et 1.0, obtenu {_iaa['kappa_inter']}"
assert _iaa["n_tokens"] == sum(len(t) for t,_,_,_ in CORPUS_NER)
print("Validation 4 : OK")

# %% [FOURNI]
print(f"\nIAA (annotateur A : erreur 8%, annotateur B : erreur 12%) :")
print(f"  Kappa inter-annotateurs  : {_iaa['kappa_inter']:.3f}")
print(f"  Accord observé           : {_iaa['agreement_observed']:.3f}")
print(f"  Kappa A vs gold          : {_iaa['kappa_a_gold']:.3f}")
print(f"  Kappa B vs gold          : {_iaa['kappa_b_gold']:.3f}")
print(f"  Tokens évalués           : {_iaa['n_tokens']}")

_interpretation = ("accord presque parfait" if _iaa["kappa_inter"] > 0.80
                   else "accord substantiel" if _iaa["kappa_inter"] > 0.60
                   else "accord modéré")
print(f"  Interprétation           : {_interpretation}")

# Sauvegarde du rapport IAA
import json, os
os.makedirs("experiments", exist_ok=True)
_iaa_report = {k: v for k, v in _iaa.items()
               if k not in ("annotator_a","annotator_b","gold")}
_iaa_report["corpus_size"] = len(CORPUS_NER)
with open("iaa_report.json", "w", encoding="utf-8") as f:
    json.dump(_iaa_report, f, indent=2, ensure_ascii=False)
print("\nLivrable : iaa_report.json")

# %% [markdown]
# **Question 4** : Quel est l'impact de la distribution de la classe `O`
# (majoritaire dans tout corpus NER) sur le calcul de $p_e$ et donc du kappa ?
# Si 80 % des tokens sont `O`, et que les deux annotateurs étiquettent `O`
# de façon indépendante, quelle est la contribution de `O` à $p_e$ ?
# Proposez une variante qui serait plus discriminante sur les classes d'entités.

# %% [markdown]
# ## Étape 5 — Évaluation seqeval (F1 micro/macro par type)
#
# seqeval évalue au niveau *span* : une entité *"jean de normandie"* compte
# comme un seul exemple, et doit être reconnue dans son intégralité
# (bons tokens, bon type, bonnes frontières) pour être un vrai positif.
#
# **À vous de jouer.** Complétez `evaluate_ner`.

# %% [TODO]
def evaluate_ner(y_true: list[list[str]],
                 y_pred: list[list[str]]) -> dict:
    """
    Évalue un modèle NER avec seqeval au niveau span.

    Paramètres
    ----------
    y_true : list[list[str]]  séquences d'étiquettes de référence
    y_pred : list[list[str]]  séquences d'étiquettes prédites

    Retourne
    --------
    dict avec :
        "f1_micro"   : float  F1 micro (pondéré par la fréquence des entités)
        "f1_macro"   : float  F1 macro (même poids pour chaque type)
        "precision"  : float  précision micro
        "recall"     : float  rappel micro
        "per_entity" : dict   rapport complet par type d'entité

    Indice
    ------
    from seqeval.metrics import classification_report, f1_score, ...
    from seqeval.scheme  import IOB2
    Passer scheme=IOB2 et zero_division=0 à toutes les fonctions seqeval.
    Pour per_entity, utiliser output_dict=True dans classification_report
    et filtrer les clés 'micro avg', 'macro avg', 'weighted avg'.
    """
    # TODO
    pass   # ← remplacer

# %% [markdown]
# **Validation 5**

# %% [FOURNI — validation]
# Prédictions parfaites : F1 doit être 1.0
_perfect = evaluate_ner(_y_true_all, _y_true_all)
assert _perfect is not None,            "evaluate_ner doit retourner un dict"
assert _perfect["f1_micro"] == 1.0,     "F1-micro parfait = 1.0"
assert _perfect["f1_macro"] == 1.0,     "F1-macro parfait = 1.0"
# Prédictions nulles : F1 doit être 0.0
_null = [["O"] * len(s) for s in _y_true_all]
_zero = evaluate_ner(_y_true_all, _null)
assert _zero["f1_micro"] == 0.0,        "F1-micro nul = 0.0"
print("Validation 5 : OK")

# %% [FOURNI]
# Évaluation de la baseline gazetier avec seqeval
_metrics_gaz = evaluate_ner(_y_true_all, _y_pred_gaz)
print(f"\nBaseline gazetier :")
print(f"  F1-micro : {_metrics_gaz['f1_micro']:.4f}")
print(f"  F1-macro : {_metrics_gaz['f1_macro']:.4f}")
print()
print(classification_report(_y_true_all, _y_pred_gaz, scheme=IOB2, zero_division=0))

# %% [markdown]
# ## Étape 6 — Analyse des erreurs
#
# L'analyse d'erreurs identifie *pourquoi* le modèle se trompe,
# pas seulement *combien* de fois.
#
# **À vous de jouer.** Complétez `build_ner_spans` (conversion BIO→spans)
# et `build_error_report` (matrice de confusion au niveau span).

# %% [TODO]
def build_ner_spans(tokens:     list[str],
                    bio_labels: list[str]) -> list[dict]:
    """
    Convertit une séquence BIO en liste de spans avec offsets caractères.

    Algorithme
    ----------
    Parcourir les (token, label) en maintenant l'état d'une entité en cours :
    - 'B-X' : démarrer une nouvelle entité (clore la précédente si active).
    - 'I-X' : continuer l'entité en cours (ajouter le token).
    - 'O'   : clore l'entité en cours si active.
    Après la boucle, clore l'entité en cours si elle existe.

    Offsets caractères : la position de début du token i est
    sum(len(tok)+1 for tok in tokens[:i]) (chaque token est suivi d'un espace).
    L'offset de fin est start + len(' '.join(entity_tokens)).

    Retourne
    --------
    list[dict] avec pour chaque span :
        {"start": int, "end": int, "label": str, "text": str}
    où "text" = ' '.join(entity_tokens)
    et normalized[start:end] == text (à vérifier avec build_enriched_contract).
    """
    spans = []
    in_entity      = False
    entity_label   = None
    entity_start   = 0
    entity_tokens  = []
    char_pos       = 0

    for tok, lab in zip(tokens, bio_labels):
        if lab.startswith('B-'):
            if in_entity:
                # TODO : clore l'entité précédente et l'ajouter à spans
                pass
            # TODO : démarrer la nouvelle entité
            pass
        elif lab.startswith('I-') and in_entity:
            # TODO : continuer l'entité en cours
            pass
        else:
            if in_entity:
                # TODO : clore l'entité en cours
                pass
        char_pos += len(tok) + 1

    if in_entity:
        # TODO : clore la dernière entité
        pass

    return spans


def build_error_report(y_true:      list[list[str]],
                        y_pred:      list[list[str]],
                        tokens_list: list[list[str]]) -> dict:
    """
    Construit un rapport d'erreurs structuré depuis les prédictions NER.

    Algorithme
    ----------
    Pour chaque paire (phrase vraie, phrase prédite) :
    1. Extraire les spans vrais et prédits avec build_ner_spans.
    2. Construire des dicts {(start,end): label} pour true et pred.
    3. Pour chaque span vrai :
       - Trouver le label prédit au même offset (ou 'O' si absent).
       - Incrémenter confusion[true_label][pred_label].
       - Si erreur, ajouter le texte du span à error_examples["{vrai}->{prédit}"].
    4. Pour les spans prédits sans correspondance dans le vrai :
       - Incrémenter confusion['O'][pred_label] (faux positif).

    Retourne
    --------
    dict avec :
        "confusion_matrix" : pd.DataFrame  (types en lignes et colonnes)
        "error_examples"   : dict           {"{vrai}->{prédit}": [texte, ...]}
        (limiter à 3 exemples par type d'erreur)
    """
    from seqeval.metrics.sequence_labeling import get_entities
    confusion     = defaultdict(lambda: defaultdict(int))
    error_examples = defaultdict(list)

    for sent_true, sent_pred, sent_toks in zip(y_true, y_pred, tokens_list):
        true_spans_list = build_ner_spans(sent_toks, sent_true)
        pred_spans_list = build_ner_spans(sent_toks, sent_pred)
        true_spans = {(s['start'], s['end']): s['label'] for s in true_spans_list}
        pred_spans = {(s['start'], s['end']): s['label'] for s in pred_spans_list}

        for (s, e), true_lab in true_spans.items():
            pred_lab = pred_spans.get((s, e), 'O')
            # TODO : incrémenter confusion et stocker les exemples d'erreur
            pass

        for (s, e) in set(pred_spans) - set(true_spans):
            # TODO : faux positifs → confusion['O'][pred_lab]
            pass

    all_types = sorted(set(k for d in confusion.values() for k in d) |
                       set(confusion.keys()))
    conf_df   = pd.DataFrame(
        [[confusion[tt][tp] for tp in all_types] for tt in all_types],
        index=all_types, columns=all_types
    )
    return {
        "confusion_matrix": conf_df,
        "error_examples":   {k: v[:3] for k, v in error_examples.items()},
    }

# %% [markdown]
# **Validation 6**

# %% [FOURNI — validation]
_toks_v  = ["le","sénéchal","jean","de","normandie","porta"]
_bio_v   = ["O","B-TITLE","B-PER","I-PER","I-PER","O"]
_spans_v = build_ner_spans(_toks_v, _bio_v)
assert _spans_v is not None and len(_spans_v) == 2, \
    f"2 spans attendus, obtenu {_spans_v}"
assert _spans_v[0]["text"]  == "sénéchal",         "Premier span = sénéchal"
assert _spans_v[0]["label"] == "TITLE",             "Label = TITLE"
assert _spans_v[1]["text"]  == "jean de normandie", "Deuxième span = jean de normandie"
assert _spans_v[1]["label"] == "PER",               "Label = PER"

# Vérification des offsets caractères
_normalized_v = " ".join(_toks_v)
for _span in _spans_v:
    _extracted = _normalized_v[_span["start"]:_span["end"]]
    assert _extracted == _span["text"], \
        f"Offset incorrect : [{_span['start']}:{_span['end']}] = {_extracted!r}"

_report = build_error_report(_y_true_all, _y_pred_gaz,
                              [t for t,_,_,_ in CORPUS_NER])
assert "confusion_matrix" in _report
assert "error_examples"   in _report
print("Validation 6 : OK")

# %% [FOURNI]
print("\nMatrice de confusion (gazetier baseline) :")
print(_report["confusion_matrix"].to_string())
print("\nExemples d'erreurs :")
for err_type, examples in list(_report["error_examples"].items())[:5]:
    print(f"  {err_type:25s} : {examples}")

# Sauvegarder le rapport d'erreurs
_error_report_serializable = {
    "confusion_matrix": _report["confusion_matrix"].to_dict(),
    "error_examples":   _report["error_examples"],
    "f1_micro_baseline": _metrics_gaz["f1_micro"],
    "f1_macro_baseline": _metrics_gaz["f1_macro"],
}
with open("error_report.json", "w", encoding="utf-8") as f:
    json.dump(_error_report_serializable, f, indent=2, ensure_ascii=False)
print("\nLivrable : error_report.json")

# %% [markdown]
# **Question 6** : La matrice de confusion NER a une propriété inhabituelle :
# la ligne `O` représente les *faux positifs* (entités prédites qui ne sont
# pas dans la référence), et la colonne `O` représente les *faux négatifs*
# (entités de référence non détectées). Identifiez dans votre matrice les
# deux confusions inter-types les plus fréquentes et expliquez leur cause
# linguistique dans le contexte médiéval.

# %% [markdown]
# ## Étape 7 — Data contract enrichi et export CoNLL-U
#
# Le data contract enrichi relie chaque annotation (NER, POS, lemme) à sa
# ligne de manuscrit via le `polygon_ref`. C'est l'artefact de sortie
# central du Volet 2 — il est l'entrée du Chapitre 7.
#
# **À vous de jouer.** Complétez `build_enriched_contract` et `export_conllu`.

# %% [TODO]
def build_enriched_contract(line_id:       str,
                             transcription: str,
                             confidence:    float,
                             normalized:    str,
                             ner_spans:     list[dict],
                             pos_tags:      list[str],
                             lemmas:        list[str],
                             polygon_ref:   str) -> dict:
    """
    Construit et valide un enregistrement de data contract enrichi.

    Invariants à vérifier (lever AssertionError si violation) :
    1. len(pos_tags) == len(normalized.split())
    2. len(lemmas)   == len(normalized.split())
    3. Pour chaque span : normalized[span['start']:span['end']] == span['text']

    Retourne
    --------
    dict avec les clés :
        "line_id", "transcription", "confidence", "normalized",
        "ner_spans", "pos_tags", "lemmas", "polygon_ref"
    """
    tokens = normalized.split()
    # TODO : vérifier les invariants et retourner le dict
    pass   # ← remplacer


def export_conllu(sentences:     list[list[dict]],
                  ner_spans_list: list[list[dict]],
                  output_path:   str) -> None:
    """
    Exporte un corpus annoté au format CoNLL-U.

    Format d'une ligne (10 champs tabulation-séparés) :
    ID  FORM  LEMMA  UPOS  XPOS  FEATS  HEAD  DEPREL  DEPS  MISC

    Pour ce TP :
    - XPOS, FEATS, DEPS = '_'
    - HEAD = 0, DEPREL = '_'  (arcs de dépendance non disponibles)
    - MISC = '_'  (sauf si polygon_ref disponible dans le token dict)

    En-tête de phrase (lignes commençant par #) :
        # sent_id = {sent_id}
        # text = {tokens reconstitués}
        # ner_spans = {texte:label; ...}  (si spans non vides)

    Paramètres
    ----------
    sentences      : list de phrases, chaque phrase = list de dicts token
                     avec clés "id", "form", "lemma", "upos"
    ner_spans_list : list de listes de spans NER pour chaque phrase
    output_path    : chemin du fichier de sortie
    """
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for sent_id, (sentence, spans) in enumerate(zip(sentences, ner_spans_list)):
            # TODO : écrire l'en-tête de phrase
            # TODO : écrire chaque token (10 champs)
            # TODO : écrire la ligne vide de séparation
            pass   # ← remplacer

# %% [markdown]
# **Validation 7**

# %% [FOURNI — validation]
_toks0, _ner0, _pos0, _lem0 = CORPUS_NER[0]
_norm0   = " ".join(_toks0)
_spans0  = build_ner_spans(_toks0, _ner0)
_contract = build_enriched_contract(
    line_id       = "charte_test_001",
    transcription = _norm0,
    confidence    = 0.91,
    normalized    = _norm0,
    ner_spans     = _spans0,
    pos_tags      = _pos0,
    lemmas        = _lem0,
    polygon_ref   = "fol01_bbox_100_200_800_230",
)
assert _contract is not None,                 "build_enriched_contract ne doit pas retourner None"
assert _contract["line_id"] == "charte_test_001"
assert len(_contract["pos_tags"]) == len(_toks0)
assert len(_contract["lemmas"])   == len(_toks0)
# Test d'invariant : mauvaise longueur pos_tags doit lever AssertionError
try:
    build_enriched_contract("x","y",1.0,"a b",[][:],["NOUN"],["le"],"_")
    assert False, "AssertionError attendue"
except AssertionError:
    pass
print("Validation 7 : OK")

# %% [FOURNI]
# Construire et exporter le corpus enrichi complet
_all_contracts = []
for i, (toks, ner, pos, lem) in enumerate(CORPUS_NER):
    _norm = " ".join(toks)
    _spans = build_ner_spans(toks, ner)
    _c = build_enriched_contract(
        line_id       = f"charte_medieval_{i:03d}",
        transcription = _norm,
        confidence    = round(0.85 + 0.15 * random.random(), 3),
        normalized    = _norm,
        ner_spans     = _spans,
        pos_tags      = pos,
        lemmas        = lem,
        polygon_ref   = f"fol{i//5+1:02d}_bbox_100_{200+i*30}_{800}_{230+i*30}",
    )
    _all_contracts.append(_c)

with open("enriched_corpus.jsonl", "w", encoding="utf-8") as f:
    for c in _all_contracts:
        f.write(json.dumps(c, ensure_ascii=False) + '\n')
print(f"Livrable : enriched_corpus.jsonl ({len(_all_contracts)} enregistrements)")

# Export CoNLL-U
_sentences_ud = [
    [{"id": j+1, "form": tok, "lemma": lem, "upos": pos}
     for j, (tok, pos, lem) in enumerate(zip(t, p, l))]
    for t, _, p, l in CORPUS_NER
]
_ner_spans_all = [build_ner_spans(t, n) for t, n, _, _ in CORPUS_NER]
export_conllu(_sentences_ud, _ner_spans_all, "corpus_ner.conllu")
print("Livrable : corpus_ner.conllu")

# %% [markdown]
# ## Étape 8 — Bilan mémoire CamemBERT-LoRA et alignement subwords
#
# Avant d'entraîner, calculez l'empreinte mémoire et vérifiez que vous
# comprenez comment les étiquettes NER s'alignent sur les subwords BPE.
#
# **À vous de jouer.** Complétez `compute_camembert_lora_params`
# et `align_labels_to_subwords`.

# %% [TODO]
def compute_camembert_lora_params(d_model:        int,
                                   d_kv:           int,
                                   n_layers:       int,
                                   r:              int,
                                   target_modules: list[str],
                                   n_labels:       int = 11) -> dict:
    """
    Calcule le bilan mémoire de CamemBERT-NER avec LoRA.

    CamemBERT-base : 110M paramètres totaux, d_model=768, d_kv=64, 12 couches.

    Paramètres LoRA (Q et V uniquement) :
        lora_trainable = n_layers * len(target_modules) * r * (d_model + d_kv)

    Tête de classification NER (entraînable en full) :
        head_params = d_model * n_labels + n_labels

    Mémoire (en Mo) :
        mem_model_fp16  : modèle gelé float16 = 110M * 2 / 1e6
        mem_lora_adam   : Adam sur LoRA float32 = lora_trainable * 16 / 1e6
        mem_activations : ~500 Mo pour batch=16, seq=128

    Retourne
    --------
    dict avec "lora_trainable", "head_params", "pct_of_total",
              "mem_model_fp16", "mem_lora_adam", "mem_activations_mb"
    """
    # TODO
    pass   # ← remplacer


def align_labels_to_subwords(word_ids: list,
                              labels:   list[int]) -> list[int]:
    """
    Aligne les étiquettes NER (niveau mot) sur les subwords BPE.
    Stratégie first-token : seul le premier subword d'un mot reçoit
    l'étiquette, les suivants reçoivent -100 (ignoré par la loss).

    Paramètres
    ----------
    word_ids : list  identifiants de mot pour chaque subword
               (None pour [CLS] et [SEP], int pour les vrais tokens)
    labels   : list[int]  étiquettes numériques au niveau mot

    Retourne
    --------
    list[int]  étiquettes alignées sur les subwords (-100 pour les ignorés)

    Algorithme
    ----------
    Pour chaque word_id dans word_ids :
    - None → -100  (token spécial)
    - word_id != prev_word_id → labels[word_id]  (premier subword)
    - word_id == prev_word_id → -100  (subword suivant)
    """
    aligned, prev = [], None
    for wid in word_ids:
        # TODO
        pass   # ← remplacer
        prev = wid
    return aligned

# %% [markdown]
# **Validation 8**

# %% [FOURNI — validation]
_cfg = compute_camembert_lora_params(768, 64, 12, 8, ["query","value"])
assert _cfg is not None,                          "compute_camembert_lora_params ne doit pas retourner None"
assert _cfg["lora_trainable"] == 159_744,         f"Attendu 159744, obtenu {_cfg['lora_trainable']}"
assert _cfg["pct_of_total"]   < 1.0,              "LoRA < 1% du total CamemBERT"
assert _cfg["mem_model_fp16"] == 220.0,           "Modèle fp16 = 220 Mo"

# Test alignement subwords : [CLS, 0, 1, 1, 1, 2, 3, SEP]
# mot 0 → étiq 0 (O), mot 1 → étiq 9 (B-TITLE), mot 2 → étiq 1 (B-PER), mot 3 → étiq 0 (O)
_wids    = [None, 0, 1, 1, 1, 2, 3, None]
_lbls    = [0, 9, 1, 0]
_aligned = align_labels_to_subwords(_wids, _lbls)
assert _aligned == [-100, 0, 9, -100, -100, 1, 0, -100], \
    f"Alignement incorrect : {_aligned}"
print("Validation 8 : OK")

# %% [FOURNI]
print("\nBilan mémoire — CamemBERT-base NER + LoRA r=8 (Q+V, 12 couches)")
print(f"  Paramètres LoRA entraînables : {_cfg['lora_trainable']:>10,}  ({_cfg['pct_of_total']:.4f}% du total)")
print(f"  Tête de classification NER   : {_cfg['head_params']:>10,}")
print(f"  Modèle gelé (float16)        : {_cfg['mem_model_fp16']:>8.0f} Mo")
print(f"  Adam LoRA + tête (float32)   : {_cfg['mem_lora_adam']:>8.2f} Mo")
print(f"  Activations (batch=16)       : {_cfg['mem_activations_mb']:>8} Mo (estimé)")
_total = _cfg['mem_model_fp16'] + _cfg['mem_lora_adam'] + _cfg['mem_activations_mb']
print(f"  Total GPU estimé             : {_total:>8.0f} Mo ≈ {_total/1024:.1f} Go")

# %% [markdown]
# ## Étape 9 — Fine-tuning CamemBERT-NER avec LoRA
#
# Cette étape entraîne CamemBERT pour la classification de tokens (NER)
# avec des adaptateurs LoRA. Elle suit exactement les mêmes conventions
# que l'Étape 9 du TP Chapitre 3+4, avec `TaskType.TOKEN_CLS` à la place
# de `SEQ_2_SEQ_LM` et `target_modules=["query","value"]` à la place de
# `["q","v"]` (nommage RoBERTa vs T5).

# %% [FOURNI]
import torch
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
USE_FP16   = DEVICE == "cuda"
BATCH_SIZE = 16 if DEVICE == "cuda" else 4
N_EPOCHS   = 20 if DEVICE == "cuda" else 5

print(f"Dispositif : {DEVICE.upper()}")
print(f"  fp16={USE_FP16}, batch={BATCH_SIZE}, epochs={N_EPOCHS}")
if DEVICE == "cpu":
    print("  Mode CPU : entraînement indicatif (~25 min pour 5 epochs).")
    print("  Lancer en parallèle des étapes 10 et 11 pendant l'attente.")

LABEL2ID = {"O":0,"B-PER":1,"I-PER":2,"B-LOC":3,"I-LOC":4,
            "B-DATE":5,"I-DATE":6,"B-ORG":7,"I-ORG":8,"B-TITLE":9,"I-TITLE":10}
ID2LABEL  = {v: k for k, v in LABEL2ID.items()}

# %% [TODO]
def prepare_ner_dataset(corpus:     list[tuple],
                         tokenizer,
                         label2id:   dict = LABEL2ID,
                         max_length: int  = 128) -> list[dict]:
    """
    Tokenise le corpus NER pour CamemBERT et aligne les étiquettes.

    Pour chaque phrase (tokens, ner_labels, pos_tags, lemmas) :
    1. Tokeniser avec is_split_into_words=True, return_tensors=None,
       truncation=True, max_length=max_length.
    2. Récupérer word_ids = encoding.word_ids().
    3. Convertir ner_labels en indices numériques via label2id.
    4. Aligner avec align_labels_to_subwords(word_ids, label_ids).
    5. Retourner un dict avec "input_ids", "attention_mask", "labels".

    Note : return_tensors=None retourne des listes Python,
    le DataCollatorForTokenClassification s'occupe du padding.
    """
    dataset = []
    for tokens, ner_labels, _, _ in corpus:
        # TODO
        pass   # ← remplacer
    return dataset


def build_ner_lora_config(r:    int   = 8,
                           alpha: int   = 16,
                           dropout: float = 0.1) -> "LoraConfig":
    """
    Construit la configuration LoRA pour CamemBERT-NER.

    Différences avec le TP Chapitre 3+4 :
    - task_type = TaskType.TOKEN_CLS  (classification de tokens)
    - target_modules = ["query", "value"]  (nommage RoBERTa)
    - modules_to_save = ["classifier"]  (sauvegarder la tête NER)

    Retourne
    --------
    LoraConfig (peft)
    """
    from peft import LoraConfig, TaskType
    # TODO
    pass   # ← remplacer

# %% [FOURNI]
def train_camembert_ner(corpus:      list[tuple],
                         model_name: str   = "almanach/camembert-base",
                         r:          int   = 8,
                         n_epochs:   int   = N_EPOCHS,
                         batch_size: int   = BATCH_SIZE,
                         output_dir: str   = "./camembert_ner_lora") -> dict:
    """Entraîne CamemBERT-NER avec LoRA. Retourne les métriques de validation."""
    from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                               TrainingArguments, DataCollatorForTokenClassification,
                               EarlyStoppingCallback)
    from peft import get_peft_model
    from datasets import Dataset
    import numpy as np

    lora_config = build_ner_lora_config(r=r)
    if lora_config is None:
        print("build_ner_lora_config non implémenté — skip entraînement.")
        return {}

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=len(LABEL2ID), id2label=ID2LABEL, label2id=LABEL2ID)
    model     = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Split 80/10/10
    rng = random.Random(42)
    idx = list(range(len(corpus))); rng.shuffle(idx)
    n_train = int(len(idx)*0.8); n_val = int(len(idx)*0.1)
    train_corp = [corpus[i] for i in idx[:n_train]]
    val_corp   = [corpus[i] for i in idx[n_train:n_train+n_val]]
    test_corp  = [corpus[i] for i in idx[n_train+n_val:]]

    samples    = prepare_ner_dataset(corpus, tokenizer)
    if not samples: print("prepare_ner_dataset non implémenté."); return {}

    train_data = Dataset.from_list(prepare_ner_dataset(train_corp, tokenizer))
    val_data   = Dataset.from_list(prepare_ner_dataset(val_corp,   tokenizer))
    collator   = DataCollatorForTokenClassification(tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        y_true = [[ID2LABEL[l] for l in row if l != -100] for row in labels]
        y_pred = [[ID2LABEL[p] for p, l in zip(pr, la) if l != -100]
                  for pr, la in zip(preds, labels)]
        return {"f1": f1_score(y_true, y_pred, average="micro", scheme=IOB2,
                               zero_division=0)}

    args = TrainingArguments(
        output_dir=output_dir, num_train_epochs=n_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=2e-4, lr_scheduler_type="cosine", warmup_ratio=0.1,
        fp16=USE_FP16, eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="f1",
        greater_is_better=True, logging_steps=5, report_to="none",
    )
    from transformers import Trainer
    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_data, eval_dataset=val_data,
        data_collator=collator, compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )
    trainer.train()
    metrics = trainer.evaluate()
    model.save_pretrained(output_dir)
    # Évaluation sur le test
    test_data = Dataset.from_list(prepare_ner_dataset(test_corp, tokenizer))
    test_metrics = trainer.predict(test_data)
    return {"eval_f1": metrics.get("eval_f1", 0),
            "test_metrics": test_metrics.metrics}

# %% [FOURNI]
print("\nLancement de l'entraînement CamemBERT-NER + LoRA r=8...")
try:
    ner_metrics = train_camembert_ner(CORPUS_NER, r=8, output_dir="./ner_lora_r8")
    print(f"Entraînement terminé. eval_f1 = {ner_metrics.get('eval_f1', 'N/A'):.4f}")
except Exception as e:
    print(f"Entraînement non disponible ({e}).")
    ner_metrics = {}

# %% [markdown]
# ## Étape 10 — Tableau d'ablation NER
#
# **À vous de jouer.** Complétez `build_ner_ablation_table`.

# %% [TODO]
def build_ner_ablation_table(results: dict) -> pd.DataFrame:
    """
    Construit le tableau d'ablation comparatif des configurations NER.

    Paramètre results
    -----------------
    dict {config_name: {"f1_micro", "f1_macro", "params", "note"}}

    La ligne "Gazetier (baseline)" doit être calculée en appelant
    evaluate_ner sur les prédictions du gazetier (déjà dans _metrics_gaz).

    Retourne
    --------
    pd.DataFrame avec colonnes :
        Configuration | F1-micro ↑ | F1-macro ↑ | Params entr. | Note

    La table doit contenir au minimum :
    - "Gazetier (baseline)"
    - "CamemBERT+LoRA r=8" (depuis results["lora_r8"])
    """
    rows = []
    # TODO : construire au moins 2 lignes
    pass   # ← remplacer
    return pd.DataFrame(rows)

# %% [FOURNI]
_ablation_results = {
    "lora_r8": {
        "f1_micro": ner_metrics.get("eval_f1", "—"),
        "f1_macro": "—",
        "params":   159_744,
        "note":     f"{N_EPOCHS} epochs · LoRA r=8 Q+V",
    },
}
_ablation = build_ner_ablation_table(_ablation_results)
if _ablation is not None and len(_ablation) > 0:
    print("\nTableau d'ablation NER :")
    print(_ablation.to_string(index=False))
else:
    print("Tableau non disponible (TODO non complété).")

# %% [markdown]
# **Question 10** : Analysez votre tableau d'ablation.
#
# Premièrement : le gain du modèle LoRA par rapport au gazetier est-il
# principalement dû à la meilleure reconnaissance des entités mono-tokens
# (déjà couverts par le gazetier) ou des entités multi-tokens
# (*"jean de normandie"*, *"cour du parlement de paris"*) ?
#
# Deuxièmement : pour quelle(s) catégorie(s) le modèle améliore-t-il
# le plus le F1 par rapport à la baseline ? Cela correspond-il à ce
# que vous attendiez d'après l'analyse d'erreurs de l'Étape 6 ?

# %% [markdown]
# ---
#
# ## Récapitulatif — Ce que vous avez implémenté
#
# | Étape | Fonction(s) | Lien avec le cours |
# |---|---|---|
# | 1 Tokenisation | `tokenize_medieval`, `token_boundary_accuracy` | Chapitre 5 §1 |
# | 2 Gazetier | `annotate_with_gazetteer` | Chapitre 6 §1.2 |
# | 3 CoNLL-2003 | `write_conll2003`, `read_conll2003` | Chapitre 6 §1.3 |
# | 4 IAA | `cohen_kappa`, `simulate_annotator`, `compute_iaa` | Chapitre 5 §5 |
# | 5 seqeval | `evaluate_ner` | Chapitre 6 §4 |
# | 6 Erreurs | `build_ner_spans`, `build_error_report` | Chapitre 6 §5 |
# | 7 Data contract | `build_enriched_contract`, `export_conllu` | Chapitre 6 §6 |
# | 8 Mémoire + subwords | `compute_camembert_lora_params`, `align_labels_to_subwords` | Chapitre 5 §4 · Chapitre 6 §2 |
# | 9 LoRA NER | `prepare_ner_dataset`, `build_ner_lora_config` | Chapitre 6 §2 |
# | 10 Ablation | `build_ner_ablation_table` | Chapitre 6 §4.1 |
#
# **Livrables produits :**
# `corpus_ner.conll`, `corpus_ner.conllu`, `enriched_corpus.jsonl`,
# `iaa_report.json`, `error_report.json`, checkpoint `ner_lora_r8/` (si GPU).
