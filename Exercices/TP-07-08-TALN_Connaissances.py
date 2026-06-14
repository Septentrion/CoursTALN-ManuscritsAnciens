"""
TP Autonome — Base de connaissances médiévale interrogeable
           BERTopic · Relations · Coréférence · Graphe · TEI · Data contract v2
Chapitres 7 & 8 — Module NLP · Master Data/IA · MD5 Volet 2 · 2026
─────────────────────────────────────────────────────────────────────
Ce TP est autonome : vous n'avez pas de squelette de fonctions à compléter.
Vous écrivez votre code depuis les consignes, en vous appuyant sur les
Chapitres 7 et 8 comme références.

Pipeline à construire :

  enriched_corpus.jsonl  (Jour 3)
         │
  [Étape 1]  Agrégation des lignes en actes + BERTopic
             → topic_report.json
         │
  [Étape 2]  Extraction de relations (PER – LOC – DATE)
             + validation manuelle sur 20 triplets → precision_report.json
         │
  [Étape 3]  Résolution de coréférence (règles)
             → coref_chains.json
         │
  [Étape 4]  Construction du graphe (NetworkX) + export JSON-LD
             → knowledge_graph.jsonld
         │
  [Étape 5]  Export TEI-XML (un fichier par acte, xmllint validé)
             → tei_output/*.xml
         │
  [Étape 6]  Data contract NLP v2 (schéma complet avec topics + relations
             + coref + graph_node_id + tei_ref)
             → enriched_corpus_v2.jsonl
         │
  [Étape 7]  Protocole de rétroaction HTR (delta de corrections)
             → htr_feedback.json
         │
  [Étape 8]  Interface de requête minimale (plein-texte + filtres)

Durée estimée : 3 h 30
Livrables attendus :
  - topic_report.json          (BERTopic : topics, mots représentatifs, pages)
  - tei_output/*.xml           (un .xml par acte, validé xmllint)
  - enriched_corpus_v2.jsonl   (data contract NLP v2 complet)
  - htr_feedback.json          (delta de corrections pour réentraîner le HTR)
  - knowledge_graph.jsonld     (graphe de connaissances JSON-LD)
  - Interface de requête        (CLI ou cellule interactive)

Instructions générales
──────────────────────
Ce TP est autonome : vous écrivez vous-mêmes toutes les fonctions.
Les cellules marquées  # FOURNI  fournissent le corpus et le boilerplate
de départ (imports, corpus, constantes) ; elles sont à exécuter telles quelles.
Tout le reste est à écrire.

Les Chapitres 7 et 8 contiennent le code de référence complet pour chaque
étape ; consultez-les si vous êtes bloqués.
"""

# %% [markdown]
# # TP Autonome — Base de connaissances médiévale interrogeable
#
# Vous allez transformer le corpus annoté du Jour 3 en une base de
# connaissances interrogeable : topics thématiques, relations entre entités,
# graphe de connaissances, export TEI-XML, et protocole de rétroaction
# pour améliorer le modèle HTR.
#
# **Fil conducteur :** chaque artefact produit doit être ancré au document
# source via le `polygon_ref`. Un triplet ou une entité sans traçabilité
# vers le parchemin original n'est pas un résultat scientifique.

# %% [markdown]
# ## Partie 0 — Corpus de départ

# %% [FOURNI]
import re, json, os, math, random, hashlib, datetime, editdistance
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from lxml import etree

random.seed(42)

# ── Corpus enrichi — 40 lignes médiévales sur 8 actes ────────────────────
# En TP réel : charger depuis enriched_corpus.jsonl produit au Jour 3.
# Ici : corpus synthétique reprenant exactement les 40 phrases du TP NER.

_CORPUS_RAW = [
    # (tokens, ner_labels, pos_tags, lemmas)
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

# Variantes HTR simulées (remplacent la transcription brute)
_HTR_VARIANTS = {
    'sénéchal':'s^r', 'normandie':'norm~die', 'champagne':'champ~e',
    'france':'fra~ce', 'seigneur':'s^r', 'évêque':'evesque',
    'gisors':'gisorz', 'juillet':'juill~t', 'parlement':'parl~ent',
    'connétable':'conne table', 'châtelet':'chastelet',
}

def _inject_htr(text, rate=0.40, seed=None):
    rng = random.Random(seed or 0)
    return ' '.join(
        _HTR_VARIANTS.get(w.lower(), w) if rng.random() < rate else w
        for w in text.split()
    )

def _build_spans(tokens, bio_labels):
    spans=[]; in_e=False; el=None; es=0; et=[]
    cp=0
    for tok, lab in zip(tokens, bio_labels):
        if lab.startswith('B-'):
            if in_e:
                txt=' '.join(et)
                spans.append({'start':es,'end':es+len(txt),'label':el,'text':txt})
            el=lab[2:]; es=cp; et=[tok]; in_e=True
        elif lab.startswith('I-') and in_e:
            et.append(tok)
        else:
            if in_e:
                txt=' '.join(et)
                spans.append({'start':es,'end':es+len(txt),'label':el,'text':txt})
                in_e=False
        cp+=len(tok)+1
    if in_e:
        txt=' '.join(et)
        spans.append({'start':es,'end':es+len(txt),'label':el,'text':txt})
    return spans

# Construire les 40 enregistrements enrichis
RECORDS = []
for i, (toks, ner, pos, lem) in enumerate(_CORPUS_RAW):
    acte_id = f"acte_{i//5+1:03d}"
    norm = " ".join(toks)
    htr  = _inject_htr(norm, rate=0.4, seed=i)
    RECORDS.append({
        "line_id":       f"{acte_id}_l{i%5+1:02d}",
        "polygon_ref":   f"fol{i//5+1:02d}_bbox_{100+i*12}_{200}_{800}_{230}",
        "transcription": htr,
        "confidence":    round(0.78 + 0.20 * random.random(), 3),
        "normalized":    norm,
        "ner_spans":     _build_spans(toks, ner),
        "pos_tags":      list(pos),
        "lemmas":        list(lem),
    })

ACTES = sorted({r["line_id"].rsplit("_l", 1)[0] for r in RECORDS})
print(f"Corpus chargé : {len(RECORDS)} lignes · {len(ACTES)} actes")
print(f"  Entités NER  : "
      f"{ {k: sum(any(s['label']==k for s in r['ner_spans']) for r in RECORDS) for k in ['PER','LOC','DATE','ORG','TITLE']} }")

# %% [markdown]
# ---
# ## Étape 1 — Modélisation thématique avec BERTopic
#
# ### 1.1 Agrégation des lignes en actes
#
# BERTopic requiert des documents d'au moins quelques dizaines de tokens.
# Agrégez les lignes par acte (5 lignes par acte → ~40 tokens par document).
#
# **Consigne :**
#
# Écrivez une fonction `aggregate_to_documents(records)` qui regroupe
# les lignes par acte en concaténant les `normalized` et les `lemmas`.
# Chaque document doit contenir les clés :
# `doc_id`, `text`, `lemma_text`, `ner_spans`, `line_ids`.
#
# Les 40 lignes forment 8 actes de 5 lignes — le `doc_id` est le préfixe
# du `line_id` avant le dernier `_lXX`.

# %% [Votre code ici]
# def aggregate_to_documents(records: list[dict]) -> list[dict]:
#     ...

# %% [markdown]
# ### 1.2 Configuration et entraînement BERTopic
#
# **Note d'environnement :** BERTopic requiert `bertopic`, `sentence-transformers`,
# `umap-learn`, et `hdbscan`. Si ces packages ne sont pas disponibles dans votre
# environnement, utilisez la simulation fournie ci-dessous (Étape 1.3).
#
# **Consigne :**
#
# Configurez BERTopic pour un corpus de 8 documents :
# - `embedding_model` : `"camembert-base"` (SentenceTransformer)
# - UMAP : `n_neighbors=3`, `n_components=3`, `min_dist=0.0`
# - HDBSCAN : `min_cluster_size=2`, `min_samples=1`
# - `nr_topics="auto"`
#
# Entraînez le modèle sur les `lemma_text` des 8 documents agrégés.
# Stockez le topic assigné à chaque document dans `doc["topic_id"]`
# et un label humain lisible dans `doc["topic_label"]`.
#
# **Topics attendus :** juridique (actes, signatures, lettres),
# nobiliaire (seigneurs, chevaliers, rois), liturgique (abbayes, moines).

# %% [markdown]
# ### 1.3 Simulation BERTopic (fallback sans GPU / sans installation)
#
# Si BERTopic n'est pas disponible, utilisez cette simulation
# basée sur des mots-clés thématiques :

# %% [FOURNI — simulation BERTopic]
TOPIC_KEYWORDS = {
    0: {"label": "juridique",
        "words": {"acte","signer","charte","lettre","jugement","procès",
                  "missive","tribunal","condamner","rendre"}},
    1: {"label": "nobiliaire",
        "words": {"seigneur","roi","duc","comte","chevalier","hommage",
                  "vassal","connétable","bailli","prévôt","châtelain",
                  "dame","marguerite","isabelle","marguerite"}},
    2: {"label": "liturgique",
        "words": {"abbaye","moine","chapitre","pâques","noël","carême",
                  "pentecôte","saint","prière","donation","cluny"}},
}

def simulate_bertopic(documents: list[dict]) -> list[dict]:
    """
    Simulation BERTopic par vote de mots-clés.
    Utilisée quand le package bertopic n'est pas installé.
    Assigne à chaque document le topic dont les mots-clés
    sont les plus représentés dans son lemma_text.
    """
    for doc in documents:
        lemmas_set = set(doc.get("lemma_text", "").lower().split())
        scores = {
            tid: len(lemmas_set & info["words"])
            for tid, info in TOPIC_KEYWORDS.items()
        }
        best_tid = max(scores, key=scores.get)
        doc["topic_id"]    = best_tid if scores[best_tid] > 0 else -1
        doc["topic_label"] = TOPIC_KEYWORDS.get(best_tid, {}).get("label", "noise")
    return documents

# %% [markdown]
# ### 1.4 Sauvegarde du rapport de topics
#
# **Consigne :**
#
# Écrivez la fonction `save_topic_report(documents, output_path)` qui
# sauvegarde dans `topic_report.json` :
# - le nombre de topics
# - les mots représentatifs de chaque topic (top 10)
# - la liste des actes (`doc_id`) assignés à chaque topic
# - les actes non assignés (topic −1 = bruit)

# %% [Votre code ici]
# def save_topic_report(documents: list[dict], output_path: str) -> None:
#     ...

# Appel attendu :
# documents = aggregate_to_documents(RECORDS)
# documents = simulate_bertopic(documents)   # ou run_bertopic(documents)
# save_topic_report(documents, "topic_report.json")

# %% [markdown]
# ---
# ## Étape 2 — Extraction de relations PER — LOC — DATE
#
# **Consigne :**
#
# Écrivez la fonction `extract_relations(records)` qui parcourt le corpus
# enregistrement par enregistrement et extrait les triplets de relations
# en exploitant les `ner_spans` de chaque ligne.
#
# Relations à extraire (Chapitre 8, Tableau 2.1) :
# - `porte_titre`  : (PER, TITLE) dans la même ligne
# - `réside_à`     : (PER, LOC)   dans la même ligne
# - `agit_lors_de` : (PER, DATE)  dans la même ligne
#
# Chaque triplet doit contenir les clés :
# `sujet`, `relation`, `objet`, `confiance`, `method`, `source_line`
#
# **Validation manuelle :**
#
# Après extraction, prélevez un échantillon aléatoire de 20 triplets
# et calculez la précision en considérant comme correct tout triplet
# dont la confiance >= 0.70 (simulant l'annotation humaine).
# Sauvegardez le résultat dans `precision_report.json`.

# %% [Votre code ici]
# def extract_relations(records: list[dict]) -> list[dict]:
#     ...
#
# def compute_extraction_precision(triplets: list[dict],
#                                   sample_size: int = 20) -> dict:
#     ...

# %% [markdown]
# ---
# ## Étape 3 — Résolution de coréférence par règles
#
# **Consigne :**
#
# Écrivez la fonction `build_coref_chains(records)` qui parcourt le corpus
# dans l'ordre et construit les chaînes de coréférence selon les règles
# du Chapitre 8, section 3.2 :
#
# 1. Forme exacte déjà vue → même chaîne canonique.
# 2. Sous-forme d'une entité plus longue du même type → même chaîne.
#
# Retourne un dict `{canonical_id: {"canonical", "mentions", "type",
# "source_lines"}}`.
#
# Écrivez ensuite `resolve_coref_mentions(records, chains)` qui enrichit
# chaque `ner_span` avec les champs `canonical_id` et `canonical`.
#
# Sauvegardez les chaînes dans `coref_chains.json`.

# %% [Votre code ici]
# def build_coref_chains(records: list[dict]) -> dict:
#     ...
#
# def resolve_coref_mentions(records: list[dict], chains: dict) -> list[dict]:
#     ...

# %% [markdown]
# ---
# ## Étape 4 — Construction du graphe et export JSON-LD
#
# **Consigne :**
#
# Écrivez la fonction `build_knowledge_graph(records, triplets, chains)`
# qui construit un graphe NetworkX orienté (`nx.DiGraph`) :
#
# - Nœuds : une entité canonique par chaîne de coréférence.
#   Attributs requis : `label`, `type`, `n_mentions`, `polygon_ref`
#   (polygon de la première ligne source).
# - Arêtes : un triplet de relation par arête.
#   Attributs requis : `relation`, `confiance`, `method`.
#
# Calculez et affichez : nombre de nœuds, d'arêtes, densité, top-5 hubs
# (nœuds à fort degré entrant).
#
# Écrivez ensuite `graph_to_jsonld(G, output_path)` qui exporte le graphe
# en JSON-LD (voir Chapitre 8, section 4.2 pour le `@context` minimal).
# Chaque nœud doit inclure son `polygon_ref`.

# %% [FOURNI — contexte JSON-LD]
JSONLD_CONTEXT = {
    "@vocab":          "http://schema.org/",
    "medieval":        "http://example.org/medieval#",
    "tei":             "http://www.tei-c.org/ns/1.0/",
    "porte_titre":     "medieval:porteTitre",
    "réside_à":        "medieval:residesAt",
    "agit_lors_de":    "medieval:agitLorsDe",
    "signa_à":         "medieval:signaAt",
    "appartient_à":    "medieval:appartientA",
    "polygon_ref":     "medieval:polygonRef",
    "n_mentions":      "medieval:nMentions",
    "source_lines":    "medieval:sourceLines",
}

# %% [Votre code ici]
# def build_knowledge_graph(records, triplets, chains) -> nx.DiGraph:
#     ...
#
# def graph_to_jsonld(G: nx.DiGraph, output_path: str) -> None:
#     ...

# %% [markdown]
# ---
# ## Étape 5 — Export TEI-XML par acte
#
# **Consigne :**
#
# Écrivez la fonction `records_to_tei_file(doc_records, doc_id,
# output_dir, coref_chains)` qui produit un fichier TEI-XML pour un acte.
#
# Contraintes :
# - Namespace TEI : `http://www.tei-c.org/ns/1.0`
# - Chaque ligne → `<lb n="{line_id}" facs="{polygon_ref}"/>`
# - Tokens → `<w lemma="{lemma}" pos="{pos}">{token}</w>`
# - Entités NER imbriquées autour des `<w>` correspondants :
#   PER → `<persName key="{canonical_id}">`, LOC → `<placeName>`,
#   DATE → `<date>`, ORG → `<orgName>`, TITLE → `<roleName type="nobility">`
# - Utiliser `lxml.etree` pour la génération.
#
# Produire un fichier par acte dans `tei_output/`.
#
# **Validation :** si `xmllint` est disponible, vérifier avec :
# ```bash
# xmllint --noout tei_output/*.xml
# ```

# %% [FOURNI — mapping NER → balise TEI]
NER_TO_TEI = {
    "PER":   "persName",
    "LOC":   "placeName",
    "DATE":  "date",
    "ORG":   "orgName",
    "TITLE": "roleName",
}
TEI_NS = "http://www.tei-c.org/ns/1.0"

# %% [Votre code ici]
# def records_to_tei_file(doc_records: list[dict], doc_id: str,
#                          output_dir: str, coref_chains: dict) -> str:
#     ...
#
# # Exporter les 8 actes
# for acte_id in ACTES:
#     doc_records = [r for r in RECORDS if r["line_id"].startswith(acte_id)]
#     records_to_tei_file(doc_records, acte_id, "tei_output", chains)

# %% [markdown]
# ---
# ## Étape 6 — Data contract NLP v2
#
# **Consigne :**
#
# Écrivez la fonction `update_to_v2(record, doc, triplets, chains, G)`
# qui met à jour un enregistrement v1 vers le data contract v2 en ajoutant :
#
# - `schema_version: "2.0"`
# - `topics` : liste de `{"topic_id", "label"}` depuis le document parent
# - `relations` : liste des triplets dont `source_line == record["line_id"]`
# - `coref_chain` : `canonical_id` de la première entité PER (ou `null`)
# - `graph_node_id` : `"medieval:{coref_chain}"` (ou `"medieval:unknown_..."`)
# - `tei_ref` : `"tei_output/{doc_id}.xml#{line_id}"`
#
# Appliquez cette fonction à tous les enregistrements et sauvegardez
# dans `enriched_corpus_v2.jsonl` (une ligne JSON par enregistrement).

# %% [Votre code ici]
# def update_to_v2(record, doc, triplets, chains, G) -> dict:
#     ...
#
# # v2_records = [update_to_v2(r, line_to_doc[r["line_id"]], ...) for r in RECORDS]
# # with open("enriched_corpus_v2.jsonl", "w", ...) as f: ...

# %% [markdown]
# ---
# ## Étape 7 — Protocole de rétroaction HTR
#
# **Consigne :**
#
# Écrivez la fonction `build_htr_feedback(records)` qui compare
# `transcription` (HTR brut) et `normalized` (corrigé) pour chaque
# enregistrement et construit une liste de corrections documentées.
#
# Critères d'inclusion :
# - `confidence >= 0.80` (ligne globalement bien reconnue)
# - `CER entre transcription et normalized >= 0.03` (correction non triviale)
#
# Chaque correction doit contenir :
# `line_id`, `transcription_htr`, `forme_corrigee`, `cer`,
# `correction_type` (`"abréviation"` / `"graphie_uv"` / `"terminaison"` /
# `"autre"`), `confidence_htr`, `polygon_ref`.
#
# Sauvegardez dans `htr_feedback.json` avec :
# - `n_corrections` : nombre total
# - `type_distribution` : comptage par type
# - `mean_cer` : CER moyen des corrections
# - `corrections` : liste triée par CER décroissant
#
# **Attention :** ajoutez un champ `"validation_required": true` et une
# note rappelant qu'un échantillon de 10 % doit être validé manuellement
# avant injection dans les données d'entraînement HTR.

# %% [Votre code ici]
# def build_htr_feedback(records: list[dict],
#                         min_confidence: float = 0.80,
#                         min_cer: float = 0.03) -> list[dict]:
#     ...

# %% [markdown]
# ---
# ## Étape 8 — Interface de requête minimale
#
# **Consigne :**
#
# Écrivez la fonction `search_corpus(records, query, entity_type,
# topic_label, min_confidence)` qui filtre le corpus v2 selon ces
# quatre dimensions (tous les paramètres sont optionnels).
#
# Écrivez également `format_results(results, max_display=10)` qui affiche
# les résultats de façon lisible : `line_id`, texte tronqué à 80 chars,
# entités, topics.
#
# Testez avec au minimum ces quatre requêtes :
# ```python
# search_corpus(v2_records, query="normandie")
# search_corpus(v2_records, entity_type="PER")
# search_corpus(v2_records, topic_label="juridique")
# search_corpus(v2_records, query="acte", entity_type="PER")
# ```

# %% [Votre code ici]
# def search_corpus(records, query="", entity_type=None,
#                   topic_label=None, min_confidence=0.0) -> list[dict]:
#     ...
#
# def format_results(results: list[dict], max_display: int = 10) -> None:
#     ...

# %% [markdown]
# ---
# ## Récapitulatif des livrables
#
# Vérifiez que tous les fichiers sont produits avant la fin de la séance.
#
# | Livrable | Étape | Format |
# |---|---|---|
# | `topic_report.json` | 1 | JSON |
# | `knowledge_graph.jsonld` | 4 | JSON-LD |
# | `tei_output/acte_*.xml` | 5 | TEI-XML (×8) |
# | `enriched_corpus_v2.jsonl` | 6 | JSONL |
# | `htr_feedback.json` | 7 | JSON |
# | Interface de requête | 8 | Notebook/CLI |
#
# **Vérification de cohérence :**
# - Chaque nœud du graphe JSON-LD doit avoir un `polygon_ref` non vide.
# - Chaque ligne de `enriched_corpus_v2.jsonl` doit avoir un `tei_ref`
#   pointant vers un fichier existant dans `tei_output/`.
# - Le `split_hash` du Jour 2 doit figurer dans `htr_feedback.json`.

# %% [FOURNI — vérification finale]
def check_deliverables() -> None:
    """Vérifie que les six livrables attendus sont présents et cohérents."""
    issues = []

    # 1. topic_report.json
    if not Path("topic_report.json").exists():
        issues.append("ABSENT : topic_report.json")
    else:
        with open("topic_report.json", encoding="utf-8") as f:
            tr = json.load(f)
        if not tr.get("topics"):
            issues.append("topic_report.json : champ 'topics' vide")

    # 2. knowledge_graph.jsonld
    if not Path("knowledge_graph.jsonld").exists():
        issues.append("ABSENT : knowledge_graph.jsonld")
    else:
        with open("knowledge_graph.jsonld", encoding="utf-8") as f:
            kg = json.load(f)
        n_with_polygon = sum(1 for e in kg.get("@graph", [])
                             if e.get("polygon_ref"))
        print(f"  Graphe : {len(kg.get('@graph', []))} nœuds, "
              f"{n_with_polygon} avec polygon_ref")

    # 3. tei_output/
    tei_files = list(Path("tei_output").glob("*.xml")) \
                if Path("tei_output").exists() else []
    if len(tei_files) == 0:
        issues.append("ABSENT : aucun fichier dans tei_output/")
    else:
        print(f"  TEI : {len(tei_files)} fichiers")
        for tei_path in tei_files[:2]:
            try:
                etree.parse(str(tei_path))
            except etree.XMLSyntaxError as e:
                issues.append(f"TEI invalide : {tei_path.name} — {e}")

    # 4. enriched_corpus_v2.jsonl
    if not Path("enriched_corpus_v2.jsonl").exists():
        issues.append("ABSENT : enriched_corpus_v2.jsonl")
    else:
        with open("enriched_corpus_v2.jsonl", encoding="utf-8") as f:
            v2 = [json.loads(l) for l in f if l.strip()]
        missing_tei = [r for r in v2 if not r.get("tei_ref")]
        if missing_tei:
            issues.append(f"{len(missing_tei)} enregistrements sans tei_ref")
        print(f"  Data contract v2 : {len(v2)} enregistrements")

    # 5. htr_feedback.json
    if not Path("htr_feedback.json").exists():
        issues.append("ABSENT : htr_feedback.json")
    else:
        with open("htr_feedback.json", encoding="utf-8") as f:
            fb = json.load(f)
        print(f"  Rétroaction HTR : {fb.get('n_corrections', 0)} corrections")

    # Résultat
    if issues:
        print("\nProblèmes détectés :")
        for iss in issues:
            print(f"  ⚠  {iss}")
    else:
        print("\nTous les livrables sont présents et cohérents.")

# Appeler en fin de séance :
# check_deliverables()
