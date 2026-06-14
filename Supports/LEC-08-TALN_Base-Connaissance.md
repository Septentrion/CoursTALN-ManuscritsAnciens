# Chapitre 8 — Base de connaissances médiévale interrogeable : TEI, graphe et data contract final

**Module NLP · Master Data/IA · MD5 Volet 2 · 2026**  
TP autonome — 3 heures 30

---

## Avant-propos : le Jour 4 comme acte de clôture

Le Chapitre 7 a introduit les outils : BERTopic pour la structure thématique, l'extraction de relations par prompting, NetworkX pour le graphe, TEI-XML pour la publication scientifique, JSON-LD pour le web sémantique, et la boucle de rétroaction vers le modèle HTR. Il a posé les principes et illustré les mécanismes.

Ce chapitre met ces outils en œuvre de façon intégrée, sur votre corpus réel. La différence n'est pas triviale. Quand vous appliquez BERTopic à 40 phrases réparties sur deux types de documents, vous n'obtenez pas des topics propres et lisibles — vous obtenez du bruit, des clusters instables, et des topics qui se recoupent. C'est normal, et ce chapitre vous explique comment lire ces résultats honnêtement, comment les valider, et comment décider ce qui mérite d'être conservé.

De même, une extraction de relations par prompting sur des chartes médiévales normalisées produit des triplets dont certains sont faux, d'autres incomplets, d'autres ambigus. La précision calculée sur 50 triplets validés manuellement n'est pas un score de victoire — c'est une mesure de confiance qui détermine ce que vous pouvez affirmer sur votre corpus et ce que vous devez garder comme incertain.

Le fil conducteur de ce chapitre est la **traçabilité** : chaque affirmation produite — un topic, un triplet, une résolution de coréférence — doit pouvoir être vérifiée dans le document source via le `polygon_ref`. Un graphe de connaissances sans ancrage à la source primaire n'est pas un outil de recherche — c'est une hallucination organisée.

---

## 1. BERTopic sur le corpus agrégé : interprétation des topics

### 1.1 Agrégation des lignes en documents

Les entrées du data contract du Jour 3 sont des lignes de manuscrit, pas des documents. BERTopic opère sur des documents — des unités textuelles suffisamment longues pour que leurs embeddings capturent la sémantique. Avant d'appliquer BERTopic, il faut donc agréger les lignes en unités documentaires pertinentes.

Trois niveaux d'agrégation sont possibles, avec des conséquences différentes sur les topics produits :

**Par folio ou page :** toutes les lignes d'un même folio sont concaténées en un document. C'est le niveau naturel du manuscrit. Il produit des topics qui reflètent le contenu d'une page — souvent hétérogène pour les chartes, où une même page peut contenir un acte fiscal et un acte de donation.

**Par acte ou pièce documentaire :** si le corpus est structuré par acte (ce qui est le cas pour les chartes), chaque acte constitue un document. C'est le niveau sémantique le plus cohérent : un acte traite d'un seul sujet juridique. C'est le niveau recommandé pour obtenir des topics interprétables.

**Par lot thématique connu :** si vous avez déjà des métadonnées sur les types documentaires (comptes de bailliage, actes de donation, rôles fiscaux), grouper par type avant BERTopic permet de valider que les topics produits correspondent aux types attendus. Cette validation croisée est utile pour estimer la cohérence du modèle.

```python
from collections import defaultdict

def aggregate_lines_to_documents(enriched_records: list[dict],
                                  level: str = "acte") -> list[dict]:
    """
    Agrège les lignes du data contract en documents pour BERTopic.

    Paramètres
    ----------
    enriched_records : list[dict]   data contracts enrichis du Jour 3
    level            : str          "acte" | "folio" | "all"
                                    "acte" : groupe par prefixe de line_id avant _l\d+
                                    "folio": groupe par prefixe avant _lXX
                                    "all"  : concatène tout le corpus en un seul doc

    Retourne
    --------
    list[dict] avec "doc_id", "text", "lemmas", "ner_spans", "line_ids"
    """
    import re
    groups = defaultdict(list)

    for rec in enriched_records:
        lid = rec["line_id"]
        if level == "acte":
            key = re.sub(r'_l\d+$', '', lid)   # ex. charte_1346_fol12
        elif level == "folio":
            key = re.sub(r'_l\d+$', '', lid)
            key = re.sub(r'_fol\d+$', '', key)  # ex. charte_1346
        else:
            key = "corpus"
        groups[key].append(rec)

    documents = []
    for doc_id, recs in sorted(groups.items()):
        text   = " ".join(r["normalized"] for r in recs)
        lemmas = [l for r in recs for l in r.get("lemmas", [])]
        spans  = [s for r in recs for s in r.get("ner_spans", [])]
        lines  = [r["line_id"] for r in recs]
        documents.append({
            "doc_id":    doc_id,
            "text":      text,
            "lemmas":    lemmas,
            "ner_spans": spans,
            "line_ids":  lines,
        })
    return documents
```

### 1.2 Application de BERTopic et calibration des hyperparamètres

Sur un corpus de taille modeste (40 à 200 documents agrégés), BERTopic nécessite une calibration soigneuse de `min_cluster_size` dans HDBSCAN. La valeur par défaut (`min_cluster_size=10`) est conçue pour des corpus de plusieurs milliers de documents. Sur 40 documents, elle est trop élevée et produit un unique cluster de bruit.

```python
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

documents = aggregate_lines_to_documents(enriched_records, level="acte")
texts     = [d["text"] for d in documents]

# Modèle d'embeddings français — utiliser le même que pour la NER (cohérence)
embedding_model = SentenceTransformer("almanach/camembert-base")

# Hyperparamètres calibrés pour corpus court (40–200 documents)
umap_model = UMAP(
    n_neighbors   = 5,      # petit corpus : peu de voisins pour éviter le collapse
    n_components  = 3,      # espace cible réduit (3D au lieu de 5D)
    min_dist      = 0.0,
    metric        = "cosine",
    random_state  = 42,
)
hdbscan_model = HDBSCAN(
    min_cluster_size  = 3,  # cluster minimal de 3 documents sur 40
    min_samples       = 2,  # seuil de densité locale
    metric            = "euclidean",
    prediction_data   = True,
)
# Stop words médiévaux pour c-TF-IDF
stop_words_med = ['le','la','les','de','du','des','en','et','à','au','pour',
                  'un','une','il','elle','se','qui','que','ce','son','sa','ses',
                  'li','lo','lesdits','ledit','ladite','audit','dudit']
vectorizer_model = CountVectorizer(stop_words=stop_words_med, min_df=1)

topic_model = BERTopic(
    embedding_model   = embedding_model,
    umap_model        = umap_model,
    hdbscan_model     = hdbscan_model,
    vectorizer_model  = vectorizer_model,
    language          = "french",
    nr_topics         = "auto",       # laisser HDBSCAN décider du nombre
    calculate_probabilities = True,
    verbose           = True,
)

topics, probs = topic_model.fit_transform(texts)
topic_info    = topic_model.get_topic_info()
print(topic_info[["Topic","Count","Name","Representation"]])
```

### 1.3 Les quatre topics attendus et leur interprétation

Sur un corpus de chartes normandes du XIVe siècle, quatre topics sont typiquement identifiés avec une cohérence acceptable. Leur lecture ne se fait pas en regardant les mots bruts — elle se fait en croisant les mots représentatifs avec les entités nommées des documents assignés à chaque topic.

**Topic liturgique.** Mots représentatifs : *abbaye, moine, âme, chapelle, dieu, prière, offrande, salut, grâce, fondation*. Documents assignés : actes de donation à des établissements religieux, fondations de messes anniversaires, testaments pieux. Entités nommées dominantes : ORG (abbayes, chapitres), DATE (fêtes liturgiques — Pâques, Noël, Pentecôte). Ce topic est le plus facile à identifier parce que son vocabulaire est lexicalement distinct des autres.

**Topic juridique.** Mots représentatifs : *acte, charte, sceller, témoin, consentir, accord, stipuler, droits, renoncer, ratifier*. Documents assignés : actes de vente, de cession de droits, de reconnaissance de dettes. Entités nommées dominantes : PER (parties signataires), TITLE (notaires, témoins), DATE (date de l'acte). Ce topic est le plus homogène sur le plan formel : les formules juridiques varient peu d'un acte à l'autre, ce qui produit un topic très cohérent mais peu informatif sur le contenu.

**Topic fiscal.** Mots représentatifs : *denier, livre, taxe, bailli, lever, rendre, compte, somme, redevance, fermage*. Documents assignés : comptes de bailliage, rôles de taille, quittances de paiement. Entités nommées dominantes : DATE (exercices fiscaux), LOC (localités imposées), TITLE (baillis, prévôts). Ce topic est le plus hétérogène géographiquement — les lieux varient beaucoup d'un compte à l'autre.

**Topic narratif ou politique.** Mots représentatifs : *roi, guerre, chevalier, armée, siège, victoire, hommage, vassal, fief, seigneur*. Documents assignés : chroniques, lettres royales, actes d'hommage féodal. Ce topic est souvent le plus difficile à isoler proprement parce qu'il partage du vocabulaire avec le topic juridique (hommage, fief) et le topic nobiliaire.

### 1.4 Validation et interprétation critique

Deux métriques permettent d'évaluer la qualité des topics produits.

**Cohérence c_v :** calculée sur les 10 mots représentatifs de chaque topic. Une valeur supérieure à 0.50 indique un topic acceptable ; supérieure à 0.60, un topic interprétable avec confiance. Sur un corpus court, des valeurs entre 0.40 et 0.55 sont fréquentes et ne disqualifient pas le modèle — elles signalent simplement qu'une validation humaine est nécessaire.

```python
from gensim.models.coherencemodel import CoherenceModel
import gensim.corpora as corpora

# Construction du dictionnaire Gensim depuis les textes tokenisés
tokenized = [d["lemmas"] for d in documents]
dictionary = corpora.Dictionary(tokenized)
corpus_bow  = [dictionary.doc2bow(tok) for tok in tokenized]

# Mots représentatifs extraits de BERTopic
topic_words = [
    [word for word, _ in topic_model.get_topic(tid)]
    for tid in sorted(topic_model.get_topics().keys())
    if tid != -1          # exclure le topic "bruit"
]

coherence_model = CoherenceModel(
    topics    = topic_words,
    texts     = tokenized,
    dictionary= dictionary,
    coherence = "c_v",
)
print(f"Cohérence c_v moyenne : {coherence_model.get_coherence():.4f}")
```

**Validation par entités :** pour chaque topic, vérifier que les entités nommées des documents assignés sont cohérentes avec l'interprétation proposée. Un topic "liturgique" dont les documents contiennent majoritairement des PER laïcs et aucune ORG religieuse est suspect — soit le topic est mal nommé, soit des documents ont été assignés incorrectement.

```python
def validate_topic_by_entities(documents:   list[dict],
                                 topics:      list[int],
                                 topic_model) -> None:
    """
    Pour chaque topic, affiche la distribution des types d'entités
    des documents assignés. Permet de valider l'interprétation thématique.
    """
    from collections import Counter
    topic_entity_types = defaultdict(Counter)

    for doc, topic_id in zip(documents, topics):
        if topic_id == -1:
            continue   # documents de bruit
        for span in doc.get("ner_spans", []):
            topic_entity_types[topic_id][span["label"]] += 1

    for tid, counter in sorted(topic_entity_types.items()):
        total = sum(counter.values())
        words = [w for w, _ in topic_model.get_topic(tid)[:5]]
        print(f"\nTopic {tid:2d} [{', '.join(words)}] :")
        for etype, count in counter.most_common():
            print(f"  {etype:8s} : {count:3d} ({count/total*100:.0f}%)")
```

---

## 2. Extraction de relations simples PER–LOC–DATE

### 2.1 Le triplet ciblé et sa justification

Le syllabus cible les triplets de la forme *(PER, relation, LOC)* et *(PER, relation, DATE)*. Ce choix est délibéré : ces triplets sont les plus fréquents dans les chartes médiévales et les plus utiles pour les historiens. Savoir que *"Jean de Normandie"* est associé à *"Gisors"* en *"mars 1346"* permet de construire des chronologies d'acteurs, de cartographier des réseaux de seigneuries, et d'identifier des co-présences dans les actes.

Les relations ciblées sont au nombre de cinq :

| Relation | Type sujet | Type objet | Exemple |
|---|---|---|---|
| `réside_à` | PER | LOC | Jean de Normandie – Gisors |
| `porte_titre` | PER | TITLE | Jean de Normandie – sénéchal |
| `agit_lors_de` | PER | DATE | Jean de Normandie – mars 1346 |
| `signe_acte_à` | PER | LOC | le bailli – Paris |
| `appartient_à` | PER/LOC | ORG | le chapitre – abbaye Saint-Denis |

### 2.2 Extraction par prompting avec schéma structuré

Le prompting avec schéma contraint le LLM à produire un JSON directement exploitable et limite les hallucinations. La clé est de fournir un exemple dans le prompt (*few-shot*) : un LLM sans exemple produit souvent des triplets dont les entités ne correspondent pas aux annotations NER, rendant l'alignement avec le graphe impossible.

```python
import json, re

RELATION_SCHEMA = {
    "réside_à":    "PER → LOC : une personne est associée à un lieu (origine, résidence, fief)",
    "porte_titre": "PER → TITLE : une personne porte un titre ou une fonction",
    "agit_lors_de":"PER → DATE : une personne agit à une date ou période donnée",
    "signe_acte_à":"PER → LOC : une personne signe un acte dans un lieu",
    "appartient_à":"PER/LOC → ORG : appartenance à une organisation",
}

def build_relation_prompt(record: dict) -> str:
    """
    Construit le prompt d'extraction de relations pour un enregistrement.
    Inclut le texte normalisé et les entités déjà identifiées par le NER
    du Jour 3, ce qui réduit les hallucinations d'entités.
    """
    schema_lines = "\n".join(
        f"  {rel}: {desc}" for rel, desc in RELATION_SCHEMA.items()
    )
    ner_context = ", ".join(
        f"{s['text']} ({s['label']})"
        for s in record.get("ner_spans", [])
    ) or "aucune entité identifiée"

    return f"""Extrait les relations du texte médiéval suivant.
Retourne UNIQUEMENT un tableau JSON valide, sans texte ni balises autour.

Relations cibles :
{schema_lines}

Entités déjà identifiées par l'analyse NER :
  {ner_context}

Texte normalisé : "{record['normalized']}"

Exemple de sortie attendue :
[
  {{"sujet": "Jean de Normandie", "relation": "réside_à",    "objet": "Normandie", "confiance": 0.9}},
  {{"sujet": "Jean de Normandie", "relation": "porte_titre", "objet": "sénéchal",  "confiance": 0.95}}
]

Sortie :"""


def extract_relations(record:   dict,
                       call_llm: callable) -> list[dict]:
    """
    Extrait les relations d'un enregistrement via prompting LLM.

    Paramètre call_llm
    ------------------
    Fonction prenant un prompt (str) et retournant la réponse du LLM (str).
    Peut appeler l'API Anthropic, OpenAI, ou un modèle local HuggingFace.

    Retourne
    --------
    list[dict]  triplets extraits, chacun avec :
        {"sujet", "relation", "objet", "confiance", "source_line"}
    """
    prompt = build_relation_prompt(record)
    raw    = call_llm(prompt)

    # Nettoyer les balises Markdown éventuelles avant parsing JSON
    raw = re.sub(r'^```(?:json)?\s*', '', raw.strip())
    raw = re.sub(r'\s*```$',         '', raw.strip())

    try:
        relations = json.loads(raw)
        for rel in relations:
            rel["source_line"] = record["line_id"]
        return relations
    except (json.JSONDecodeError, TypeError):
        return []
```

### 2.3 Validation manuelle et calcul de précision

La validation manuelle sur un échantillon de 50 triplets est une étape non optionnelle. Un triplet est considéré **correct** si : (a) les entités sujet et objet correspondent exactement à des spans NER identifiés, (b) la relation est du bon type, et (c) le triplet est attesté dans le texte source (pas une inférence non textuelle).

```python
def validate_relations_sample(relations: list[dict],
                               records:   list[dict],
                               n_sample:  int = 50,
                               seed:      int = 42) -> dict:
    """
    Interface de validation manuelle sur un échantillon aléatoire.
    Affiche chaque triplet avec son contexte source pour jugement humain.

    Retourne
    --------
    dict avec "tp", "fp", "fn", "precision", "recall", "f1"
    """
    import random
    rng = random.Random(seed)
    sample = rng.sample(relations, min(n_sample, len(relations)))

    records_by_id = {r["line_id"]: r for r in records}

    tp = fp = 0
    judgments = []

    for rel in sample:
        source = records_by_id.get(rel.get("source_line", ""), {})
        context = source.get("normalized", "N/A")
        ner_spans = [s["text"] for s in source.get("ner_spans", [])]

        print(f"\nContexte : {context}")
        print(f"NER connu: {ner_spans}")
        print(f"Triplet  : ({rel['sujet']}, {rel['relation']}, {rel['objet']})")
        print(f"Confiance LLM : {rel.get('confiance', '?')}")

        verdict = input("Correct ? [o/n] : ").strip().lower()
        if verdict == 'o':
            tp += 1
            judgments.append({**rel, "verdict": "correct"})
        else:
            fp += 1
            reason = input("  Raison (type_error/entity_error/not_attested/other) : ")
            judgments.append({**rel, "verdict": "incorrect", "reason": reason})

    # Estimation du rappel : triplets manqués dans l'échantillon
    fn_estimate = int(len(sample) * 0.30)   # estimation : 30% de triplets manqués
    precision   = tp / max(tp + fp, 1)
    recall_est  = tp / max(tp + fn_estimate, 1)
    f1_est      = 2 * precision * recall_est / max(precision + recall_est, 1e-9)

    print(f"\n=== Résultat validation ({n_sample} triplets) ===")
    print(f"TP={tp}, FP={fp}, FN estimé≈{fn_estimate}")
    print(f"Précision : {precision:.3f}")
    print(f"Rappel    : {recall_est:.3f} (estimé)")
    print(f"F1        : {f1_est:.3f} (estimé)")

    return {"tp": tp, "fp": fp, "fn_estimate": fn_estimate,
            "precision": round(precision, 3),
            "recall_estimate": round(recall_est, 3),
            "f1_estimate": round(f1_est, 3),
            "judgments": judgments}
```

**Valeurs attendues :** sur un corpus de chartes médiévales normalisées, un LLM guidé par schéma et entités NER pré-identifiées atteint typiquement une précision de 0.80–0.85, un rappel estimé de 0.70–0.80, et un F1 de 0.75–0.82. Les erreurs les plus fréquentes sont les triplets `réside_à` où le LLM confond la localisation de l'acte avec la résidence de la personne, et les triplets `appartient_à` pour les ORG implicites non mentionnées textuellement.

---

## 3. Résolution de coréférence sur les entités du Jour 3

### 3.1 Le problème sur corpus multi-actes

La coréférence dans les chartes médiévales présente une difficulté spécifique qui n'existe pas dans les textes narratifs modernes : une même personne peut apparaître dans plusieurs actes distincts du corpus, sous des formes de mention différentes. *"Jean de Normandie"* dans la charte de 1346, *"Messire Jean"* dans la charte de 1348, et *"ledit seigneur"* dans la charte de 1349 peuvent désigner le même individu — ou trois individus différents homonymes. Seul le contexte historique (dates, lieux, titres associés) permet de trancher.

La résolution de coréférence opère donc à deux niveaux dans votre corpus.

**Niveau intra-acte :** au sein d'un même document, les pronoms (*il*, *elle*) et les anaphores (*ledit seigneur*, *ladite dame*) coréfèrent généralement à la dernière entité mentionnée de même type. Ces cas sont résolvables par les règles simples introduites au Chapitre 7.

**Niveau inter-actes :** entre documents, la résolution nécessite une comparaison des mentions avec leurs contextes. Un graphe de coréférence (*entity linking*) est construit en regroupant les mentions qui désignent le même individu dans le monde réel.

### 3.2 Construction des chaînes de coréférence

```python
from difflib import SequenceMatcher

def build_coref_chains(enriched_records: list[dict],
                        similarity_threshold: float = 0.75) -> dict:
    """
    Construit les chaînes de coréférence inter-actes.

    Algorithme :
    1. Collecter toutes les mentions d'entités PER et LOC.
    2. Grouper par similarité de texte (SequenceMatcher) et de type.
    3. Choisir la forme canonique (mention la plus longue et la plus fréquente).
    4. Résoudre les anaphores intra-actes par règles.

    Retourne
    --------
    dict { "canonical_id": { "canonical_text", "type", "mentions": [...] } }
    """
    from collections import defaultdict

    # Collecter toutes les mentions PER et LOC
    mentions_by_type = defaultdict(list)
    for rec in enriched_records:
        for span in rec.get("ner_spans", []):
            if span["label"] in ("PER", "LOC", "ORG"):
                mentions_by_type[span["label"]].append({
                    "text":     span["text"],
                    "line_id":  rec["line_id"],
                    "start":    span["start"],
                    "end":      span["end"],
                })

    chains = {}

    for etype, mentions in mentions_by_type.items():
        # Regroupement par similarité de texte
        groups = []
        for mention in mentions:
            placed = False
            for group in groups:
                rep = group[0]["text"]
                sim = SequenceMatcher(None,
                                      mention["text"].lower(),
                                      rep.lower()).ratio()
                if sim >= similarity_threshold:
                    group.append(mention)
                    placed = True
                    break
            if not placed:
                groups.append([mention])

        # Pour chaque groupe : forme canonique = mention la plus longue
        for group in groups:
            canonical_text = max(group, key=lambda m: len(m["text"]))["text"]
            canonical_id   = canonical_text.lower().replace(" ", "_")
            chains[canonical_id] = {
                "canonical_text": canonical_text,
                "type":           etype,
                "n_mentions":     len(group),
                "mentions":       group,
            }

    return chains


def resolve_intra_doc_anaphors(record:       dict,
                                coref_chains: dict) -> dict:
    """
    Résout les anaphores intra-document et enrichit l'enregistrement
    avec un champ "coref_chain" (identifiant canonique de l'entité).

    Règles :
    - "ledit / ladite / lesdits" → dernière PER ou LOC connue.
    - "il / elle" → dernière PER de genre correspondant.
    """
    tokens    = record["normalized"].split()
    ner_spans = record.get("ner_spans", [])

    span_by_start = {s["start"]: s for s in ner_spans}
    last_per      = None
    canonical_id  = None
    char_pos      = 0

    for tok in tokens:
        if tok.lower() in ('ledit', 'ladite', 'lesdits', 'lesdites'):
            if last_per:
                canonical_id = last_per["text"].lower().replace(" ", "_")

        if char_pos in span_by_start:
            span = span_by_start[char_pos]
            if span["label"] == "PER":
                last_per = span
                # Chercher l'identifiant canonique dans les chaînes
                for cid, chain in coref_chains.items():
                    if any(m["text"].lower() == span["text"].lower()
                           for m in chain["mentions"]):
                        canonical_id = cid
                        break

        char_pos += len(tok) + 1

    record["coref_chain"] = canonical_id
    return record
```

### 3.3 Évaluation qualitative des chaînes

Sur votre corpus de 40 phrases, la résolution de coréférence produit un nombre limité de chaînes — souvent entre 8 et 15 identités canoniques distinctes. L'évaluation quantitative (MUC, B³, CEAF) nécessite une référence annotée que vous n'avez pas. L'évaluation est donc qualitative : vérifier que les chaînes produites sont linguistiquement cohérentes et que les fusions inter-actes sont justifiées.

Deux questions guident cette vérification. Premièrement, les mentions regroupées dans une même chaîne sont-elles compatibles chronologiquement ? Si *"Jean de Normandie"* apparaît dans un acte de 1340 et *"Jean de Normandie"* dans un acte de 1390, il s'agit probablement de deux personnes différentes — cinquante ans d'écart pour un même individu est possible mais doit être noté comme incertain. Deuxièmement, les titres associés aux mentions d'une chaîne sont-ils cohérents ? Un même individu ne peut pas être simultanément *"bailli"* et *"sénéchal"* dans des actes contemporains.

---

## 4. Construction du graphe de connaissances

### 4.1 Enrichissement du graphe avec topics et coréférence

Le graphe du Chapitre 7 contenait des nœuds (entités) et des arêtes (relations). Ce chapitre l'enrichit avec deux types d'information supplémentaires : les topics BERTopic (attribut de document, propagé aux entités) et les identifiants canoniques de coréférence (qui fusionnent les nœuds dupliqués).

```python
import networkx as nx

def build_enriched_graph(enriched_records: list[dict],
                          relations:        list[dict],
                          coref_chains:     dict,
                          doc_topics:       dict) -> nx.DiGraph:
    """
    Construit le graphe de connaissances enrichi avec topics et coréférence.

    Paramètres
    ----------
    enriched_records : data contracts du Jour 3
    relations        : triplets extraits à l'Étape 2
    coref_chains     : chaînes de coréférence de l'Étape 3
    doc_topics       : {doc_id: [(topic_id, poids), ...]}
    """
    G = nx.DiGraph()

    # Mapping mention → canonical_id via les chaînes de coréférence
    mention_to_canonical = {}
    for cid, chain in coref_chains.items():
        for mention in chain["mentions"]:
            mention_to_canonical[mention["text"].lower()] = cid

    # Ajouter les nœuds canoniques
    for cid, chain in coref_chains.items():
        G.add_node(cid,
                    label    = chain["canonical_text"],
                    type     = chain["type"],
                    n_mentions = chain["n_mentions"])

    # Ajouter les entités non coréférentielles (DATE, TITLE isolés)
    for rec in enriched_records:
        for span in rec.get("ner_spans", []):
            nid = span["text"].lower().replace(" ", "_")
            if nid not in G:
                G.add_node(nid,
                            label      = span["text"],
                            type       = span["label"],
                            polygon    = rec.get("polygon_ref", ""),
                            source_line= rec["line_id"])

    # Ajouter les arêtes depuis les relations validées
    for rel in relations:
        s_raw  = rel["sujet"].lower()
        o_raw  = rel["objet"].lower()
        s_id   = mention_to_canonical.get(s_raw,
                     s_raw.replace(" ", "_"))
        o_id   = mention_to_canonical.get(o_raw,
                     o_raw.replace(" ", "_"))
        for nid in (s_id, o_id):
            if not G.has_node(nid):
                G.add_node(nid, label=nid, type="UNKNOWN")
        G.add_edge(s_id, o_id,
                   relation  = rel["relation"],
                   confiance = rel.get("confiance", 1.0),
                   source    = rel.get("source_line", ""))

    # Propager les topics aux nœuds via les lignes sources
    for rec in enriched_records:
        doc_id = re.sub(r'_l\d+$', '', rec["line_id"])
        topics = doc_topics.get(doc_id, [])
        for span in rec.get("ner_spans", []):
            nid = mention_to_canonical.get(span["text"].lower(),
                      span["text"].lower().replace(" ", "_"))
            if G.has_node(nid) and topics:
                existing = G.nodes[nid].get("topics", [])
                G.nodes[nid]["topics"] = existing + topics

    return G
```

### 4.2 Export JSON-LD complet avec ancrage spatial

L'export JSON-LD du Chapitre 7 était minimal. Ce chapitre produit un export complet qui inclut les attributs de topics, les identifiants canoniques de coréférence, et les références spatiales vers le Volet 1.

```python
import json

JSONLD_CONTEXT = {
    "@vocab":       "http://schema.org/",
    "medieval":     "http://example.org/medieval#",
    "tei":          "http://www.tei-c.org/ns/1.0/",
    "oa":           "http://www.w3.org/ns/oa#",
    # Relations médiévales
    "réside_à":     "medieval:residesAt",
    "porte_titre":  "medieval:porteTitre",
    "agit_lors_de": "medieval:agitLorsDe",
    "signe_acte_à": "medieval:signeActeA",
    "appartient_à": "medieval:appartientA",
    # Attributs
    "polygon_ref":  "medieval:polygonRef",
    "topic_label":  "medieval:topicLabel",
    "coref_id":     "medieval:corefCanonicalId",
    "n_mentions":   "medieval:nMentions",
    "tei_ref":      "medieval:teiRef",
}

def graph_to_jsonld_full(G:           nx.DiGraph,
                          coref_chains: dict,
                          output_path:  str) -> None:
    """
    Exporte le graphe enrichi en JSON-LD complet.
    Chaque nœud inclut ses attributs (type, topics, polygon, tei_ref).
    """
    import re
    entities = []

    for node_id, data in G.nodes(data=True):
        ent = {
            "@id":    f"medieval:{node_id}",
            "@type":  data.get("type", "Thing"),
            "name":   data.get("label", node_id),
        }
        # Attributs optionnels
        if data.get("polygon"):
            ent["polygon_ref"] = data["polygon"]
        if data.get("n_mentions"):
            ent["n_mentions"] = data["n_mentions"]
        if data.get("topics"):
            ent["topic_label"] = [t[0] for t in data["topics"]
                                  if isinstance(t, (list, tuple))]
        if data.get("source_line"):
            doc_id  = re.sub(r'_l\d+$', '', data["source_line"])
            ent["tei_ref"] = f"{doc_id}.xml#{data['source_line']}"

        # Relations sortantes
        for _, target, edata in G.out_edges(node_id, data=True):
            rel = edata.get("relation", "relatedTo")
            ent[rel] = {
                "@id":       f"medieval:{target}",
                "confiance": edata.get("confiance", 1.0),
            }

        entities.append(ent)

    jsonld_doc = {"@context": JSONLD_CONTEXT, "@graph": entities}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(jsonld_doc, f, indent=2, ensure_ascii=False)

    print(f"Graphe JSON-LD : {output_path}")
    print(f"  {G.number_of_nodes()} nœuds · {G.number_of_edges()} arêtes")
```

---

## 5. Export TEI : chaque acte vers un fichier XML

### 5.1 Principe : un fichier par acte, pas un fichier par ligne

Le Chapitre 7 a exposé la structure TEI. Ce chapitre précise la granularité : chaque acte (au sens de l'agrégation définie à l'Étape 1) donne lieu à un fichier XML distinct. Cette décision est conforme aux pratiques éditoriales en humanités numériques : les TEI guides recommandent que chaque document soit une unité de publication autonome, avec son propre `teiHeader` décrivant la source physique.

```python
from lxml import etree
import re, os

TEI_NS = "http://www.tei-c.org/ns/1.0"

def records_to_tei_file(records:     list[dict],
                         doc_id:      str,
                         output_dir:  str,
                         coref_chains: dict,
                         graph:        nx.DiGraph) -> str:
    """
    Produit un fichier TEI-XML pour un acte (ensemble de lignes).

    Nommage : {doc_id}.xml
    Chaque ligne devient un élément <lb/> avec @facs (polygon_ref).
    Les entités NER sont balisées avec persName, placeName, date, orgName, roleName.
    Les tokens POS sont dans des éléments <w> avec @lemma et @pos.

    Retourne
    --------
    str  chemin du fichier produit
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{doc_id}.xml")

    root    = etree.Element(f"{{{TEI_NS}}}TEI", nsmap={None: TEI_NS})
    header  = _build_tei_header(root, doc_id, records)
    text_el = etree.SubElement(root, f"{{{TEI_NS}}}text")
    body    = etree.SubElement(text_el, f"{{{TEI_NS}}}body")
    div     = etree.SubElement(body,  f"{{{TEI_NS}}}div",
                                type="acte", n=doc_id)

    for rec in records:
        _add_line_to_tei(div, rec, coref_chains)

    tree = etree.ElementTree(root)
    tree.write(output_path,
               pretty_print    = True,
               xml_declaration = True,
               encoding        = "UTF-8")
    return output_path


def _build_tei_header(root:    etree.Element,
                       doc_id:  str,
                       records: list[dict]) -> etree.Element:
    """Construit le teiHeader avec fileDesc et les métadonnées de l'acte."""
    header   = etree.SubElement(root, f"{{{TEI_NS}}}teiHeader")
    fileDesc = etree.SubElement(header, f"{{{TEI_NS}}}fileDesc")

    titleStmt = etree.SubElement(fileDesc, f"{{{TEI_NS}}}titleStmt")
    title_el  = etree.SubElement(titleStmt, f"{{{TEI_NS}}}title")
    title_el.text = f"Acte {doc_id} — corpus médiéval NLP 2026"

    pubStmt = etree.SubElement(fileDesc, f"{{{TEI_NS}}}publicationStmt")
    p_pub   = etree.SubElement(pubStmt, f"{{{TEI_NS}}}p")
    p_pub.text = "Produit par le pipeline NLP MD5 Volet 2"

    srcDesc  = etree.SubElement(fileDesc, f"{{{TEI_NS}}}sourceDesc")
    msDesc   = etree.SubElement(srcDesc,  f"{{{TEI_NS}}}msDesc")
    msId     = etree.SubElement(msDesc,   f"{{{TEI_NS}}}msIdentifier")
    # Remplir depuis les métadonnées du corpus si disponibles
    etree.SubElement(msId, f"{{{TEI_NS}}}idno").text = doc_id

    return header


def _add_line_to_tei(parent:       etree.Element,
                      record:       dict,
                      coref_chains: dict) -> None:
    """
    Ajoute une ligne annotée dans le parent TEI.
    Encode les entités NER (persName, placeName, date, orgName, roleName)
    et les tokens POS (<w lemma=… pos=…>).
    """
    NER_TO_TAG = {
        "PER":   "persName",
        "LOC":   "placeName",
        "DATE":  "date",
        "ORG":   "orgName",
        "TITLE": "roleName",
    }
    NER_ATTRIBS = {
        "PER":   {"type": "person"},
        "LOC":   {"type": "place"},
        "TITLE": {"type": "nobility"},
    }

    # Ancrage spatial : @facs pointe vers le polygone Volet 1
    lb = etree.SubElement(parent, f"{{{TEI_NS}}}lb",
                           n    = record["line_id"],
                           facs = record.get("polygon_ref", ""))

    tokens   = record["normalized"].split()
    pos_tags = record.get("pos_tags", ["_"] * len(tokens))
    lemmas   = record.get("lemmas",   tokens)

    # Construire un index des spans actifs par position de token
    span_start_idx = {}   # token_index → span
    span_end_idx   = {}   # token_index → True
    char = 0
    for i, tok in enumerate(tokens):
        for span in record.get("ner_spans", []):
            if char == span["start"]:
                span_start_idx[i] = span
            if char + len(tok) == span["end"]:
                span_end_idx[i] = span
        char += len(tok) + 1

    current_ner_elem = None
    i = 0
    while i < len(tokens):
        if i in span_start_idx:
            span     = span_start_idx[i]
            tag_name = NER_TO_TAG.get(span["label"], "name")
            attribs  = dict(NER_ATTRIBS.get(span["label"], {}))
            # Identifiant canonique de coréférence
            cid = span["text"].lower().replace(" ", "_")
            if cid in coref_chains:
                attribs["key"] = cid
            current_ner_elem = etree.SubElement(
                parent, f"{{{TEI_NS}}}{tag_name}", **attribs
            )

        target = current_ner_elem if current_ner_elem is not None else parent
        w = etree.SubElement(target, f"{{{TEI_NS}}}w",
                              lemma = lemmas[i] if i < len(lemmas) else tokens[i],
                              pos   = pos_tags[i] if i < len(pos_tags) else "_")
        w.text = tokens[i]

        if i in span_end_idx:
            current_ner_elem = None
        i += 1


def export_corpus_tei(enriched_records: list[dict],
                       output_dir:        str,
                       coref_chains:      dict,
                       graph:             nx.DiGraph) -> list[str]:
    """
    Exporte l'ensemble du corpus en fichiers TEI, un par acte.

    Retourne la liste des chemins produits.
    """
    # Grouper les enregistrements par acte
    from collections import defaultdict
    actes = defaultdict(list)
    for rec in enriched_records:
        doc_id = re.sub(r'_l\d+$', '', rec["line_id"])
        actes[doc_id].append(rec)

    output_paths = []
    for doc_id, records in sorted(actes.items()):
        path = records_to_tei_file(records, doc_id, output_dir,
                                    coref_chains, graph)
        output_paths.append(path)
        print(f"  {path}")

    print(f"\n{len(output_paths)} fichier(s) TEI produit(s) dans {output_dir}/")
    print(f"Validation : xmllint --relaxng tei_all.rng --noout {output_dir}/*.xml")
    return output_paths
```

---

## 6. Interface de requête minimale

### 6.1 Index plein-texte et filtre par entité ou topic

L'interface de requête ne nécessite pas de base de données — un index en mémoire Python suffit pour un corpus de quelques centaines de lignes. L'objectif est que les étudiants puissent interroger le corpus en ligne de commande ou dans le notebook, sans déployer de service.

```python
class MedievalCorpusIndex:
    """
    Index minimal pour le corpus médiéval enrichi.
    Supporte : recherche plein-texte, filtre par entité, filtre par topic.
    """

    def __init__(self,
                 enriched_records: list[dict],
                 doc_topics:       dict):
        self.records    = enriched_records
        self.doc_topics = doc_topics
        self._build_indexes()

    def _build_indexes(self) -> None:
        """Construit trois index en mémoire."""
        from collections import defaultdict

        # Index plein-texte : lemme → liste de line_ids
        self.text_index  = defaultdict(set)
        # Index entités   : (texte_entité.lower, type) → ligne_ids
        self.entity_index = defaultdict(set)
        # Index topics    : topic_id → doc_ids
        self.topic_index  = defaultdict(set)

        for rec in self.records:
            lid    = rec["line_id"]
            doc_id = re.sub(r'_l\d+$', '', lid)

            for lemma in rec.get("lemmas", []):
                self.text_index[lemma.lower()].add(lid)

            for span in rec.get("ner_spans", []):
                key = (span["text"].lower(), span["label"])
                self.entity_index[key].add(lid)

            for topic_id, _ in self.doc_topics.get(doc_id, []):
                self.topic_index[topic_id].add(lid)

    def search_text(self, query: str) -> list[dict]:
        """Recherche plein-texte sur les lemmes. Retourne les lignes correspondantes."""
        terms    = query.lower().split()
        line_ids = set.intersection(*(self.text_index.get(t, set()) for t in terms))
        return [r for r in self.records if r["line_id"] in line_ids]

    def filter_by_entity(self, text: str, etype: str = None) -> list[dict]:
        """Filtre les lignes contenant une entité donnée."""
        if etype:
            line_ids = self.entity_index.get((text.lower(), etype), set())
        else:
            line_ids = set().union(*(
                v for (t, _), v in self.entity_index.items()
                if t == text.lower()
            ))
        return [r for r in self.records if r["line_id"] in line_ids]

    def filter_by_topic(self, topic_id: int) -> list[dict]:
        """Retourne les lignes appartenant à des actes assignés au topic donné."""
        line_ids = self.topic_index.get(topic_id, set())
        return [r for r in self.records if r["line_id"] in line_ids]

    def query(self, text: str = None,
               entity: str = None, entity_type: str = None,
               topic_id: int = None) -> list[dict]:
        """
        Interface unifiée : combine les critères par intersection.

        Exemple :
            index.query(entity="Normandie", entity_type="LOC", topic_id=2)
        """
        results = set(r["line_id"] for r in self.records)

        if text:
            results &= set(r["line_id"] for r in self.search_text(text))
        if entity:
            results &= set(r["line_id"]
                           for r in self.filter_by_entity(entity, entity_type))
        if topic_id is not None:
            results &= set(r["line_id"]
                           for r in self.filter_by_topic(topic_id))

        return [r for r in self.records if r["line_id"] in results]


# Utilisation
# index = MedievalCorpusIndex(enriched_records, doc_topics)
#
# Recherche plein-texte
# results = index.search_text("sénéchal normandie")
#
# Filtre par entité et topic
# results = index.query(entity="Normandie", entity_type="LOC", topic_id=1)
# for r in results[:5]:
#     print(f"  [{r['line_id']}] {r['normalized']}")
```

---

## 7. Data contract NLP v2 : schéma complet

### 7.1 Évolution depuis la version 1

Le data contract v1, produit au Jour 3, contenait huit champs. La version 2, produite ce jour, en contient seize. Le tableau ci-dessous documente chaque nouveau champ, son type, et son origine dans le pipeline.

| Champ | Type | Origine | Version |
|---|---|---|---|
| `line_id` | str | HTR Volet 1 | v1 |
| `polygon_ref` | str | HTR Volet 1 | v1 |
| `transcription` | str | HTR Volet 1 | v1 |
| `confidence` | float | HTR Volet 1 | v1 |
| `normalized` | str | Pipeline Jour 2 | v1 |
| `ner_spans` | list[dict] | NER Jour 3 | v1 |
| `pos_tags` | list[str] | pie-extended Jour 3 | v1 |
| `lemmas` | list[str] | pie-extended Jour 3 | v1 |
| `schema_version` | str | Métadonnée pipeline | v2 |
| `pipeline_version` | str | Métadonnée pipeline | v2 |
| `split_hash` | str | Split Jour 2 | v2 |
| `topics` | list[dict] | BERTopic Jour 4 | v2 |
| `relations` | list[dict] | Extraction RE Jour 4 | v2 |
| `coref_chain` | str \| null | Coréférence Jour 4 | v2 |
| `graph_node_id` | str \| null | Graphe Jour 4 | v2 |
| `tei_ref` | str | Export TEI Jour 4 | v2 |

### 7.2 Schéma JSON complet annoté

```json
{
  "schema_version":  "2.0",
  "pipeline_version": "nlp_v1.0",
  "split_hash":       "76aff95784fd4d4e...",

  "line_id":      "charte_1346_fol12_l03",
  "polygon_ref":  "fol12_bbox_230_415_890_440",

  "transcription": "li s^r de gisors signa",
  "confidence":    0.87,
  "normalized":    "li seigneur de gisors signa",

  "ner_spans": [
    {"start": 3,  "end": 11, "label": "TITLE", "text": "seigneur"},
    {"start": 15, "end": 21, "label": "LOC",   "text": "gisors"}
  ],
  "pos_tags":  ["DET", "NOUN", "ADP", "PROPN", "VERB"],
  "lemmas":    ["le", "seigneur", "de", "gisors", "signer"],

  "topics": [
    {"topic_id": 2, "label": "juridique",  "weight": 0.72},
    {"topic_id": 5, "label": "nobiliaire", "weight": 0.18}
  ],

  "relations": [
    {
      "sujet":      "seigneur_gisors",
      "relation":   "réside_à",
      "objet":      "gisors",
      "confiance":  0.91,
      "source_line":"charte_1346_fol12_l03"
    }
  ],

  "coref_chain":   "seigneur_gisors",
  "graph_node_id": "medieval:seigneur_gisors",
  "tei_ref":       "charte_1346_fol12.xml#charte_1346_fol12_l03"
}
```

### 7.3 Écriture et validation du schéma

La validation du data contract v2 doit vérifier les invariants hérités de la v1 (longueurs de `pos_tags` et `lemmas`, cohérence des offsets de `ner_spans`) plus les nouveaux invariants de la v2.

```python
def validate_contract_v2(record: dict) -> list[str]:
    """
    Valide un enregistrement de data contract v2.

    Retourne
    --------
    list[str]  liste des erreurs détectées (vide si valide)
    """
    errors = []
    tokens = record.get("normalized", "").split()

    # Invariants v1
    if len(record.get("pos_tags", [])) != len(tokens):
        errors.append(f"pos_tags length: {len(record.get('pos_tags',[]))} != {len(tokens)}")
    if len(record.get("lemmas", [])) != len(tokens):
        errors.append(f"lemmas length: {len(record.get('lemmas',[]))} != {len(tokens)}")
    for span in record.get("ner_spans", []):
        norm = record.get("normalized", "")
        extracted = norm[span["start"]:span["end"]]
        if extracted != span["text"]:
            errors.append(f"span offset: [{span['start']}:{span['end']}]={extracted!r}")

    # Invariants v2
    if record.get("schema_version") != "2.0":
        errors.append(f"schema_version attendu '2.0'")

    for topic in record.get("topics", []):
        if not {"topic_id", "label", "weight"} <= set(topic.keys()):
            errors.append(f"topic incomplet : {topic}")
        if not (0.0 <= topic.get("weight", -1) <= 1.0):
            errors.append(f"topic weight hors [0,1] : {topic['weight']}")

    for rel in record.get("relations", []):
        if not {"sujet", "relation", "objet"} <= set(rel.keys()):
            errors.append(f"relation incomplète : {rel}")
        if rel.get("relation") not in ("réside_à","porte_titre","agit_lors_de",
                                        "signe_acte_à","appartient_à"):
            errors.append(f"relation inconnue : {rel['relation']}")

    if "tei_ref" in record and not record["tei_ref"].endswith(record["line_id"]):
        errors.append(f"tei_ref incohérent avec line_id")

    return errors
```

---

## 8. Protocole de rétroaction HTR : le fichier diff JSON

### 8.1 Structure du fichier de rétroaction

Le fichier de rétroaction HTR (`htr_feedback.json`) est l'artefact qui ferme la boucle du pipeline complet. Sa structure doit répondre à deux besoins distincts : fournir au modèle HTR des paires (image_region, transcription_corrigée) pour le réentraînement, et documenter la traçabilité de chaque correction pour la validation scientifique.

```python
def generate_htr_feedback(enriched_records: list[dict],
                            raw_records:      list[dict],
                            output_path:      str,
                            min_confidence:   float = 0.85,
                            min_cer:          float = 0.05) -> dict:
    """
    Génère le fichier de rétroaction pour le réentraînement du modèle HTR.

    Critères de sélection des corrections :
    - Confiance HTR ≥ min_confidence : la ligne est globalement bien reconnue,
      les corrections sont donc fiables (pas un artefact de mauvaise HTR).
    - CER de la correction ≥ min_cer : la correction est non triviale
      (pas une simple normalisation d'espace).
    - Correction validée : la forme normalisée est dans le DMF ou validée
      par une règle documentée dans CONVENTIONS_NLP.md.
    """
    import editdistance, re, datetime

    raw_by_id = {r["line_id"]: r for r in raw_records}
    corrections = []

    for rec in enriched_records:
        raw = raw_by_id.get(rec["line_id"])
        if not raw:
            continue

        htr_text  = raw["transcription"]
        norm_text = rec["normalized"]

        if htr_text == norm_text:
            continue

        cer = editdistance.eval(htr_text, norm_text) / max(len(norm_text), 1)

        if rec.get("confidence", 0) < min_confidence or cer < min_cer:
            continue

        # Classifier le type de correction
        if re.search(r'[~\^ñ]', htr_text):
            ctype = "abréviation"
        elif any(c in htr_text for c in ['u','v']) and cer < 0.3:
            ctype = "graphie_uv_ij"
        elif cer > 0.4:
            ctype = "erreur_majeure"
        else:
            ctype = "autre_graphie"

        corrections.append({
            "line_id":           rec["line_id"],
            "polygon_ref":       rec.get("polygon_ref", ""),
            "transcription_htr": htr_text,
            "forme_corrigee":    norm_text,
            "cer_correction":    round(cer, 4),
            "correction_type":   ctype,
            "confiance_htr":     rec.get("confidence", 0),
            "split_hash":        rec.get("split_hash", ""),
        })

    # Statistiques globales
    from collections import Counter
    type_counts = Counter(c["correction_type"] for c in corrections)
    mean_cer    = (sum(c["cer_correction"] for c in corrections)
                   / max(len(corrections), 1))

    feedback_doc = {
        "generated_at":     datetime.datetime.now().isoformat(),
        "pipeline_version": "nlp_v1.0",
        "split_hash":       enriched_records[0].get("split_hash", "") if enriched_records else "",
        "n_corrections":    len(corrections),
        "mean_cer":         round(mean_cer, 4),
        "correction_types": dict(type_counts),
        "validation_status": "pending",   # → "validated" après vérification manuelle
        "corrections":      corrections,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(feedback_doc, f, indent=2, ensure_ascii=False)

    print(f"Fichier de rétroaction HTR : {output_path}")
    print(f"  {len(corrections)} corrections ({dict(type_counts)})")
    print(f"  CER moyen de correction : {mean_cer:.4f}")
    print(f"  Status : pending — validation manuelle requise avant injection")
    return feedback_doc
```

### 8.2 Cycle de validation avant injection

Le fichier de rétroaction est produit avec le statut `"validation_status": "pending"`. Il ne doit pas être injecté dans le corpus d'entraînement HTR avant qu'un humain ait vérifié un échantillon représentatif. Le protocole de validation est le suivant :

**Étape 1 — Échantillonnage stratifié.** Tirer 20 corrections par type (abréviations, graphies u/v, erreurs majeures, autres), soit 60–80 corrections au total selon les types présents.

**Étape 2 — Vérification dans le TEI.** Pour chaque correction de l'échantillon, ouvrir le fichier TEI correspondant et vérifier que la forme corrigée est attestée dans le texte source et cohérente avec le contexte. Le `polygon_ref` permet d'afficher l'image de la ligne pour confirmation visuelle.

**Étape 3 — Mise à jour du statut.** Si le taux de corrections correctes dans l'échantillon est supérieur à 90 %, passer `"validation_status"` à `"validated"`. Sinon, documenter les types d'erreurs trouvés et affiner les règles du pipeline avant de générer un nouveau fichier.

**Étape 4 — Injection.** Les corrections validées sont converties au format attendu par le pipeline HTR (paires image_region + texte, au format PAGE-XML ou similaire selon l'outil utilisé dans le Volet 1), et injectées dans le corpus d'entraînement en respectant la séparation des splits.

---

## Bibliographie de référence

### Modélisation thématique et interprétation

Blei, D. M. (2012). **Probabilistic Topic Models**. *Communications of the ACM*, 55(4). — Version de synthèse accessible de LDA.

Chang, J., Gerrish, S., Wang, C., Boyd-Graber, J., & Blei, D. (2009). **Reading Tea Leaves: How Humans Interpret Topic Models**. *NeurIPS 2009*. — Sur la validité des évaluations humaines de topics par rapport aux métriques automatiques.

Mimno, D., Wallach, H., Talley, E., Leenders, M., & McCallum, A. (2011). **Optimizing Semantic Coherence in Topic Models**. *EMNLP 2011*. — Fondement de la cohérence UMass.

Grootendorst, M. (2022). **BERTopic: Neural topic modeling with a class-based TF-IDF procedure**. [arXiv:2203.05794](https://arxiv.org/abs/2203.05794)

### Extraction de relations et évaluation

Mintz, M., Bills, S., Snow, R., & Jurafsky, D. (2009). **Distant supervision for relation extraction without labeled data**. *ACL 2009*.

Wei, X., Cui, X., Cheng, N., Wang, X., Zhang, X., & Huang, S. (2023). **Zero-Shot Information Extraction via Chatting with ChatGPT**. [arXiv:2302.10205](https://arxiv.org/abs/2302.10205) — Sur le prompting LLM pour l'extraction d'information structurée.

### Coréférence

Lee, K., He, L., Lewis, M., & Zettlemoyer, L. (2017). **End-to-end Neural Coreference Resolution**. *EMNLP 2017*. [arXiv:1707.07045](https://arxiv.org/abs/1707.07045)

Pradhan, S., Luo, X., Recasens, M., Hovy, E., Ng, V., & Strube, M. (2014). **Scoring Coreference Partitions of Predicted Mentions: A Reference Implementation of the CoNLL 2011/2012 Scorers**. *ACL 2014*. — Métriques MUC, B³, CEAF.

### TEI et humanités numériques

Burnard, L., & Bauman, S. (eds.) (2023). **TEI P5: Guidelines for Electronic Text Encoding and Interchange**. TEI Consortium. [https://www.tei-c.org/Guidelines/P5/](https://www.tei-c.org/Guidelines/P5/)

Clavaud, F. (2019). **Initiation au TEI**. École nationale des chartes. — Introduction pratique en français.

### Interface de requête et RAG

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W.-T., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks**. *NeurIPS 2020*. [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)

### Outils

BERTopic. [GitHub: MaartenGr/BERTopic](https://github.com/MaartenGr/BERTopic)

NetworkX documentation. [https://networkx.org/documentation/stable/](https://networkx.org/documentation/stable/)

lxml. [https://lxml.de](https://lxml.de)

Coreferee (résolution de coréférence multilingue). [GitHub: msg-systems/coreferee](https://github.com/msg-systems/coreferee)

---

*Support de cours rédigé pour le Master Data/IA · Module NLP · MD5 Volet 2 · 2026. Ce document accompagne le TP autonome du Jour 4 (13h30–17h00). Les livrables attendus en fin de séance sont : le modèle BERTopic et son rapport de topics, les fichiers TEI-XML validés, le data contract NLP v2, le fichier de rétroaction HTR, le graphe de connaissances exporté en JSON-LD, et l'interface de requête.*
