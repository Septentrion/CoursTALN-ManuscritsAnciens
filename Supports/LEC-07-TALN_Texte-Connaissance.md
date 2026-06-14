# Chapitre 7 — Du texte à la connaissance structurée

**Module NLP · Master Data/IA · MD5 Volet 2 · 2026**  
Cours magistral — 3 heures

---

## Avant-propos : ce que trois jours de pipeline ont produit

À l'issue du Jour 3, vous disposez d'un corpus entièrement annoté : chaque ligne de manuscrit est normalisée, étiquetée en POS, lemmatisée, et décorée d'entités nommées. Le data contract enrichi du Chapitre 6 contient, pour chaque ligne, les champs `ner_spans`, `pos_tags`, `lemmas`, et `polygon_ref`. C'est beaucoup — mais c'est encore du texte annoté, pas de la connaissance.

Ce chapitre franchit la dernière étape du pipeline : extraire de ce texte annoté des structures exploitables par des systèmes d'information. Trois opérations distinctes y contribuent. La modélisation thématique révèle la structure latente du corpus — quels sujets y sont traités, dans quelles proportions. L'extraction de relations identifie les liens sémantiques entre entités nommées — qui agit sur qui, où, quand. La construction d'un graphe de connaissances organise ces relations dans une structure interrogeable. Et deux formats de sortie assurent la pérennité de ce travail : le TEI-XML, standard des humanités numériques depuis trente ans, et le JSON-LD, passerelle vers le web sémantique.

Le dernier point du chapitre — la boucle de rétroaction NLP → HTR — est le plus important stratégiquement : il montre que le pipeline NLP n'est pas seulement un consommateur de données HTR, mais un producteur de vérité terrain améliorée pour réentraîner le modèle HTR lui-même.

---

## 1. Modélisation thématique

### 1.1 Le problème : trouver la structure sans supervision

Un corpus de 200 chartes médiévales ne porte pas d'étiquette de thème. Personne n'a écrit *"ce document parle de fiscalité"* sur le parchemin. La modélisation thématique est une famille de méthodes non supervisées qui inférent, depuis la distribution des mots dans les documents, des **topics** — des groupes de mots qui co-occurrent fréquemment et dont la co-occurrence révèle un thème latent.

L'hypothèse fondamentale est que chaque document est un mélange de plusieurs topics, et que chaque topic est une distribution sur le vocabulaire. Un document de charte fiscale aura une forte proportion du topic "taxation" (mots *taxe*, *denier*, *levée*, *bailli*) et une proportion moindre du topic "noblesse" (mots *seigneur*, *chevalier*, *fief*). Ces topics ne sont pas définis a priori — ils émergent des données.

### 1.2 LDA : allocation de Dirichlet latente

LDA (*Latent Dirichlet Allocation*, Blei et al. 2003) est le modèle génératif fondateur de la modélisation thématique. Il suppose que chaque document $d$ a été généré par le processus suivant :

1. Tirer la distribution de topics $\theta_d \sim \text{Dir}(\alpha)$ — un vecteur de $K$ probabilités.
2. Pour chaque mot $w_n$ du document :
   a. Tirer un topic $z_n \sim \text{Multinomial}(\theta_d)$.
   b. Tirer le mot $w_n \sim \text{Multinomial}(\phi_{z_n})$, où $\phi_k$ est la distribution du topic $k$ sur le vocabulaire.

Les paramètres à inférer sont $\Phi \in \mathbb{R}^{K \times V}$ (les $K$ distributions mot-topic) et $\Theta \in \mathbb{R}^{D \times K}$ (les $D$ distributions topic-document). Sur un corpus de $D=200$ documents, $K=10$ topics, et $V=5\,000$ mots distincts :

$$|\Phi| = K \times V = 10 \times 5\,000 = 50\,000 \text{ paramètres}$$
$$|\Theta| = D \times K = 200 \times 10 = 2\,000 \text{ paramètres}$$

Les hyperparamètres $\alpha$ et $\beta$ contrôlent respectivement la concentration des distributions topic-document (petit $\alpha$ → chaque document est dominé par peu de topics) et mot-topic (petit $\beta$ → chaque topic est représenté par peu de mots).

```python
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# Corpus : textes normalisés du Jour 2
texts = [" ".join(record["lemmas"])
         for record in enriched_corpus]

# Vectorisation par sac de mots (stop words médiévaux inclus)
stop_words_med = {'le','la','les','de','du','des','en','et','à','au','pour',
                  'un','une','il','elle','se','qui','que','ce','son','sa','ses'}
vectorizer = CountVectorizer(
    max_features  = 5000,
    min_df        = 2,          # mot présent dans au moins 2 documents
    stop_words    = list(stop_words_med),
)
X = vectorizer.fit_transform(texts)   # (n_docs, vocab_size)

# Modèle LDA
lda = LatentDirichletAllocation(
    n_components     = 10,       # nombre de topics
    max_iter         = 50,
    learning_method  = "batch",
    random_state     = 42,
    doc_topic_prior  = 0.1,      # alpha : topics peu nombreux par document
    topic_word_prior = 0.01,     # beta  : mots peu nombreux par topic
)
lda.fit(X)

# Affichage des top-10 mots par topic
feature_names = vectorizer.get_feature_names_out()
for topic_id, topic in enumerate(lda.components_):
    top_words = [feature_names[i]
                 for i in topic.argsort()[:-11:-1]]
    print(f"Topic {topic_id:2d} : {' | '.join(top_words)}")
```

**Interprétation sur corpus médiéval :** trois types de topics sont typiquement identifiés sur des chartes du XIVe siècle :
- Topic **juridique** : *acte, charte, sceller, témoin, consentir, accord, stipuler*.
- Topic **fiscal** : *denier, livre, taxe, bailli, lever, rendre, compte*.
- Topic **nobiliaire** : *seigneur, chevalier, fief, hommage, vassal, duc, comte*.

Un quatrième topic, **liturgique**, apparaît dans les documents de donation à des abbayes : *abbaye, moine, dieu, âme, prière, chapelle*.

**Évaluation :** la cohérence $c_v$ mesure la qualité des topics en corrélant la co-occurrence des mots de chaque topic dans un corpus de référence externe. Une cohérence supérieure à 0.5 est acceptable ; 0.6–0.7 est bon.

### 1.3 NMF : une alternative matricielle

NMF (*Non-negative Matrix Factorization*) décompose la matrice TF-IDF du corpus en deux matrices non négatives :

$$X \approx W \cdot H, \quad W \in \mathbb{R}^{D \times K}_{\geq 0}, \quad H \in \mathbb{R}^{K \times V}_{\geq 0}$$

$H$ contient les $K$ topics (distributions mot-topic), $W$ les poids de chaque topic pour chaque document. NMF est souvent plus rapide que LDA sur des petits corpus et produit des topics plus "purs" (moins de mots partagés entre topics), au prix d'une moins bonne base probabiliste.

```python
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vec = TfidfVectorizer(max_features=5000, min_df=2,
                             stop_words=list(stop_words_med))
X_tfidf   = tfidf_vec.fit_transform(texts)

nmf = NMF(n_components=10, random_state=42, max_iter=400)
nmf.fit(X_tfidf)
```

### 1.4 BERTopic : modélisation thématique neurale

BERTopic (Grootendorst 2022) remplace la représentation bag-of-words par des embeddings contextuels. Le pipeline est le suivant :

**Étape 1 — Embeddings SBERT :** chaque document est encodé par un modèle Sentence-BERT (ou CamemBERT pour le français) en un vecteur de dimension 768.

**Étape 2 — Réduction de dimension UMAP :** les embeddings 768D sont projetés dans un espace de faible dimension (typiquement 5D) par UMAP (*Uniform Manifold Approximation and Projection*). UMAP préserve mieux la structure locale des données que PCA, ce qui produit des clusters plus séparés.

**Étape 3 — Clustering HDBSCAN :** les documents dans l'espace réduit sont groupés par HDBSCAN (*Hierarchical Density-Based Spatial Clustering of Applications with Noise*). Contrairement à K-Means, HDBSCAN ne requiert pas de spécifier le nombre de clusters à l'avance, et il identifie les documents ne correspondant à aucun topic (label −1, bruit).

**Étape 4 — c-TF-IDF :** pour chaque cluster, BERTopic concatène les textes et calcule un c-TF-IDF (*class-based TF-IDF*) pour identifier les mots représentatifs du topic. Le c-TF-IDF d'un mot $w$ dans le topic $t$ est :

$$\text{c-TF-IDF}(w, t) = \frac{f(w,t)}{A_t} \times \log\!\left(1 + \frac{m}{\sum_{k} f(w,k)}\right)$$

où $f(w,t)$ est la fréquence du mot $w$ dans le topic $t$, $A_t$ le nombre moyen de mots par document du topic, et $m$ le nombre total de documents.

```python
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN

# Modèle d'embeddings français
embedding_model = SentenceTransformer("camembert-base")

# Configuration pour corpus court (~200 documents)
umap_model   = UMAP(n_neighbors=10, n_components=5,
                     min_dist=0.0,  metric='cosine', random_state=42)
hdbscan_model = HDBSCAN(min_cluster_size=5, min_samples=3,
                         metric='euclidean', prediction_data=True)

topic_model = BERTopic(
    embedding_model = embedding_model,
    umap_model      = umap_model,
    hdbscan_model   = hdbscan_model,
    language        = "french",
    calculate_probabilities = True,
    verbose         = True,
)

topics, probs = topic_model.fit_transform(texts)

# Résultats
topic_info = topic_model.get_topic_info()
print(topic_info[["Topic","Count","Name","Representation"]])
```

**Pourquoi BERTopic sur un corpus court ?** LDA et NMF sont efficaces quand les documents sont longs et nombreux. Sur un corpus de 200 chartes médiévales de 80 tokens en moyenne, les sacs de mots sont trop creux pour que les co-occurrences soient significatives. BERTopic contourne ce problème : les embeddings SBERT capturent la sémantique même des phrases courtes, et la projection UMAP résout le problème de la malédiction de la dimensionnalité.

**Limitation :** HDBSCAN peut attribuer jusqu'à 30–40 % des documents au bruit (label −1) sur un corpus hétérogène. Ce n'est pas un défaut mais une information : ces documents ne rentrent dans aucun topic dominant, ce qui est souvent vrai pour les documents de contenu mixte.

---

## 2. Extraction de relations

### 2.1 Du triplet à la relation

L'extraction de relations (*Relation Extraction*, RE) identifie les liens sémantiques entre entités nommées. À partir du texte *"le sénéchal Jean de Normandie porta les lettres au roi"*, on peut extraire le triplet :

```
(Jean de Normandie, porte_titre, sénéchal)
(Jean de Normandie, effectue_action, porter les lettres)
(Jean de Normandie, destinataire, roi)
```

Chaque triplet est de la forme $(s, r, o)$ — sujet, relation, objet. L'ensemble de ces triplets constitue les arêtes du graphe de connaissances.

### 2.2 Approche classique par règles et patrons syntaxiques

L'approche la plus simple et la plus contrôlable utilise des patrons syntaxiques sur l'annotation UD produite par Stanza. Par exemple, la relation `porte_titre` peut être extraite par le patron :

```
TITLE PROPN+ ADP PROPN+ → (PROPN+, porte_titre, TITLE)
```

Ce patron correspond à une séquence où un TITLE est suivi d'un nom propre composé (*"sénéchal Jean de Normandie"*), ce qui indique que le titre appartient à la personne.

```python
from itertools import groupby

def extract_title_person_relations(sentence: list[dict]) -> list[tuple]:
    """
    Extrait les relations (personne, porte_titre, titre) depuis les
    annotations UD d'une phrase.

    Heuristique : un TITLE immédiatement suivi d'un span PROPN
    indique une relation de titre-porteur.
    """
    relations = []
    i = 0
    while i < len(sentence):
        tok = sentence[i]
        if tok.get('upos') == 'NOUN' and tok.get('ner', '').startswith('B-TITLE'):
            title = tok['form']
            # Collecter le span de personne qui suit
            j = i + 1
            person_tokens = []
            while j < len(sentence) and sentence[j].get('ner', '').startswith(('B-PER','I-PER')):
                person_tokens.append(sentence[j]['form'])
                j += 1
            if person_tokens:
                person = ' '.join(person_tokens)
                relations.append((person, 'porte_titre', title))
            i = j
        else:
            i += 1
    return relations
```

### 2.3 Extraction par prompting LLM avec schéma

L'approche moderne consiste à fournir à un LLM un schéma de relations cibles et à lui demander d'extraire les triplets correspondants. L'avantage par rapport aux règles est la généralisation : un LLM peut reconnaître *"Jean, sénéchal de Normandie"* et *"Messire Jean de Gisors, qui tenait le titre de bailli"* comme deux manifestations de la même relation `porte_titre`, même si les structures syntaxiques diffèrent.

```python
# Schéma de relations cibles pour le corpus médiéval
RELATION_SCHEMA = {
    "porte_titre":     "Une personne (PER) porte un titre ou une fonction (TITLE)",
    "réside_à":        "Une personne (PER) réside ou est associée à un lieu (LOC)",
    "agit_lors_de":    "Une action se produit à une date (DATE)",
    "appartient_à":    "Une personne ou un lieu appartient à une organisation (ORG)",
    "signe_acte_avec": "Deux personnes (PER) co-signent un acte",
}

def extract_relations_llm(text: str,
                           schema: dict,
                           model_fn: callable) -> list[dict]:
    """
    Extrait les relations via prompting LLM avec schéma structuré.

    Retourne une liste de dicts :
        {"sujet": str, "relation": str, "objet": str, "confiance": float}
    """
    schema_str = "\n".join(
        f"- {rel}: {desc}" for rel, desc in schema.items()
    )
    prompt = f"""Extrait les relations du texte médiéval suivant.
Retourne UNIQUEMENT un JSON valide, sans texte autour.

Relations cibles :
{schema_str}

Texte : "{text}"

Format de réponse :
[
  {{"sujet": "...", "relation": "...", "objet": "...", "confiance": 0.0-1.0}},
  ...
]"""
    raw = model_fn(prompt)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return []
```

**Validation manuelle :** les triplets extraits par LLM nécessitent une validation humaine sur un échantillon. La précision typique d'un LLM bien guidé par schéma sur des textes médiévaux normalisés est de 75–85 %. La validation manuelle sur 50 triplets représentatifs suffit pour estimer la précision globale.

### 2.4 Résolution de coréférence

La résolution de coréférence identifie les chaînes de référence : *"le roi"*, *"Philippe"*, *"il"*, *"ledit seigneur"* peuvent tous référer au même individu dans un acte. Sans résolution de coréférence, le graphe de connaissances aura plusieurs nœuds pour la même entité.

Trois niveaux d'approche existent selon la complexité et les ressources disponibles :

**Règles de base :** pour les textes médiévaux, les règles simples couvrent une fraction significative des cas. *"Ledit seigneur"* et *"ledit chevalier"* coréfèrent à la dernière entité TITLE mentionnée. *"Il"* (cas sujet) coréfère au dernier sujet masculin singulier.

**SpanBERT :** SpanBERT (Joshi et al. 2020) est un modèle pré-entraîné pour la résolution de coréférence. Il encode des spans de texte et prédit quels spans coréfèrent. Il est disponible en français via le modèle `coreferee` (Hugging Face) mais n'a pas été entraîné spécifiquement sur du moyen français.

**LLM par prompting :** les LLM modernes (GPT-4, Claude) résolvent bien la coréférence par prompting pour les textes courts (moins de 512 tokens). Pour les textes plus longs, une fenêtre glissante avec chevauchement est nécessaire.

```python
def resolve_coreference_rules(tokens:    list[str],
                               ner_spans: list[dict]) -> dict[int, str]:
    """
    Résolution de coréférence par règles pour le moyen français.

    Règles implémentées :
    1. 'ledit/ladite' + TITLE/NOUN → coréfère à la dernière entité du type correspondant.
    2. Pronom 'il/elle/lui' → coréfère au dernier NOM/PROPN de genre correspondant.

    Retourne
    --------
    dict {position_token: canonical_entity_id}
    """
    coref_map   = {}
    last_per    = None
    last_title  = None
    last_loc    = None

    span_by_start = {s['start']: s for s in ner_spans}

    char_pos = 0
    for i, tok in enumerate(tokens):
        if tok.lower() in ('ledit', 'ladite', 'ledits', 'lesdites'):
            if last_per:
                coref_map[i] = last_per['text']
        # Tracking des dernières entités rencontrées
        if char_pos in span_by_start:
            span = span_by_start[char_pos]
            if span['label'] == 'PER':
                last_per   = span
            elif span['label'] == 'TITLE':
                last_title = span
            elif span['label'] == 'LOC':
                last_loc   = span
        char_pos += len(tok) + 1

    return coref_map
```

---

## 3. Construction du graphe de connaissances

### 3.1 Structure du graphe

Un graphe de connaissances est un graphe orienté labellisé $G = (V, E)$ où :
- Les nœuds $V$ représentent des entités (personnes, lieux, dates, organisations, titres).
- Les arêtes $E$ représentent des relations typées entre entités.

Chaque arête est un triplet $(s, r, o)$ — sujet, relation, objet — directement issu des étapes d'extraction précédentes. En Python, NetworkX est la bibliothèque standard pour manipuler ces graphes.

```python
import networkx as nx

def build_knowledge_graph(enriched_records: list[dict],
                           relations:        list[dict]) -> nx.DiGraph:
    """
    Construit un graphe de connaissances depuis les enregistrements
    enrichis et les relations extraites.
    """
    G = nx.DiGraph()

    # Ajouter les entités comme nœuds
    for record in enriched_records:
        for span in record.get('ner_spans', []):
            node_id = span['text'].lower().replace(' ', '_')
            if not G.has_node(node_id):
                G.add_node(node_id,
                            label     = span['text'],
                            type      = span['label'],
                            source    = record['line_id'],
                            polygon   = record.get('polygon_ref', ''))

    # Ajouter les relations comme arêtes
    for rel in relations:
        s_id = rel['sujet'].lower().replace(' ', '_')
        o_id = rel['objet'].lower().replace(' ', '_')
        # Créer les nœuds si absents (entités implicites)
        for nid in (s_id, o_id):
            if not G.has_node(nid):
                G.add_node(nid, label=nid, type='UNKNOWN')
        G.add_edge(s_id, o_id,
                   relation   = rel['relation'],
                   confiance  = rel.get('confiance', 1.0),
                   source     = rel.get('source_line', ''))

    return G

# Statistiques du graphe
def describe_graph(G: nx.DiGraph) -> None:
    print(f"Nœuds       : {G.number_of_nodes()}")
    print(f"Arêtes      : {G.number_of_edges()}")
    print(f"Densité     : {nx.density(G):.4f}")
    # Distribution par type de nœud
    from collections import Counter
    type_counts = Counter(data['type']
                          for _, data in G.nodes(data=True))
    for node_type, count in type_counts.most_common():
        print(f"  {node_type:8s} : {count} nœuds")
    # Hubs : nœuds à fort degré entrant
    in_deg = sorted(G.in_degree(), key=lambda x: -x[1])[:5]
    print(f"Top-5 hubs  : {[(G.nodes[n]['label'], d) for n,d in in_deg]}")
```

**Lien avec le Volet 1 :** chaque nœud du graphe porte un attribut `polygon` reprenant le `polygon_ref` du data contract. Ce lien permet, depuis n'importe quelle requête sur le graphe, de remonter à la position physique de l'entité dans l'image du manuscrit. C'est la condition nécessaire pour que les humanistes puissent vérifier chaque triplet dans le document source.

### 3.2 Stockage : RDF/Turtle, Neo4j, JSON-LD

Trois technologies de stockage couvrent les besoins différents des projets de graphes de connaissances.

**RDF/Turtle** est le format du web sémantique W3C. Chaque triplet $(s, r, o)$ est encodé avec des URIs pour les nœuds et les relations :

```turtle
@prefix medieval: <http://example.org/medieval#> .
@prefix schema:   <http://schema.org/> .

medieval:jean_normandie a schema:Person ;
    schema:name          "Jean de Normandie" ;
    medieval:porteTitre  medieval:seneschal ;
    medieval:residesAt   medieval:normandie ;
    schema:description   "Sénéchal mentionné dans charte 1346" .

medieval:seneschal a medieval:Title ;
    schema:name "sénéchal" .

medieval:normandie a schema:Place ;
    schema:name "Normandie" .
```

RDF/Turtle est le format recommandé pour les projets d'interopérabilité et d'archivage à long terme.

**Neo4j** est une base de données graphe native avec son langage de requête Cypher. Elle est adaptée aux projets nécessitant des requêtes complexes en temps réel (traversées de graphe, recommandation de relations, détection de communautés). Son inconvénient est l'absence de standard d'interopérabilité natif.

**JSON-LD** (*JavaScript Object Notation for Linked Data*) est la représentation la plus compacte et la plus accessible pour les développeurs. Il étend JSON avec un `@context` qui mappe les clés vers des URIs :

```python
import json

def graph_to_jsonld(G: nx.DiGraph,
                    context: dict = None) -> dict:
    """
    Sérialise le graphe de connaissances en JSON-LD.
    """
    if context is None:
        context = {
            "@vocab":    "http://schema.org/",
            "medieval":  "http://example.org/medieval#",
            "tei":       "http://www.tei-c.org/ns/1.0/",
            "porteTitre":   "medieval:porteTitre",
            "residesAt":    "medieval:residesAt",
            "agitLorsde":   "medieval:agitLorsDe",
            "polygon_ref":  "medieval:polygonRef",
        }
    entities = []
    for node_id, data in G.nodes(data=True):
        ent = {
            "@id":   f"medieval:{node_id}",
            "@type": data.get('type', 'Thing'),
            "name":  data.get('label', node_id),
        }
        if data.get('polygon'):
            ent["polygon_ref"] = data['polygon']
        # Relations sortantes
        out_edges = G.out_edges(node_id, data=True)
        for _, target, edge_data in out_edges:
            rel  = edge_data.get('relation', 'relatedTo')
            ent[rel] = {"@id": f"medieval:{target}"}
        entities.append(ent)

    return {"@context": context, "@graph": entities}

# Sauvegarde
jsonld_doc = graph_to_jsonld(G)
with open("knowledge_graph.jsonld", "w", encoding="utf-8") as f:
    json.dump(jsonld_doc, f, indent=2, ensure_ascii=False)
```

**Choix pratique :** pour un projet de recherche en humanités numériques, la combinaison JSON-LD (échange et archivage) + Neo4j (exploration interactive) est souvent optimale. RDF/Turtle est préféré quand l'interopérabilité avec d'autres bases du web sémantique (Wikidata, DBpedia) est un objectif explicite.

### 3.3 Interrogation : SPARQL, Cypher, RAG

**SPARQL** est le langage de requête standard pour les graphes RDF. Une requête SPARQL typique sur le graphe médiéval :

```sparql
PREFIX medieval: <http://example.org/medieval#>
PREFIX schema:   <http://schema.org/>

# Toutes les personnes portant le titre de sénéchal
# avec leur lieu d'association et la date de l'acte

SELECT ?personne ?lieu ?date
WHERE {
    ?personne a schema:Person ;
              medieval:porteTitre medieval:seneschal ;
              medieval:residesAt  ?lieu .
    OPTIONAL {
        ?personne medieval:agitLorsDe ?date .
    }
}
ORDER BY ?personne
```

**Cypher** (Neo4j) est plus expressif pour les traversées multi-niveaux :

```cypher
// Trouver tous les sénéchaux et leurs relations directes
MATCH (p:PER)-[:porteTitre]->(t:TITLE {name: "sénéchal"})
OPTIONAL MATCH (p)-[:residesAt]->(l:LOC)
OPTIONAL MATCH (p)-[:agitLorsDe]->(d:DATE)
RETURN p.name, t.name, l.name, d.name
ORDER BY p.name
```

**RAG (*Retrieval-Augmented Generation*) :** une interface de requête en langage naturel peut être construite en combinant un index vectoriel (embeddings des nœuds et textes des chartes) avec un LLM. L'utilisateur pose une question en français moderne (*"Quels sénéchaux ont signé des actes en Normandie ?"*), le RAG récupère les entités et relations pertinentes, et le LLM formule la réponse.

```python
# Interface RAG minimale avec NetworkX + LLM
def query_graph_rag(question: str,
                    G:        nx.DiGraph,
                    model_fn: callable) -> str:
    """
    Interface RAG pour le graphe de connaissances.

    1. Extraire les entités de la question via le gazetier.
    2. Récupérer les sous-graphes locaux autour de ces entités.
    3. Formater les triplets comme contexte pour le LLM.
    4. Générer la réponse.
    """
    # Extraction d'entités clés depuis la question
    question_lower = question.lower()
    relevant_nodes = [
        (n, d) for n, d in G.nodes(data=True)
        if d.get('label', '').lower() in question_lower
    ]

    # Sous-graphe local (voisinage à 2 sauts)
    context_triplets = []
    for node_id, _ in relevant_nodes:
        for _, target, data in G.out_edges(node_id, data=True):
            t_label = G.nodes[target].get('label', target)
            context_triplets.append(
                f"({G.nodes[node_id]['label']}, {data['relation']}, {t_label})"
            )

    context = "\n".join(context_triplets[:20])   # max 20 triplets
    prompt  = f"""Réponds à la question en t'appuyant uniquement sur les triplets fournis.

Triplets :
{context}

Question : {question}
Réponse :"""

    return model_fn(prompt)
```

---

## 4. Export TEI-XML

### 4.1 Le standard TEI en humanités numériques

TEI (*Text Encoding Initiative*) est le standard d'encodage XML des textes en sciences humaines depuis 1987. Il définit une centaine de balises pour représenter la structure physique (pages, lignes, colonnes), la structure logique (paragraphes, sections, listes), et les annotations linguistiques (POS, lemmes, entités nommées). Dans le contexte d'un projet de manuscrits médiévaux numériques, le TEI est la forme de publication scientifique attendue.

Un document TEI minimal a la structure suivante :

```xml
<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt>
        <title>Charte de Gisors, mars 1346</title>
      </titleStmt>
      <publicationStmt>
        <p>Projet CREMMA — Volet 2 NLP 2026</p>
      </publicationStmt>
      <sourceDesc>
        <msDesc>
          <msIdentifier>
            <repository>Archives départementales de l'Eure</repository>
            <idno>1J 234</idno>
          </msIdentifier>
        </msDesc>
      </sourceDesc>
    </fileDesc>
  </teiHeader>
  <text>
    <body>
      <div type="charte">
        <!-- Contenu annoté ici -->
      </div>
    </body>
  </text>
</TEI>
```

### 4.2 Encodage des annotations NLP en TEI

Les balises TEI pertinentes pour vos annotations :

| Annotation | Balise TEI | Attributs principaux |
|---|---|---|
| Personne (PER) | `<persName>` | `@key` (identifiant dans le graphe) |
| Lieu (LOC) | `<placeName>` | `@key`, `@type` |
| Date (DATE) | `<date>` | `@when` (ISO 8601) |
| Organisation (ORG) | `<orgName>` | `@key` |
| Titre (TITLE) | `<roleName>` | `@type="nobility"` |
| Token POS+lemme | `<w>` | `@lemma`, `@pos` |
| Ligne manuscrite | `<lb/>` | `@n`, `@facs` (polygon_ref) |

La balise `@facs` (facsimile) de `<lb/>` est le point d'ancrage vers l'image du manuscrit — elle reprend le `polygon_ref` du data contract Volet 1, réalisant ainsi la connexion entre l'annotation linguistique et la position physique sur le parchemin.

```python
from lxml import etree

TEI_NS = "http://www.tei-c.org/ns/1.0"
XML_NS = "http://www.w3.org/XML/1998/namespace"

def record_to_tei_line(record: dict,
                        parent: etree.Element) -> None:
    """
    Ajoute une ligne annotée en TEI à un élément parent.

    Encode les entités nommées, les tokens POS/lemme, et
    l'ancrage spatial via @facs.
    """
    # Balise de ligne avec ancrage spatial
    lb = etree.SubElement(parent, f"{{{TEI_NS}}}lb",
                           n    = record['line_id'],
                           facs = record.get('polygon_ref', ''))

    tokens   = record['normalized'].split()
    pos_tags = record.get('pos_tags', [])
    lemmas   = record.get('lemmas', [])

    # Index des spans NER par position de token
    span_starts = {}
    span_ends   = {}
    for span in record.get('ner_spans', []):
        # Convertir offsets caractères → indices de tokens
        char = 0
        for i, tok in enumerate(tokens):
            if char == span['start']:
                span_starts[i] = span
            char += len(tok) + 1
            if char - 1 == span['end']:
                span_ends[i] = span

    i = 0
    current_entity_elem = None

    while i < len(tokens):
        tok   = tokens[i]
        pos   = pos_tags[i] if i < len(pos_tags) else '_'
        lemma = lemmas[i]   if i < len(lemmas)   else tok

        # Ouvrir une balise d'entité si nécessaire
        if i in span_starts:
            span     = span_starts[i]
            ner_tag  = {
                'PER':   f'{{{TEI_NS}}}persName',
                'LOC':   f'{{{TEI_NS}}}placeName',
                'DATE':  f'{{{TEI_NS}}}date',
                'ORG':   f'{{{TEI_NS}}}orgName',
                'TITLE': f'{{{TEI_NS}}}roleName',
            }.get(span['label'], f'{{{TEI_NS}}}name')

            ner_key  = span['text'].lower().replace(' ', '_')
            attrib   = {'key': ner_key}
            if span['label'] == 'TITLE':
                attrib['type'] = 'nobility'

            current_entity_elem = etree.SubElement(parent, ner_tag, **attrib)

        # Ajouter le token avec POS et lemme
        target = current_entity_elem if current_entity_elem else parent
        w_elem = etree.SubElement(target, f'{{{TEI_NS}}}w',
                                   lemma = lemma,
                                   pos   = pos)
        w_elem.text = tok

        # Fermer la balise d'entité si c'est le dernier token
        if i in span_ends:
            current_entity_elem = None

        i += 1


def corpus_to_tei(records: list[dict], output_path: str) -> None:
    """
    Exporte le corpus enrichi en un fichier TEI-XML valide.
    """
    root = etree.Element(f'{{{TEI_NS}}}TEI',
                          nsmap={None: TEI_NS})

    # teiHeader
    header  = etree.SubElement(root, f'{{{TEI_NS}}}teiHeader')
    fileDesc = etree.SubElement(header, f'{{{TEI_NS}}}fileDesc')
    titleStmt = etree.SubElement(fileDesc, f'{{{TEI_NS}}}titleStmt')
    title_el  = etree.SubElement(titleStmt, f'{{{TEI_NS}}}title')
    title_el.text = "Corpus médiéval annoté — Projet NLP MD5 2026"

    # body
    text_el = etree.SubElement(root, f'{{{TEI_NS}}}text')
    body    = etree.SubElement(text_el, f'{{{TEI_NS}}}body')
    div     = etree.SubElement(body, f'{{{TEI_NS}}}div', type="corpus")

    for record in records:
        p_elem = etree.SubElement(div, f'{{{TEI_NS}}}p',
                                   n=record['line_id'])
        record_to_tei_line(record, p_elem)

    tree = etree.ElementTree(root)
    tree.write(output_path,
               pretty_print    = True,
               xml_declaration = True,
               encoding        = "UTF-8")
    print(f"TEI exporté : {output_path}")
    print(f"  Validation : xmllint --schema tei_all.rng {output_path}")
```

**Validation :** la conformité TEI est vérifiable avec `xmllint` et le schéma RelaxNG de TEI All :

```bash
# Télécharger le schéma TEI All (une fois)
wget https://www.tei-c.org/Vault/P5/current/xml/tei/custom/schema/relaxng/tei_all.rng

# Valider le fichier produit
xmllint --relaxng tei_all.rng --noout corpus_medieval.xml
# Sortie attendue : corpus_medieval.xml validates
```

---

## 5. La boucle de rétroaction NLP → HTR

### 5.1 Le principe

Le pipeline que vous avez construit jusqu'ici va de l'image au graphe de connaissances : HTR → normalisation → NER → graphe. Mais il existe un chemin de retour que la plupart des projets ignorent : les corrections NLP constituent une nouvelle vérité terrain pour le modèle HTR.

Voici le mécanisme. Le pipeline de normalisation du Jour 2 a produit, pour chaque transcription HTR brute, une forme normalisée validée. Par exemple :

```
Transcription HTR  : "norm~die" (CER = 0.22 par rapport à "normandie")
Forme normalisée   : "normandie"
```

Cette paire (transcription HTR + forme corrigée) est exactement le type de données d'entraînement dont le modèle HTR a besoin pour améliorer son comportement sur les abréviations. En l'injectant dans le corpus d'entraînement du modèle HTR, on ferme la boucle.

### 5.2 Protocole de réinjection

```python
import json
from pathlib import Path

def build_htr_feedback(enriched_records: list[dict],
                        raw_records:      list[dict],
                        min_confidence:   float = 0.85,
                        min_correction:   float = 0.05) -> list[dict]:
    """
    Construit le fichier delta de corrections pour le modèle HTR.

    Ne conserve que les paires où :
    1. La confiance HTR est suffisante (ligne bien reconnue globalement).
    2. Il y a effectivement une correction (transcription != normalisée).
    3. La correction est non triviale (CER > min_correction).

    Retourne
    --------
    list[dict] avec pour chaque paire :
        {"line_id", "transcription_htr", "forme_corrigee",
         "correction_type", "polygon_ref"}
    """
    import editdistance

    feedback = []
    raw_by_id = {r['line_id']: r for r in raw_records}

    for record in enriched_records:
        raw = raw_by_id.get(record['line_id'])
        if raw is None:
            continue

        htr_text  = raw['transcription']
        norm_text = record['normalized']

        if htr_text == norm_text:
            continue   # pas de correction

        # CER de la correction
        cer = editdistance.eval(htr_text, norm_text) / max(len(norm_text), 1)

        if (record.get('confidence', 0) >= min_confidence
                and cer >= min_correction):
            # Classifier le type de correction
            if '~' in htr_text or '^' in htr_text or 'ñ' in htr_text:
                correction_type = "abréviation"
            elif any(c in htr_text for c in 'uv') and 'v' not in norm_text:
                correction_type = "graphie_uv"
            else:
                correction_type = "autre"

            feedback.append({
                "line_id":           record['line_id'],
                "transcription_htr": htr_text,
                "forme_corrigee":    norm_text,
                "cer_correction":    round(cer, 4),
                "correction_type":   correction_type,
                "polygon_ref":       record.get('polygon_ref', ''),
            })

    return feedback


def write_htr_feedback(feedback: list[dict],
                        output_path: str) -> None:
    """Sauvegarde le fichier de rétroaction HTR en JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "n_corrections":      len(feedback),
            "correction_types":   {
                ct: sum(1 for r in feedback if r['correction_type'] == ct)
                for ct in set(r['correction_type'] for r in feedback)
            },
            "corrections": feedback,
        }, f, indent=2, ensure_ascii=False)
    print(f"Fichier rétroaction HTR : {output_path}")
    print(f"  {len(feedback)} corrections documentées")
```

### 5.3 Risques de biais dans la boucle de rétroaction

La boucle de rétroaction est puissante mais introduit un risque sérieux : le **biais de confirmation**. Si le pipeline NLP normalise systématiquement une forme fausse en une autre forme fausse (par exemple, une règle incorrecte dans `apply_grapheme_rules`), l'injection de ces corrections dans les données d'entraînement HTR va ancrer cette erreur dans le nouveau modèle. Le modèle HTR réentraîné produira davantage cette forme erronée, qui sera de nouveau normalisée de la même façon — une boucle de dégradation auto-entretenue.

**Mitigation :** trois garde-fous sont nécessaires.

Premièrement, une validation humaine sur un échantillon aléatoire des corrections avant injection (10 % des corrections, seuil minimal). Cette étape est documentée dans le protocole et son résultat consigné dans le journal d'expériences.

Deuxièmement, le `split_hash` du Chapitre 4 garantit que les données de test du modèle HTR restent séparées des données de rétroaction. Il est interdit d'injecter des corrections produites depuis les lignes du jeu de test HTR.

Troisièmement, la comparaison CER avant/après réentraînement sur un jeu de test indépendant est la seule validation objective. Si le CER sur le jeu de test ne s'améliore pas, les corrections n'ont pas été bénéfiques, même si les métriques de normalisation NLP s'améliorent.

---

## 6. Synthèse : le data contract NLP final

### 6.1 Structure complète

Le data contract NLP final du Volet 2 est l'artefact de sortie qui rassemble tous les enrichissements produits depuis le Jour 1 :

```json
{
  "schema_version":  "2.0",
  "pipeline_version": "nlp_v1.0",
  "split_hash":      "76aff95784fd4d4e...",

  "line_id":         "charte_1346_fol12_l03",
  "polygon_ref":     "fol12_bbox_230_415_890_440",

  "transcription":   "li s^r de gisors signa",
  "confidence":      0.87,
  "normalized":      "li seigneur de gisors signa",

  "ner_spans": [
    {"start": 3,  "end": 11, "label": "TITLE", "text": "seigneur"},
    {"start": 15, "end": 21, "label": "LOC",   "text": "gisors"}
  ],
  "pos_tags":  ["DET","NOUN","ADP","PROPN","VERB"],
  "lemmas":    ["le","seigneur","de","gisors","signer"],

  "topics": [
    {"topic_id": 2, "label": "juridique", "weight": 0.72},
    {"topic_id": 5, "label": "nobiliaire","weight": 0.18}
  ],

  "relations": [
    {"sujet": "seigneur_gisors", "relation": "réside_à",
     "objet": "gisors",         "confiance": 0.91}
  ],

  "coref_chain": "seigneur_gisors",
  "graph_node_id": "medieval:seigneur_gisors",
  "tei_ref":       "corpus_medieval.xml#charte_1346_fol12_l03"
}
```

### 6.2 Ce que ce contrat rend possible

Avec ce data contract, un historien peut :

Poser une question en SPARQL ou Cypher (*"quels seigneurs ont signé des actes à Gisors entre 1340 et 1360 ?"*) et obtenir une réponse avec références précises aux documents.

Suivre n'importe quel résultat jusqu'à l'image source via `polygon_ref` — la réponse n'est jamais déconnectée du parchemin original.

Évaluer la fiabilité de chaque annotation via le champ `confidence` (HTR) et les scores de confiance des relations extraites.

Réutiliser les données dans n'importe quel outil compatible TEI ou JSON-LD — les formats sont ouverts et standards.

---

## Bibliographie de référence

### Modélisation thématique

Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). **Latent Dirichlet Allocation**. *Journal of Machine Learning Research*, 3, 993–1022. — Article fondateur de LDA.

Lee, D. D., & Seung, H. S. (1999). **Learning the parts of objects by non-negative matrix factorization**. *Nature*, 401, 788–791. — Fondement de NMF.

Grootendorst, M. (2022). **BERTopic: Neural topic modeling with a class-based TF-IDF procedure**. [arXiv:2203.05794](https://arxiv.org/abs/2203.05794)

Röder, M., Both, A., & Hinneburg, A. (2015). **Exploring the Space of Topic Coherence Measures**. *WSDM 2015*. — Référence sur la cohérence c_v.

McInnes, L., Healy, J., & Melville, J. (2018). **UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction**. [arXiv:1802.03426](https://arxiv.org/abs/1802.03426)

Campello, R., Moulavi, D., & Sander, J. (2013). **Density-Based Clustering Based on Hierarchical Density Estimates**. *PAKDD 2013*. — Fondement de HDBSCAN.

### Extraction de relations et coréférence

Joshi, M., Chen, D., Liu, Y., Weld, D. S., Zettlemoyer, L., & Levy, O. (2020). **SpanBERT: Improving Pre-training by Representing and Predicting Spans**. *TACL*. [arXiv:1907.10529](https://arxiv.org/abs/1907.10529)

Mintz, M., Bills, S., Snow, R., & Jurafsky, D. (2009). **Distant supervision for relation extraction without labeled data**. *ACL 2009*. — Sur la supervision distante pour la RE.

### Graphes de connaissances et RDF

Bizer, C., Heath, T., & Berners-Lee, T. (2009). **Linked Data — The Story So Far**. *International Journal on Semantic Web and Information Systems*, 5(3). — Introduction au web sémantique lié.

Harris, S., & Seaborne, A. (2013). **SPARQL 1.1 Query Language**. W3C Recommendation. — Spécification officielle SPARQL.

Sporny, M., Longley, D., Kellogg, G., Lanthaler, M., & Lindström, N. (2014). **JSON-LD 1.0**. W3C Recommendation. — Spécification officielle JSON-LD.

### TEI et humanités numériques

TEI Consortium (2023). **TEI P5: Guidelines for Electronic Text Encoding and Interchange**. [https://www.tei-c.org/Guidelines/P5/](https://www.tei-c.org/Guidelines/P5/) — Documentation officielle TEI.

Burnard, L. (2014). **What is the Text Encoding Initiative? How to add intelligent markup to digital resources**. OpenEdition Press. — Introduction pratique au TEI.

Romary, L., & Witt, A. (2014). **TEI and Language Resources**. *Proceedings of LREC 2014*.

### Boucle de rétroaction et apprentissage actif

Settles, B. (2012). **Active Learning**. *Synthesis Lectures on Artificial Intelligence and Machine Learning*, 6(1). — Fondement de l'apprentissage actif applicable aux boucles HTR→NLP.

Kiela, D., Bartolo, M., Nie, Y., Kaushik, D., Geiger, A., Wu, Z., Vidgen, B., Vidgen, B., & Williams, A. (2021). **Dynabench: Rethinking Benchmarking in NLP**. *NAACL 2021*. — Sur la création dynamique de données d'entraînement par boucle modèle-humain.

### Outils

BERTopic. [GitHub: MaartenGr/BERTopic](https://github.com/MaartenGr/BERTopic)

NetworkX. [https://networkx.org](https://networkx.org)

lxml. [https://lxml.de](https://lxml.de) — Génération et validation XML en Python.

json-ld.org. [https://json-ld.org](https://json-ld.org)

---

*Support de cours rédigé pour le Master Data/IA · Module NLP · MD5 Volet 2 · 2026. Ce document accompagne le cours magistral du Jour 4 (09h00–12h00). Il est le prérequis théorique du TP autonome Chapitre 8 (construction de la base de connaissances médiévale interrogeable) de l'après-midi.*
