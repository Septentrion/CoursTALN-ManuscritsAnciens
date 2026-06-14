# Chapitre 5 — Segmentation en mots, étiquetage morpho-syntaxique et reconnaissance d'entités nommées pour les langues historiques

**Module NLP · Master Data/IA · MD5 Volet 2 · 2026**  
Cours magistral et atelier — 3 heures

---

## Avant-propos : où en sommes-nous dans le pipeline

Les Chapitres 1 à 4 vous ont conduit d'une transcription HTR brute à un texte normalisé. Vous savez maintenant que *auoir* s'écrit *avoir*, que *norm~die* se développe en *normandie*, et que *preudomme* est lemmatisé en *prudhomme* via le DMF. Le texte est lisible.

Il reste non structuré.

Ce chapitre construit la couche suivante : extraire du texte normalisé les unités linguistiques (mots, morphèmes flexionnels) et les porter à un niveau d'abstraction supérieur — catégories grammaticales, lemmes, et surtout entités nommées. Une entité nommée est un fragment de texte qui réfère à un objet du monde : une personne, un lieu, une institution, une date. Dans le contexte des humanités numériques médiévales, ces entités sont directement exploitables pour construire des bases de connaissances, cartographier des réseaux sociaux de notables, ou indexer des corpus documentaires à grande échelle.

Deux difficultés spécifiques aux langues historiques traversent tout ce chapitre. La première est la segmentation : les espaces manuscrits sont irréguliers, et les modèles HTR ne restituent pas toujours fidèlement les frontières de mots. La seconde est la rareté des données annotées : contrairement au français moderne, il existe très peu de corpus médiévaux annotés en POS et en NER, ce qui impose des stratégies d'adaptation que ce chapitre expose systématiquement.

---

## 1. Le problème de la segmentation en mots sur manuscrits

### 1.1 Les espaces manuscrits sont irréguliers

Un modèle HTR entraîné sur des images de manuscrits ne voit pas les mots — il voit des pixels. La décision de placer une espace entre deux séquences de caractères est prise au moment du décodage, sur la base de la largeur de l'écart entre deux formes graphiques consécutives. Mais les scribes médiévaux n'avaient pas de règle typographique : l'espacement varie selon la vitesse d'écriture, le niveau d'encre sur la plume, et le support.

Trois erreurs de segmentation sont fréquentes dans les sorties HTR sur manuscrits médiévaux :

**Fusion de mots** (*merge*) : deux mots distincts sont transcrit comme un seul token. *"leseigneur"* au lieu de *"le seigneur"*, *"delafrance"* au lieu de *"de la france"*. Ce cas est particulièrement problématique pour les prépositions et articles clitiques qui précèdent une entité nommée : *"dunormand"* au lieu de *"du normand"*.

**Coupure spurieuse** (*split*) : un mot est découpé en deux tokens. *"nor mandie"* au lieu de *"normandie"*, *"sei gneur"* au lieu de *"seigneur"*. Les mots longs à jambages complexes sont les plus touchés.

**Tiret résiduel** : certains scribes utilisaient un trait de liaison pour noter les coupures de mots en fin de ligne. Ces traits sont parfois transcrits comme des tirets dans le flux de texte, produisant des tokens comme *"nor-"* et *"mandie"*.

### 1.2 Évaluation de la segmentation

L'évaluation de la segmentation est mesurée par une *accuracy sur tokens* : on compare la séquence de tokens produite par le pipeline à une référence annotée manuellement. Deux métriques complémentaires s'utilisent en pratique :

**Token boundary accuracy (TBA)** : fraction des frontières de mots correctement identifiées. Si une séquence de 100 caractères a 20 frontières de mots, et que 18 sont correctement placées, TBA = 18/20 = 90 %.

**Token error rate (TER)** : analogue au CER mais calculé sur les tokens. Une fusion compte comme une substitution et une suppression simultanées.

Le syllabus demande une évaluation sur 100 tokens annotés manuellement. Cette annotation croisée (deux annotateurs indépendants) sert aussi à mesurer l'IAA, traité en section 5.

### 1.3 Deux approches : règles et CRF

**Approche par règles :** la tokenisation par règles applique une suite d'expressions régulières qui normalisent les espaces, séparent les clitiques (*de la → de la*), et gèrent les cas spéciaux (chiffres romains, abréviations avec point). Cette approche est déterministe, traçable, et suffisante pour un corpus homogène. Elle est la baseline naturelle.

```python
import re

CLITIQUES_PREP = {
    "dela": "de la",    "dun":  "d un",   "des":  "des",
    "dup":  "du p",     "du":   "du",     "au":   "au",
    "auxl": "aux l",
}

def tokenize_medieval(text: str) -> list[str]:
    """
    Tokenisation par règles pour le moyen français.
    Gère les fusions clitique+nom et les coupures spurieuses courantes.
    """
    # Normalisation des espaces multiples
    text = re.sub(r'\s+', ' ', text.strip())
    # Séparation des clitiques contractés courants
    text = re.sub(r'\bdela\b', 'de la', text)
    text = re.sub(r'\bdela\b', 'de la', text)
    text = re.sub(r"\bd[' ']un\b", "d un", text)
    # Séparation chiffres romains / texte adjacent
    text = re.sub(r'(?<=[A-Z])(?=[a-zé])', ' ', text)
    # Tirets résiduels de coupure de ligne
    text = re.sub(r'-\s+', '', text)   # "nor- mandie" → "normandie"
    return text.split()
```

**Approche CRF (*Conditional Random Field*) :** les CRF sont des modèles de séquences discriminatifs qui prédisent une étiquette pour chaque caractère ou n-gramme de caractères. Appliqué à la segmentation, un CRF prédit, pour chaque position du texte, si cette position est une frontière de mot ou non. L'avantage sur les règles est la capacité à généraliser depuis des exemples annotés, y compris pour des cas non couverts par les règles.

Un CRF pour la segmentation utilise typiquement des traits (*features*) contextuels :

```python
def extract_char_features(chars: list[str], i: int) -> dict:
    """
    Traits contextuels pour la prédiction de frontière de mot
    à la position i dans la séquence de caractères.
    """
    char = chars[i]
    features = {
        'char':       char,
        'char.lower': char.lower(),
        'char.isupper': char.isupper(),
        # Bigramme courant
        'bigram_prev': chars[i-1] + char if i > 0 else '<BOS>',
        'bigram_next': char + chars[i+1] if i < len(chars)-1 else '<EOS>',
        # Contexte élargi
        'prev2': ''.join(chars[max(0,i-2):i]),
        'next2': ''.join(chars[i+1:min(len(chars),i+3)]),
        # Trait : suivi d'espace large (à injecter depuis les métadonnées HTR)
        'BOS': i == 0,
        'EOS': i == len(chars) - 1,
    }
    return features
```

La complexité de l'apprentissage CRF est $O(n \cdot T^2)$ pour l'algorithme forward-backward, où $n$ est la longueur de la séquence et $T$ le nombre d'étiquettes. Pour une segmentation binaire (frontière / non-frontière), $T=2$ et le coût est essentiellement linéaire. Nous reverrons ce coût quadratique en $T$ en section 3 pour la NER, où $T$ peut atteindre 21 avec le schéma BIOES.

---

## 2. Étiquetage morpho-syntaxique (POS-tagging) et lemmatisation

### 2.1 Pourquoi la morphologie médiévale est difficile

Le Chapitre 4 a montré que le moyen français a une orthographe non standardisée. Le POS-tagging ajoute une deuxième couche de difficulté : la morphologie est encore partiellement flexionnelle. Le moyen français conserve des traces de la déclinaison à deux cas du latin médiéval — cas sujet (*nominatif*) et cas régime (*tous les autres cas*) — pour les substantifs et adjectifs masculins singuliers. *Li roys* (cas sujet) s'oppose à *le roi* (cas régime). Ces deux formes ont des étiquettes POS identiques (DET + NOUN), mais leurs propriétés morphologiques diffèrent.

Pour le lemmatiseur, la difficulté est de ramener *venoit*, *venait*, *vint*, *venu*, *venans* au même lemme *venir* malgré des formes de surface très différentes. Les modèles de lemmatisation entraînés sur le français moderne échouent systématiquement sur ces formes médiévales.

### 2.2 Les outils spécialisés : pie-extended et Stanza

**pie-extended** (*Ponteineptique/PieExtended*) est le principal outil de lemmatisation et POS-tagging pour le français médiéval. Il est construit sur l'architecture *Pie* (*Probabilistic Iterative Enumerator*), un modèle séquence-à-séquence à mémoire pour les langues morphologiquement riches. Le modèle `medieval-fr` de pie-extended a été entraîné sur plusieurs corpus médiévaux annotés selon le schéma *Cattex* (étiquettes morphosyntaxiques spécialisées pour le français médiéval).

```python
# Installation : pip install pie-extended
from pie_extended.pipeline import ExtensiblePipeline

# Chargement du modèle medieval-fr
pipeline = ExtensiblePipeline.from_pretrained("medieval-fr")

# Annotation d'une phrase normalisée
text = "le sénéchal porta les lettres au château"
result = pipeline.annotate(text)

# Résultat : liste de (forme, lemme, pos, morph)
for token in result:
    print(f"{token.form:15s} {token.lemma:15s} {token.pos:6s} {token.morph}")
```

**Stanza** est une bibliothèque NLP de Stanford avec des modèles pré-entraînés pour de nombreuses langues. Pour le français médiéval, deux modèles sont disponibles et leur distinction est importante :

- `fro` (*Old French* / vieux français) : couvre le français du XIe au XIIIe siècle, incluant la *Chanson de Roland* et les épopées en vers. Ce modèle a été entraîné sur le corpus SRCMF (Syntactic Reference Corpus of Medieval French).
- `frm` (*Middle French* / moyen français) : couvre le français du XIVe au XVe siècle, la période exacte de vos corpus de chartes et registres. Ce modèle est entraîné sur la partie médiévale de l'Arboratoire (*Arboratoire du français médiéval*).

```python
import stanza

# Téléchargement du modèle moyen français
stanza.download('frm')

# Pipeline POS + lemmatisation + dépendances
nlp_frm = stanza.Pipeline(
    lang     = 'frm',
    processors = 'tokenize,mwt,pos,lemma,depparse',
    tokenize_pretokenized = True,   # on fournit les tokens déjà segmentés
)

texte_normalise = "li chevalier venait de loin avec ses missives"
doc = nlp_frm(texte_normalise)
for sent in doc.sentences:
    for word in sent.words:
        print(f"{word.text:15s}  lemma={word.lemma:15s}  "
              f"upos={word.upos:6s}  head={word.head}")
```

**Quelle différence entre fro et frm ?** La question est régulièrement posée et mérite une réponse précise. Le vieux français et le moyen français partagent des racines mais diffèrent sur plusieurs points critiques pour le POS-tagger :

| Caractéristique | Vieux français (fro) | Moyen français (frm) |
|---|---|---|
| Déclinaison à deux cas | Très présente | En déclin, traces résiduelles |
| Negation | *ne* seul fréquent | *ne...pas* s'impose |
| Article défini | *li, lo, la, les* | *le, la, les* dominant |
| Morphologie verbale | Désinences latines plus proches | Formes en -oit/-ait |
| Corpus d'entraînement | SRCMF (~250K tokens) | Arboratoire (~180K tokens) |

Utiliser `fro` sur des textes du XVe siècle produit des erreurs POS systématiques sur les articles et sur les formes verbales à l'imparfait. Le modèle `frm` est le choix correct pour votre corpus.

### 2.3 Universal Dependencies pour le moyen français

Les annotations morphosyntaxiques produites par Stanza suivent le cadre *Universal Dependencies* (UD), un schéma d'annotation multilingue conçu pour être comparable entre langues. Dans UD, chaque token porte :

- **UPOS** (*Universal POS*) : étiquette de catégorie grammaticale parmi 17 valeurs universelles — NOUN, VERB, ADJ, ADV, DET, PRON, ADP (préposition), CCONJ, SCONJ, PUNCT, NUM, PROPN (nom propre), AUX, PART, INTJ, X, SYM.
- **XPOS** (*Language-specific POS*) : étiquette spécifique à la langue, ici le schéma Cattex pour le français médiéval.
- **Feats** (*Morphological features*) : traits morphologiques sous forme de paires clé-valeur — `Number=Sing|Case=Nom|Gender=Masc` pour *li roys*.
- **Head** et **DepRel** : arc de dépendance syntaxique et relation (*nsubj*, *obj*, *det*, etc.).

Le format d'export est **CoNLL-U**, qui est le format d'interchange standard UD. Chaque token occupe une ligne, les champs étant séparés par des tabulations :

```
# sent_id = charte_1346_001
# text = li sénéchal porta les lettres
1   li          le          DET    DArti  Definite=Def|Gender=Masc|Number=Sing|Case=Nom   2   det      _   _
2   sénéchal    sénéchal    NOUN   NOMcom Gender=Masc|Number=Sing|Case=Nom                3   nsubj    _   _
3   porta       porter      VERB   VERcjg Mood=Ind|Number=Sing|Person=3|Tense=Past        0   root     _   _
4   les         le          DET    DArti  Definite=Def|Gender=Masc|Number=Plur            5   det      _   _
5   lettres     lettre      NOUN   NOMcom Gender=Fem|Number=Plur                          3   obj      _   _
```

Le champ CoNLL-U sera l'un des formats d'export du Chapitre 6. Notez que la colonne 10 (Misc) est l'endroit où vous pourrez encoder le `polygon_ref` liant le token à sa position dans l'image originale — la connexion avec le data contract du Volet 1.

### 2.4 Évaluation du POS-tagging

L'évaluation du POS-tagging se fait en *accuracy* par token et en accuracy par lemme :

$$\text{POS Accuracy} = \frac{\text{tokens correctement étiquetés}}{\text{tokens totaux}}$$

Une accuracy de 90 % est un résultat typique pour pie-extended sur des textes médiévaux propres. Sur des transcriptions HTR brutes avec CER 10 %, cette valeur chute à environ 82-85 %, principalement à cause des mots mal orthographiés que le modèle ne reconnaît pas.

---

## 3. Schémas d'annotation pour la NER

### 3.1 BIO vs BIOES

Un annotateur humain lit une phrase et identifie les entités nommées. Pour entraîner un modèle de classification de tokens, il faut convertir ces annotations en étiquettes par token. Deux schémas d'étiquetage dominent la littérature.

**Schéma BIO :** chaque token reçoit une étiquette parmi :
- `B-TYPE` (*Beginning*) : premier token d'une entité de type TYPE
- `I-TYPE` (*Inside*) : token intérieur d'une entité de type TYPE
- `O` (*Outside*) : token hors d'une entité

Pour cinq types d'entités médiévales (PER, LOC, DATE, ORG, TITLE), le schéma BIO produit $2 \times 5 + 1 = 11$ étiquettes.

**Schéma BIOES :** extension de BIO avec deux étiquettes supplémentaires :
- `E-TYPE` (*End*) : dernier token d'une entité multi-tokens
- `S-TYPE` (*Single*) : entité d'un seul token

Pour les mêmes cinq types, BIOES produit $4 \times 5 + 1 = 21$ étiquettes.

```
Phrase : "le seigneur jean de normandie envoya ses chevaliers"

BIO :
  le          O
  seigneur    B-TITLE
  jean        B-PER
  de          I-PER
  normandie   I-PER
  envoya      O
  ses         O
  chevaliers  O

BIOES :
  le          O
  seigneur    S-TITLE     ← entité d'un seul token → S, pas B
  jean        B-PER
  de          I-PER
  normandie   E-PER       ← dernier token → E, pas I
  envoya      O
  ses         O
  chevaliers  O
```

**Quel schéma choisir ?** BIOES apporte une information supplémentaire au modèle : la distinction entre un token de début d'entité qui sera suivi d'autres tokens (`B`) et un token constituant une entité à lui seul (`S`) aide le décodage, notamment pour les entités mono-token. Les expériences empiriques montrent un gain de F1 de 0.5 à 2 points pour BIOES vs BIO sur les tâches de NER, au prix d'un vocabulaire d'étiquettes plus large. Pour vos corpus médiévaux où les entités TITLE et DATE sont souvent mono-token (*"seigneur"*, *"mars"*), BIOES est recommandé.

**Entités nichées (*nested entities*) :** BIO et BIOES gèrent une seule couche d'annotation. Dans *"le roi de France Jean II"*, l'entité PER *"Jean II"* est nichée dans une entité composite qui inclut son titre. Pour les entités nichées, il existe des extensions comme le multi-label BIO (plusieurs étiquettes par token) ou des modèles de span à plusieurs niveaux. Ce cas avancé est mentionné ici pour signaler son existence ; il n'est pas dans le périmètre du Chapitre 6.

### 3.2 Les entités cibles et leurs spécificités médiévales

Le syllabus définit cinq types d'entités pour votre corpus :

**PER — Personnes.** Les noms de personnes médiévaux ont des propriétés distinctives. D'abord, le prénom seul suffisait souvent à identifier une personne de rang (*"li roys"* sans prénom désigne le roi en exercice dans le contexte de la charte). Ensuite, les patronymes n'existaient pas encore au sens moderne : les personnes sont identifiées par leur prénom, leur titre, et leur lieu d'origine (*"Jean, sénéchal de Normandie"*). Enfin, les prénoms médiévaux sont fréquemment abrégés (*"Jehan"*, *"Gilles"*, *"Guill^me"*) et leurs variantes graphiques sont nombreuses.

**LOC — Lieux.** Les noms de lieux médiévaux sont particulièrement difficiles : beaucoup ne correspondent pas à des noms de lieux modernes (le *"chastel de Gisors"* est *"le château de Gisors"*, mais *"le bois de Vincenne"* peut correspondre à plusieurs lieux), d'autres n'existent plus. La confusion avec ORG est fréquente pour les lieux institutionnels (*"la cour de Paris"* est-elle un LOC ou un ORG ?).

**DATE — Dates.** Les dates médiévales suivent des conventions distinctes du calendrier grégorien moderne. La datation *"en l'an de grâce mil trois cent quarante-six"* doit être normalisée en une représentation structurée. Les fêtes religieuses sont des repères temporels (*"à la Saint-Jean"*, *"après Pâques"*) dont la résolution requiert le calendrier liturgique.

**ORG — Organisations.** Institutions, chapitres, abbayes, parlement, prévôtés. La frontière avec LOC est floue pour les institutions localisées (*"l'abbaye de Saint-Denis"*).

**TITLE — Titres.** Les titres de noblesse et fonctions (*"sénéchal"*, *"prévôt"*, *"bailli"*, *"châtelain"*, *"seigneur"*, *"monseigneur"*) sont une catégorie spécifique qui anticipe sur les relations entre entités. La confusion TITLE/PER est l'erreur la plus fréquente des modèles NER sur ce corpus : *"le seigneur"* sans antécédent peut être un TITLE ou désigner implicitement une PER.

---

## 4. Architecture du modèle : classification de tokens avec BERT

### 4.1 Le problème de l'alignement subword

CamemBERT, comme tous les modèles basés sur BERT, opère sur des subwords produits par l'algorithme BPE (Chapitre 1, section 7). Un token NER comme *"seneschal"* peut être découpé en plusieurs subwords : `['s', '##enes', '##chal']`. Or, l'annotation NER s'applique au niveau du mot, pas du subword. Il faut réconcilier les deux niveaux.

Deux stratégies sont en usage :

**Stratégie *first-token* (recommandée) :** seul le premier subword d'un mot reçoit l'étiquette NER ; les subwords suivants reçoivent l'étiquette spéciale `-100` (ignorée par la loss). Cette stratégie est simple et produit de bons résultats en pratique.

**Stratégie *majority vote* :** l'étiquette est prédite pour chaque subword, et le mot reçoit l'étiquette majoritaire. Plus coûteuse, elle peut aider pour les entités dont le radical porteur de sens se trouve en milieu de mot.

```python
def align_labels_to_subwords(tokens:     list[str],
                              labels:     list[str],
                              tokenizer,
                              strategy:   str = "first") -> dict:
    """
    Aligne les étiquettes NER (au niveau mot) avec les subwords.

    Paramètres
    ----------
    tokens    : list[str]   mots de la phrase (déjà tokenisés)
    labels    : list[str]   étiquettes NER par mot, ex. ["O","B-PER","I-PER"]
    tokenizer : tokenizer   HuggingFace BPE tokenizer
    strategy  : str         "first" ou "majority"

    Retourne
    --------
    dict avec "input_ids", "attention_mask", "labels"
    (labels contient -100 pour les subwords non-premiers)
    """
    encoding = tokenizer(
        tokens,
        is_split_into_words = True,
        return_offsets_mapping = True,
        truncation = True,
        max_length = 512,
    )
    word_ids      = encoding.word_ids()
    aligned_labels = []
    prev_word_id   = None

    for word_id in word_ids:
        if word_id is None:               # tokens spéciaux [CLS], [SEP]
            aligned_labels.append(-100)
        elif word_id != prev_word_id:     # premier subword du mot
            aligned_labels.append(labels[word_id])
        else:                             # subword suivant
            if strategy == "first":
                aligned_labels.append(-100)
            else:  # majority : propager l'étiquette
                aligned_labels.append(labels[word_id])
        prev_word_id = word_id

    encoding["labels"] = aligned_labels
    return encoding
```

### 4.2 Softmax vs CRF en sortie

L'architecture BERT+NER la plus simple ajoute une couche linéaire au-dessus des représentations contextuelles de CamemBERT, suivie d'une softmax :

$$\hat{y}_t = \text{softmax}(W \cdot h_t + b)$$

où $h_t \in \mathbb{R}^{768}$ est la représentation du token $t$ produite par CamemBERT, et $W \in \mathbb{R}^{T \times 768}$ est la matrice de classification ($T$ = nombre d'étiquettes).

**Limite de la softmax :** la softmax prédit chaque token *indépendamment*. Elle ne peut pas apprendre que la séquence `I-PER` ne peut pas immédiatement suivre `B-LOC` — une contrainte structurelle qui est pourtant triviale dans tout schéma BIO correct. Des séquences d'étiquettes incohérentes peuvent donc être produites.

**CRF (*Conditional Random Field*) linéaire :** une couche CRF ajoutée au-dessus de la softmax modélise les dépendances entre étiquettes consécutives. Elle apprend une matrice de transitions $A \in \mathbb{R}^{T \times T}$ où $A_{ij}$ représente le score de passer de l'étiquette $i$ à l'étiquette $j$ d'un token au suivant. Le score total d'une séquence d'étiquettes $(y_1, \ldots, y_n)$ est :

$$s(y_1, \ldots, y_n) = \sum_{t=1}^{n} P_t[y_t] + \sum_{t=1}^{n-1} A[y_t, y_{t+1}]$$

où $P_t[y_t]$ est le score d'émission de l'étiquette $y_t$ au token $t$ (sortie de la softmax), et $A[y_t, y_{t+1}]$ est le score de transition.

Le décodage utilise l'**algorithme de Viterbi** pour trouver la séquence d'étiquettes de score maximal en $O(n \cdot T^2)$ :

```python
import torch

def viterbi_decode(emissions: torch.Tensor,
                   transitions: torch.Tensor) -> list[int]:
    """
    Décode la meilleure séquence d'étiquettes par l'algorithme de Viterbi.

    Paramètres
    ----------
    emissions   : Tensor (n_tokens, n_labels)  scores d'émission par token
    transitions : Tensor (n_labels, n_labels)  matrice de transition CRF

    Retourne
    --------
    list[int]  indices des étiquettes optimales
    """
    n_tokens, n_labels = emissions.shape
    # viterbi[t][j] = score max de la meilleure séquence terminant à j au token t
    viterbi   = torch.full((n_tokens, n_labels), -1e9)
    backtrack = torch.zeros((n_tokens, n_labels), dtype=torch.long)

    # Initialisation
    viterbi[0] = emissions[0]

    # Récurrence
    for t in range(1, n_tokens):
        # (n_labels, 1) + (n_labels, n_labels) → max sur dim=0
        scores = viterbi[t-1].unsqueeze(1) + transitions  # (n_labels, n_labels)
        best_scores, best_tags = scores.max(dim=0)
        viterbi[t]   = best_scores + emissions[t]
        backtrack[t] = best_tags

    # Reconstruction du chemin optimal
    best_path = [viterbi[-1].argmax().item()]
    for t in range(n_tokens - 1, 0, -1):
        best_path.append(backtrack[t][best_path[-1]].item())
    best_path.reverse()
    return best_path
```

**Coût de la couche CRF :** la matrice de transitions $A$ ajoute $T^2$ paramètres. Pour $T=11$ (BIO, 5 types) : 121 paramètres. Pour $T=21$ (BIOES) : 441 paramètres. Comparé aux 110 millions de paramètres de CamemBERT, le surcoût est de l'ordre de $0.0001$% — négligeable en termes de mémoire, mais l'algorithme de Viterbi ajoute un coût computationnel à l'inférence.

**Gain empirique :** sur des tâches de NER standard, BERT+CRF surpasse BERT+softmax de 0.5 à 1.5 points de F1. Le gain est plus marqué pour les entités multi-tokens et les corpus avec schéma BIOES. Pour votre corpus médiéval avec des entités longues (noms de lieux composés, titres suivis d'un prénom), la couche CRF est recommandée.

### 4.3 CamemBERT et le moyen français

CamemBERT a été pré-entraîné sur le corpus *OSCAR* (essentiellement du web français moderne). Ses tokens subwords et ses représentations contextuelles sont optimisés pour le français du XXIe siècle. Appliqué directement sur du moyen français normalisé, il présente deux limitations :

Premièrement, le vocabulaire BPE de CamemBERT découpe les mots médiévaux résiduels en nombreux subwords peu informatifs : *"sénéchal"* → `['s', '##én', '##échal']`, *"châtelain"* → `['ch', '##âtelain']`. Le taux de subwords de faible fréquence dans le modèle est élevé, ce qui dilue la représentation contextuelle.

Deuxièmement, les co-occurrences apprises en pré-entraînement ne correspondent pas aux co-occurrences du corpus médiéval. *"Seigneur"* apparaît dans CamemBERT en contexte religieux ou formel contemporain ; dans vos chartes, il désigne systématiquement un noble local avec des co-occurrences spécifiques (*"porta les lettres"*, *"signa l'acte"*).

**Stratégie :** le fine-tuning LoRA du Chapitre 3 atténue ces deux limitations. Les adaptateurs LoRA apprennent les représentations des entités médiévales à partir des annotations disponibles, sans modifier tous les poids du modèle. Le Chapitre 6 détaille ce fine-tuning.

---

## 5. Mesure de l'accord inter-annotateurs (IAA)

### 5.1 Pourquoi l'IAA est obligatoire

Avant d'évaluer un modèle, il faut avoir une référence fiable. Pour que cette référence soit fiable, il faut que les humains qui l'ont produite soient d'accord entre eux. Si deux annotateurs experts ne s'accordent que sur 70 % des entités, un modèle atteignant 80 % n'est pas nécessairement meilleur que les humains — il peut simplement être biaisé vers l'un des annotateurs.

La mesure de l'IAA est aussi un outil de débogage des consignes d'annotation : si le kappa entre deux annotateurs est inférieur à 0.7, les consignes sont ambiguës et doivent être précisées avant d'annoter davantage. *CONVENTIONS_NLP.md* joue ici un rôle directement comparable à sa fonction pour la normalisation.

### 5.2 Le kappa de Cohen pour la NER

Le kappa de Cohen $\kappa$ mesure l'accord au-delà de ce qui serait attendu par le hasard :

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

où $p_o$ est l'accord observé (proportion de tokens sur lesquels les deux annotateurs sont d'accord) et $p_e$ est l'accord attendu par le hasard.

Pour calculer $p_e$, soit $p_k^{(1)}$ la proportion de tokens étiquetés avec l'étiquette $k$ par l'annotateur 1, et $p_k^{(2)}$ la même proportion pour l'annotateur 2. Alors :

$$p_e = \sum_{k} p_k^{(1)} \cdot p_k^{(2)}$$

**Exemple sur 100 tokens :** l'annotateur A produit 50 `O`, 25 `B-PER`, 25 autres. L'annotateur B produit 52 `O`, 24 `B-PER`, 24 autres. Accord observé : 85 tokens sur 100 concordants → $p_o = 0.85$.

$$p_e = 0.50 \times 0.52 + 0.25 \times 0.24 + \ldots \approx 0.375$$

$$\kappa = \frac{0.85 - 0.375}{1 - 0.375} = \frac{0.475}{0.625} = 0.76$$

**Interprétation du kappa :**

| Kappa | Interprétation |
|---|---|
| < 0.20 | Accord médiocre (consignes à retravailler) |
| 0.21 – 0.40 | Accord passable |
| 0.41 – 0.60 | Accord modéré |
| 0.61 – 0.80 | Accord substantiel — acceptable pour un corpus de référence |
| > 0.80 | Accord presque parfait |

Pour la NER sur corpus médiéval, un kappa de 0.70 est un objectif raisonnable pour la première session d'annotation. Les cas les plus divergents — frontières floues TITLE/PER, entités imbriquées — alimentent la mise à jour des consignes.

### 5.3 Kappa sur séquences NER : le problème de O

Dans les corpus NER, la classe `O` est massivement dominante : 70 à 85 % des tokens sont hors entité. Cela gonfle artificiellement $p_e$ (les deux annotateurs vont souvent coïncider sur `O` par chance) et rend le kappa moins discriminant sur les classes d'entités. Deux alternatives sont utilisées :

**Kappa pondéré** : pondère les désaccords selon leur gravité (confondre `B-PER` et `I-PER` est moins grave que confondre `B-PER` et `O`).

**Macro-kappa par type** : calculer le kappa indépendamment pour chaque type d'entité, en excluant les tokens `O` des deux annotateurs. Cette approche isole les difficultés par type et est plus informative pour diagnostiquer les sources de désaccord.

```python
def cohen_kappa(labels_a: list, labels_b: list) -> float:
    """
    Calcule le kappa de Cohen entre deux séquences d'étiquettes.
    """
    from collections import Counter
    assert len(labels_a) == len(labels_b)
    n = len(labels_a)
    # Accord observé
    p_o = sum(a == b for a, b in zip(labels_a, labels_b)) / n
    # Accord attendu par chance
    freq_a = Counter(labels_a)
    freq_b = Counter(labels_b)
    all_labels = set(labels_a) | set(labels_b)
    p_e = sum(
        (freq_a[k] / n) * (freq_b[k] / n)
        for k in all_labels
    )
    return (p_o - p_e) / (1 - p_e) if p_e < 1 else 0.0

# Exemple
labels_a = ["O","B-PER","I-PER","O","B-LOC","O"] * 16 + ["O","O","O","O"]
labels_b = ["O","B-PER","I-PER","O","B-LOC","O"] * 15 + ["O","B-PER","O","O","O","O","O","O","O","O"]
kappa = cohen_kappa(labels_a, labels_b)
print(f"kappa = {kappa:.3f}")
```

---

## 6. Propagation d'erreurs HTR → NER

### 6.1 Le mécanisme de dégradation

Les erreurs de transcription HTR se propagent à toutes les couches d'analyse NLP. La question du syllabus est quantitative : dans quelle mesure le CER de la transcription dégrade-t-il le F1-NER ?

La relation n'est pas uniforme selon le type d'entité :

- **LOC et DATE** sont robustes : les noms de lieux courants et les dates numériques sont reconnus même avec plusieurs erreurs de caractères. *"Normndie"* (normalement *"Normandie"*) est encore reconnaissable comme un LOC pour un modèle qui a vu des variantes graphiques en entraînement.
- **PER** est fragile : les prénoms médiévaux rares (*"Enguerrand"*, *"Thibaut"*) sont souvent des hapax dont la reconnaissance dépend de l'orthographe exacte. Une seule substitution de caractère peut rendre le token méconnaissable.
- **TITLE** est intermédiaire : les titres de noblesse sont fréquents et le modèle les reconnaît par co-occurrence avec des structures syntaxiques stables (*"le [TITLE] de [LOC]"*), donc résistants aux erreurs légères.

### 6.2 Simulation de la dégradation

On peut simuler l'effet de différents niveaux de CER sur le F1-NER en injectant des erreurs aléatoires dans un texte de référence :

```python
import random

def inject_cer_errors(text: str, cer: float, seed: int = 42) -> str:
    """
    Injecte des erreurs de type HTR dans un texte pour simuler un CER donné.

    Opérations : substitution de caractère (60%), insertion (20%),
                 suppression (20%) — proportions typiques d'un modèle HTR.
    """
    rng    = random.Random(seed)
    chars  = list(text)
    n_errors = int(len(chars) * cer)
    charset  = "abcdefghijklmnopqrstuvwxyzéàèùâêîôûç "

    for _ in range(n_errors):
        idx = rng.randint(0, len(chars) - 1)
        op  = rng.choices(['sub', 'ins', 'del'], weights=[0.6, 0.2, 0.2])[0]
        if op == 'sub':
            chars[idx] = rng.choice(charset)
        elif op == 'ins':
            chars.insert(idx, rng.choice(charset))
        elif op == 'del' and len(chars) > 1:
            chars.pop(idx)
    return ''.join(chars)

# Simulation : F1-NER estimé pour différents CER
# (basé sur Hamdi et al. 2021, extrapolé pour textes médiévaux)
cer_f1_simulation = {
    0.00: 0.820,
    0.05: 0.745,
    0.10: 0.670,
    0.12: 0.640,
}

print("Simulation propagation CER → F1-NER (corpus médiéval) :")
for cer, f1 in cer_f1_simulation.items():
    degradation = (0.820 - f1) / 0.820 * 100
    print(f"  CER {cer*100:4.0f}% → F1-NER ≈ {f1:.3f}  (dégradation : -{degradation:.1f}%)")
```

| CER HTR | F1-NER estimé | Dégradation relative |
|---|---|---|
| 0 % (texte parfait) | 0.820 | — |
| 5 % | 0.745 | −9.1 % |
| 10 % | 0.670 | −18.3 % |
| 12 % | 0.640 | −22.0 % |

**Leçon pratique :** passer de CER 12 % à CER 5 % (par exemple en améliorant le modèle HTR ou en appliquant le pipeline de normalisation du Jour 2) récupère environ 13 points de F1-NER. Cela justifie a posteriori l'investissement dans la normalisation orthographique avant la NER.

### 6.3 Rôle du pipeline de normalisation

Le pipeline de normalisation des Chapitres 3 et 4 n'est pas seulement utile pour la lisibilité du corpus — il est directement bénéfique pour la NER. En réduisant le CER de 12 % à 5-7 % (valeur typique après les règles graphiques et le cache DMF), il améliore mécaniquement le F1-NER de plusieurs points. C'est la justification quantitative de l'ordre des étapes dans votre pipeline de traitement.

---

## 7. Point éthique : biais de représentation en NER

### 7.1 Quels corpus médiévaux existent, et qui représentent-ils ?

Les corpus médiévaux annotés disponibles aujourd'hui ont été constitués sur la base de choix éditoriaux : quels documents conserver, lesquels numériser, lesquels annoter. Ces choix ont des biais systématiques qui se transmettent aux modèles NER entraînés sur ces corpus.

**Biais de classe sociale :** les chartes, registres de compte, et rôles fiscaux — les documents les plus abondants dans CREMMA — représentent quasi exclusivement les couches supérieures de la société médiévale : noblesse, clergé, bourgeoisie urbaine. Les noms de personnes de condition servile ou paysanne apparaissent peu, et quand ils apparaissent, c'est comme objets d'actes juridiques plutôt qu'en tant qu'agents. Un modèle NER entraîné sur ce corpus reconnaîtra bien les comtes et les évêques, moins bien les artisans et vilains.

**Biais géographique :** les corpus sont concentrés sur quelques grandes régions documentées (Île-de-France, Normandie, Bourgogne). Les dialectes régionaux (picard, lorrain, gascon) sont sous-représentés. Les noms de lieux de ces régions seront moins bien reconnus.

**Biais de genre :** les scripteurs médiévaux étaient presque exclusivement masculins, et les actes juridiques impliquant des femmes en tant que parties actives sont rares. Les prénoms féminins sont sous-représentés dans les entités PER.

### 7.2 Conséquences sur le modèle

Un modèle NER entraîné sur ces corpus aura des performances asymétriques :

- F1 élevé sur les entités fréquentes dans le corpus d'entraînement (nobles masculins, lieux d'Île-de-France).
- F1 bas sur les entités rares (femmes, paysans, lieux périphériques).
- Risque de biais de confirmation : si le modèle est utilisé pour construire une base de connaissances sur la société médiévale, il renforcera la sous-représentation de ceux que le corpus d'entraînement sous-représentait déjà.

### 7.3 Mitigation

Trois stratégies de mitigation sont envisageables :

**Stratification du corpus d'annotation :** lors du choix des 100 lignes à annoter manuellement pour l'IAA, s'assurer qu'elles couvrent différents types de documents, régions, et périodes. Ne pas annoter uniquement les lignes les plus "propres" ou les plus faciles.

**Augmentation ciblée :** pour les types d'entités sous-représentés, produire des exemples synthétiques supplémentaires (par substitution de noms dans des phrases templates) et les ajouter au corpus d'entraînement.

**Rapport d'analyse d'erreurs stratifié :** lors de l'évaluation du modèle (Chapitre 6), calculer le F1 par type d'entité et par sous-corpus (type de document, région, date). Rendre visible le différentiel de performance est la première étape indispensable.

---

## Bibliographie de référence

### Segmentation et tokenisation

Schmid, H. (1994). **Probabilistic Part-of-Speech Tagging Using Decision Trees**. *Proceedings of International Conference on New Methods in Language Processing*. — Référence fondatrice sur les modèles de séquence pour le POS-tagging.

Grover, C., & Tobin, R. (2006). **Rule-Based Chunking and Reusability**. *LREC 2006*. — Tokenisation par règles pour les langues historiques.

### POS-tagging et lemmatisation médiévale

Camps, J.-B., Clérice, T., & Duval, F. (2021). **Corpus and Models for Lemmatisation and POS-tagging of Old French**. *Journal of Data Mining & Digital Humanities*. — Référence directe pour pie-extended.

Qi, P., Zhang, Y., Zhang, Y., Bolton, J., & Manning, C. D. (2020). **Stanza: A Python Natural Language Processing Toolkit for Many Human Languages**. *ACL 2020 (Demos)*. [arXiv:2003.07082](https://arxiv.org/abs/2003.07082)

Souvay, G., & Pierrel, J.-M. (2009). **LGeRM : lemmatisation des mots en moyen français**. *TAL*, 50(2). — Fondement de l'intégration DMF dans le pipeline.

Nivre, J., et al. (2016). **Universal Dependencies v1: A Multilingual Treebank Collection**. *LREC 2016*. — Référence pour le cadre UD et CoNLL-U.

### NER et classification de tokens

Lample, G., Ballesteros, M., Subramanian, S., Kawakami, K., & Dyer, C. (2016). **Neural Architectures for Named Entity Recognition**. *NAACL 2016*. [arXiv:1603.01360](https://arxiv.org/abs/1603.01360) — Papier fondateur LSTM-CRF pour NER.

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**. *NAACL 2019*. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)

Martin, L., Muller, B., Suárez, P. J. O., Dupont, Y., Romary, L., de la Clergerie, É., Seddah, D., & Sagot, B. (2020). **CamemBERT: a Tasty French Language Model**. *ACL 2020*. [arXiv:1911.03894](https://arxiv.org/abs/1911.03894)

### IAA et évaluation

Cohen, J. (1960). **A coefficient of agreement for nominal scales**. *Educational and Psychological Measurement*, 20(1). — Article original du kappa de Cohen.

Artstein, R., & Poesio, M. (2008). **Inter-Coder Agreement for Computational Linguistics**. *Computational Linguistics*, 34(4). — Revue complète des mesures d'accord pour les tâches NLP.

### Propagation d'erreurs HTR → NLP

Hamdi, A., Linhares Pontes, E., Boros, E., Nguyen, T. T., Plafourcade, G., Cabrera-Diego, L. A., & Moreno, J.-G. (2021). **A Multilingual Dataset for Named Entity Recognition, Entity Linking and Stance Detection in Historical Newspapers**. *ACL-IJCNLP 2021*.

Lopresti, D. (2009). **Optical character recognition errors and their effects on natural language processing**. *IJDAR*, 12(3).

### Biais et éthique

Bender, E. M., Gebru, T., McMillan-Major, A., & Shmitchell, S. (2021). **On the Dangers of Stochastic Parrots**. *FAccT 2021*. — Référence sur les biais dans les corpus de LLM, applicable à toute constitution de corpus NLP.

Jørgensen, A., Hovy, D., & Søgaard, A. (2016). **Socio-demographic Factors in Historical Language Change**. *DH 2016*. — Sur les biais de classe et de genre dans les corpus historiques.

### Outils

pie-extended. [GitHub: PonteIneptique/pie-extended](https://github.com/PonteIneptique/pie-extended)

seqeval. [GitHub: chakki-works/seqeval](https://github.com/chakki-works/seqeval) — Bibliothèque d'évaluation NER standard (F1 micro/macro par type).

---

*Support de cours rédigé pour le Master Data/IA · Module NLP · MD5 Volet 2 · 2026. Ce document accompagne le cours magistral et l'atelier du Jour 3 (09h00–12h00). Il est le prérequis théorique du TP Chapitre 6 (pipeline NER bout en bout) de l'après-midi.*
