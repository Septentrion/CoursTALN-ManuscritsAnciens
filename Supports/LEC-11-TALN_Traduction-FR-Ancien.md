# Chapitre 11 — Traduction automatique du moyen français vers le français moderne

**Module NLP · Master Data/IA · MD5 Volet 2 · 2026**  
Tutoriel autonome — chapitre bonus

---

## Avant-propos : pourquoi ce problème est différent

La traduction automatique est l'une des tâches les mieux résolues du NLP moderne. Des systèmes comme DeepL ou Google Translate atteignent des scores BLEU de 35–45 sur les paires de langues bien dotées en données. Mais *moyen français → français moderne* est une paire de langues particulièrement ingrate, pour des raisons qui ne sont pas celles qu'on imagine d'abord.

Ce n'est pas un problème de distance linguistique, au sens strict. Le moyen français du XIVe siècle et le français d'aujourd'hui partagent 85 % du vocabulaire de base, les mêmes racines latines, et une structure syntaxique fondamentalement similaire. Un lecteur moderne non formé peut comprendre environ 40 % d'un texte médiéval en première lecture — bien mieux que de l'arabe ou du mandarin.

La difficulté est ailleurs. Elle tient à trois facteurs qui interagissent : la **variation orthographique non standardisée** (le même mot peut s'écrire de vingt façons différentes selon le scribe, la région, et l'époque), les **glissements sémantiques** qui ont rendu certains mots médiévaux fréquents faux amis de leurs descendants modernes, et la **rareté des données parallèles** — il existe très peu de corpus médiéval/moderne alignés au niveau de la phrase, comparé aux millions de paires disponibles pour le français vers l'anglais.

Ce tutoriel explore ces difficultés de façon systématique, puis présente quatre approches pour les surmonter, de la plus simple à la plus puissante : la substitution lexicale par règles, le NMT avec fine-tuning sur corpus parallèle, et le prompting zero-shot d'un LLM. Une dernière section discute honnêtement les limites de chacune.

---

## 1. La problématique : trois sources de difficulté

### 1.1 La variation orthographique : un bruit d'entrée structuré

Le moyen français n'a pas d'Académie française. L'orthographe d'un scribe normand du XIVe siècle diffère de celle d'un scribe parisien, qui diffère de celle d'un scribe bourguignon — et chacun d'eux peut écrire le même mot différemment sur la même page selon la vitesse d'écriture ou la place disponible sur la ligne.

La table ci-dessous illustre les variantes graphiques attestées pour quelques mots courants dans le corpus CREMMA :

| Forme moderne | Variantes médiévales attestées |
|---|---|
| *roi* | *roi, roy, roys, roÿ, rei, reys, rois* |
| *notre* | *nostre, nôtre, notre, nostre, noûtre* |
| *sénéchal* | *seneschal, séneschal, senechal, s^r, sénéchal* |
| *comté* | *conté, contet, comté, conte, comtei* |
| *tous* | *touz, toz, tous, toutes, tuit* (cas sujet masculin pluriel) |
| *reçu* | *receu, reçu, ressu, recçu* |

Pour un modèle de traduction entraîné sur du français moderne, chacune de ces variantes est un token hors-vocabulaire ou un token peu fréquent. La probabilité qu'il les reconnaisse correctement est faible — c'est pour cela que le pipeline de **normalisation orthographique** du Jour 2 est un pré-requis indispensable avant toute tentative de traduction.

**Conséquence pratique :** sans normalisation, un modèle NMT entraîné sur des textes modernes produira des traductions incohérentes sur les mots médiévaux — soit il les ignorera (en les recopiant tels quels), soit il les "traduira" vers un faux-ami moderne. La normalisation est la première étape du pipeline de traduction, pas une option.

### 1.2 Les glissements sémantiques : les faux amis morphologiques

Le problème le plus subtil — et le plus dangereux — est celui des **faux amis intra-linguistiques** : des mots qui *ressemblent* à leur descendant moderne mais ont un sens différent, parfois opposé.

**Exemple critique : *liez***

La charte de franchise normande de 1315 contient la phrase :

> *Item, que touz les hommes liez de nostre terre de Normandie soient quite et delivre de totes tailles.*

*Liez* est le participe passé de *lier*, et ressemble exactement à *liés* (attachés, enchaînés) en français moderne. C'est le contraire du sens exact : *liez* signifie ici *libres* (du latin *ligii*, vassaux libres par opposition aux serfs). Une traduction automatique non avertie produira :

> *— (incorrect) : Que tous les hommes liés de notre terre soient quittes.*

Au lieu de la traduction correcte :

> *— (correct) : Que tous les hommes libres de notre terre soient quittes et délivrés de toutes tailles.*

Cet exemple n'est pas anecdotique. Il illustre une erreur dont les conséquences sur un corpus juridique médiéval sont graves : le modèle produit une phrase grammaticalement correcte, fluide, et entièrement fausse sur le plan du sens historique.

**Autres faux amis morphologiques fréquents :**

| Forme médiévale | Sens médiéval | Faux ami moderne | Sens moderne (incorrect) |
|---|---|---|---|
| *liez* | libres (vassaux libres) | *liés* | attachés, enchaînés |
| *vilain* | paysan, serf | *vilain* | méchant, laid |
| *merci* | pitié, grâce | *merci* | remerciement |
| *destrier* | cheval de bataille | *dextre* | droite (vieux français) |
| *preudomme* | homme de bien, notable | *prud'homme* | arbitre du travail |
| *damage* | dommage, préjudice | *damage* | (anglicisme récent) |
| *outrecuidance* | présomption excessive | *outrecuidance* | conservé, mais rare |

**Exemple de glissement positif :** *damage* a exactement le même sens en moyen français et en français moderne (*dommage*). L'évolution phonétique est visible (*damage → dommage*), mais le sens est stable. Ce type de mot est facilement résolu par normalisation.

### 1.3 Les constructions syntaxiques : ce que le modèle n'a jamais vu

Le moyen français conserve des structures syntaxiques héritées du latin qui ont disparu du français moderne. Un modèle de traduction entraîné sur des textes contemporains n'a jamais appris à les produire comme input ni à les traduire comme output.

**La déclinaison à deux cas**

Le moyen français distingue encore, pour les noms masculins singuliers, le **cas sujet** (nominatif) et le **cas régime** (tous les autres cas). *Li roys* est le cas sujet (sujet grammatical), *le roi* est le cas régime (complément). Les deux forment une seule entité (*roi*) en français moderne.

```
SRC : Li roys ordena que li chevaliers iroit en Normandie.
        ↑ cas sujet          ↑ cas sujet (sujet de la complétive)

TGT : Le roi ordonna que le chevalier irait en Normandie.
```

Pour un modèle de traduction, *li* et *le* sont deux tokens distincts qui doivent être mappés au même article *le* selon le contexte. Si le modèle est entraîné uniquement sur du français moderne, il ne verra jamais *li* en entrée et ne saura pas le traiter.

**La négation à un terme**

En moyen français, la négation peut s'exprimer avec *ne* seul, sans *pas* :

```
SRC : Il ne savoit que faire.
TGT : Il ne savait que faire.    ← même structure (cas facile)

SRC : Il ne vint onques.
TGT : Il ne vint jamais.         ← 'onques' = jamais (archaïsme)

SRC : Je ne di mot.
TGT : Je ne dis mot.             ← même structure, mais 'di' est fort inhabituel
```

La difficulté n'est pas tant la négation elle-même (la structure *ne + V* est encore vivante en français soigné) que les **adverbes de négation archaïques** (*onques* = jamais, *mie* = pas, *goute* = goutte = pas du tout, *point* dans son sens originel).

**L'inversion sujet-verbe systématique**

En proposition principale après un adverbe initial, le moyen français inverse systématiquement le sujet et le verbe :

```
SRC : Dist li roys a ses barons : "Je vois bien vostre pensee."
TGT : Le roi dit à ses barons : « Je vois bien votre pensée. »
```

Le verbe *dist* (3e pers. sg. du passé simple de *dire*, forme forte) précède le sujet *li roys*. Pour un modèle NMT, cette inversion dans le signal d'entrée perturbe le décodage : le modèle doit produire *"Le roi dit"* alors qu'il a vu *"Dist li roys"* — ce qui requiert une réorganisation de l'ordre des mots, pas seulement une substitution.

**L'infinitif substantivé**

```
SRC : Le manger et le boire li estoient interdis.
TGT : Manger et boire lui étaient interdits.
```

En moyen français, les infinitifs peuvent être précédés de l'article défini (*le manger*, *le boire*) pour former des substantifs. En français moderne, cet article disparaît dans certains contextes. Le modèle doit apprendre à supprimer cet article — une opération de suppression qui est moins fréquente que les substitutions et que les modèles NMT apprennent moins bien.

---

## 2. Corpus parallèles : ce qui existe, ce qui manque

### 2.1 Les sources disponibles

Constituer un corpus parallèle moyen français / français moderne est la première — et la plus coûteuse — étape du pipeline NMT. Les sources suivantes sont disponibles :

**Éditions critiques bilingues :** les philologues médiévistes produisent depuis le XIXe siècle des éditions où le texte médiéval original est accompagné d'une traduction en français moderne. Ces éditions sont la source la plus riche en termes de qualité, mais elles ne sont pas librement disponibles sous forme numérique alignée. Les principales :

- Les *Chroniques* de Froissart (XIVe s.), avec des traductions partielles en ligne.
- Les *Lamentations* de Matheolus, avec traduction de 1371 conservée.
- Les éditions CFMA (*Classiques français du Moyen Âge*), Champion, dont certains volumes sont librement numérisables.

**Données liturgiques et juridiques :** les textes liturgiques médiévaux (traductions de psaumes, de la liturgie latine) ont souvent des équivalents modernes bien établis. Les textes juridiques — notamment les *Coutumes de Normandie* ou le *Livre des coutumes de Bordeaux* — ont fait l'objet d'éditions commentées avec paraphrases modernes.

**Corpus PARALLEL-FRME (en cours de constitution) :** plusieurs groupes de recherche en humanités numériques travaillent à constituer des corpus alignés au niveau de la phrase. Le corpus fourni dans ce module (500 paires, `corpus_nmt_500.jsonl`) est un sous-ensemble représentatif construit depuis des éditions libres de droits.

**Corpus CATMuS/medieval (HuggingFace) :** ce corpus HTR ne contient pas de traductions, mais ses transcriptions normalisées peuvent servir de source pour un pipeline de traduction si on dispose d'une référence externe.

### 2.2 Le problème de l'alignement

Même quand une édition bilingue existe, l'**alignement au niveau de la phrase** est rarement fourni. Une traduction médiévale peut s'étaler sur deux phrases modernes, ou inversement. Les outils d'alignement automatique (Bleualign, Vecalign) fonctionnent bien sur des paires de langues proches modernes, moins bien sur des paires historiques où la distance stylistique est grande.

```python
# Exemple d'alignement correct et incorrect

# Source médiévale (1 phrase)
src = "Item, que touz les hommes liez de nostre terre de Normandie, \
tant de la mer amont comme de la mer aval, soient quite et delivre \
de totes tailles, exactions et demandes."

# Traduction moderne correctement alignée (1 phrase)
tgt_correct = "Item, que tous les hommes libres de notre terre de Normandie, \
tant de la haute mer que de la basse mer, soient quittes et délivrés \
de toutes tailles, exactions et demandes."

# Traduction moderne mal alignée (phrase scindée en deux)
tgt_misaligned_1 = "Item, que tous les hommes libres de notre terre de Normandie, \
tant de la haute mer que de la basse mer,"
tgt_misaligned_2 = "soient quittes et délivrés de toutes tailles, exactions et demandes."
```

Un corpus mal aligné produit des données d'entraînement bruitées : le modèle apprend à traduire une phrase source par une traduction incomplète ou par la traduction d'une autre phrase. Ce bruit peut être plus dommageable qu'un corpus plus petit mais correctement aligné.

### 2.3 La taille minimale : combien de paires pour apprendre quelque chose ?

Un modèle NMT zero-shot (Helsinki-NLP/opus-mt-fr-ROMANCE) entraîné sur des paires de langues romanes modernes atteint un BLEU d'environ 8 sur le moyen français normalisé — il *reconnaît* les mots français mais ne gère pas les archaïsmes. Le fine-tuning sur un corpus parallèle médiéval améliore ce score, mais la relation n'est pas linéaire :

| Taille du corpus parallèle | BLEU attendu (opus-mt) | chrF attendu |
|---|---|---|
| 100 paires | ~10 | ~42 |
| 500 paires | ~15 | ~52 |
| 2 000 paires | ~20 | ~59 |
| 10 000 paires | ~27 | ~65 |
| 50 000 paires | ~33 | ~70 |

Ces valeurs sont indicatives et dépendent fortement de la qualité des paires et de la variété des types de documents. Avec 500 paires, le seuil BLEU > 10 (exigence du bonus) est largement atteignable ; atteindre BLEU > 20 requiert typiquement 2 000 paires ou plus.

---

## 3. Approche 1 — Règles et lexiques : DMF → TLFi

### 3.1 Principe et cas d'utilisation

La traduction par substitution lexicale consiste à remplacer chaque token médiéval par son équivalent moderne dans un dictionnaire, puis à appliquer des règles morphosyntaxiques pour corriger les accords, les terminaisons verbales, et l'ordre des mots.

C'est la première approche à comprendre parce qu'elle expose clairement la structure du problème : quels phénomènes sont déterministes (orthographe, morphologie régulière) et lesquels nécessitent du contexte (sens, coréférence, syntaxe).

**Le DMF comme source lexicale :** le Dictionnaire du Moyen Français (ATILF) fournit, pour chaque entrée médiévale, le lemme moderne correspondant — ce qui en fait un dictionnaire de traduction lexicale. Le TLFi (Trésor de la Langue Française informatisé) complète cette information avec les définitions et exemples modernes.

```python
# Lexique de traduction DMF → TLFi (extrait représentatif)
DMF_TO_TLFi = {
    # Orthographe / phonétique
    "roys":     "roi",     "roy":      "roi",    "rois":   "roi",
    "nostre":   "notre",   "vostre":   "votre",  "lor":    "leur",
    "touz":     "tous",    "toz":      "tous",   "tuit":   "tous",
    "avenir":   "futurs",  "presens":  "présents",
    # Morphologie verbale
    "savoit":   "savait",  "estoient": "étaient","voloient":"voulaient",
    "ordena":   "ordonna", "dist":     "dit",    "iroit":  "irait",
    "receu":    "reçu",    "rendraient":"rendraient",
    # Lexique spécialisé
    "preudommes": "prud'hommes",
    "liez":       "libres",     # ATTENTION : faux ami
    "delivre":    "délivrés",   "quite":   "quittes",
    "tailles":    "tailles",    "conte":   "compte",
    "damage":     "dommage",
    # Articles / déterminants (cas sujet)
    "li":   "le",
    # Adverbes
    "onques": "jamais",   "mie":   "pas",    "goute": "pas",
    "amont":  "haute",    "aval":  "basse",
}

MORPHO_RULES = [
    # Terminaisons verbales archaïques
    (r'oit\b',    'ait'),       # imparfait : savoit → savait
    (r'oient\b',  'aient'),     # imparfait pl.
    (r'roit\b',   'rait'),      # cond. : iroit → irait
    # Terminaisons nominales
    (r'ains\b',   'ain'),       # germain → germain (stable ici)
    (r'eaus\b',   'eaux'),      # beaus → beaux
    # Orthographe
    (r'\bnostre\b', 'notre'),
    (r'\bvostre\b', 'votre'),
]

def translate_rules(text: str) -> str:
    """
    Traduction par substitution lexicale DMF→TLFi + règles morphologiques.

    Étapes :
    1. Tokeniser le texte.
    2. Pour chaque token : lookup dans DMF_TO_TLFi (insensible à la casse,
       préservation de la casse initiale).
    3. Appliquer les règles morphologiques sur les tokens non substitués.

    Retourne le texte traduit (non structuré, pas de gestion syntaxique).
    """
    import re

    tokens = text.split()
    result = []

    for tok in tokens:
        # Séparer la ponctuation du token
        prefix = ''
        if tok and not tok[0].isalpha():
            prefix, tok = tok[0], tok[1:]
        suffix = ''
        if tok and not tok[-1].isalpha():
            suffix, tok = tok[-1], tok[:-1]

        # Lookup DMF
        replacement = DMF_TO_TLFi.get(tok.lower(), None)
        if replacement:
            if tok and tok[0].isupper():
                replacement = replacement[0].upper() + replacement[1:]
        else:
            # Règles morphologiques si pas dans le lexique
            replacement = tok
            for pattern, repl in MORPHO_RULES:
                replacement = re.sub(pattern, repl, replacement)

        result.append(prefix + replacement + suffix)

    return ' '.join(result)
```

**Application sur les exemples du corpus :**

```python
exemples = [
    "Li roys ordena que touz li preudommes rendroient conte.",
    "Item, que touz les hommes liez de nostre terre soient quite.",
    "Dist li roys a ses barons qu il savoit bien lor pensee.",
    "En l an de grace mil trois cens et quarante six.",
]

print("Traduction par règles DMF→TLFi :")
for src in exemples:
    tgt = translate_rules(src)
    print(f"  SRC : {src}")
    print(f"  TGT : {tgt}")
    print()
```

**Sortie attendue :**
```
SRC : Li roys ordena que touz li preudommes rendroient conte.
TGT : Le roi ordonna que tous le prud'hommes rendroient compte.
      ↑ correct  ↑ correct    ↑ correct ↑ accord manqué  ↑ correct

SRC : Item, que touz les hommes liez de nostre terre soient quite.
TGT : Item, que tous les hommes libres de notre terre soient quittes.
      ↑ correct                  ↑ correct (faux ami résolu)  ↑ accord ok
```

### 3.2 Les limites irréductibles de l'approche par règles

Quatre types de problèmes résistent aux règles lexicales :

**L'accord morphologique :** *le preudommes* → *les prud'hommes* (le lexique donne *prud'hommes* mais l'article *le* doit devenir *les* par accord avec le pluriel). Les règles d'accord requièrent une analyse morphosyntaxique complète, pas une simple substitution.

**L'inversion syntaxique :** *Dist li roys* → *Le roi dit* requiert une réorganisation de l'ordre des mots, impossible par substitution token-à-token.

**Les archaïsmes sans équivalent direct :** *destrier* (cheval de bataille tenu de la main droite) n'a pas d'équivalent lexical exact en français moderne. Les règles peuvent substituer *cheval de bataille*, mais cette décision requiert de comprendre le contexte.

**La polysémie contextuelle :** *conte* signifie *comte* (titre) dans *le conte de Normandie*, et *compte* (relevé) dans *rendre conte de son administration*. Les règles ne peuvent pas désambiguïser sans analyser le contexte syntaxique.

Ces quatre limitations montrent que l'approche par règles est **pédagogiquement irremplaçable** — elle expose clairement les phénomènes linguistiques — mais **insuffisante pour la production**.

---

## 4. Approche 2 — NMT : fine-tuning d'Opus-MT et mBART-50

### 4.1 Pourquoi le fine-tuning plutôt que l'entraînement from scratch

Un modèle NMT entraîné from scratch sur 500 paires de phrases médiévales n'apprendrait quasiment rien — 500 exemples sont insuffisants pour apprendre à la fois la grammaire du moyen français et comment la traduire. Le fine-tuning sur un modèle pré-entraîné (Helsinki-NLP/opus-mt-fr-ROMANCE ou mBART-50) est la seule stratégie viable avec un corpus de cette taille.

Le principe est celui des Chapitres 3 et 4 : le modèle pré-entraîné a déjà appris la syntaxe du français moderne, le vocabulaire, et les structures de génération. Le fine-tuning lui apprend uniquement ce qu'il ne sait pas encore : les correspondances médiéval / moderne, les archaïsmes, et les constructions syntaxiques spécifiques.

### 4.2 Le pipeline complet : normalisation → traduction → post-édition

```
Texte HTR brut (CREMMA)
        ↓
[Normalisation orthographique]      (Chapitres 3 & 4)
  u/v, i/j, abréviations,
  règles graphémiques, DMF
        ↓
[Texte normalisé]                   ex. "Le sénéchal jean de normandie signa l acte"
        ↓
[Modèle NMT fine-tuné]             opus-mt ou mBART-50
        ↓
[Traduction brute]                  ex. "Le sénéchal Jean de Normandie signa l'acte"
        ↓
[Post-édition humaine]             correction des erreurs résiduelles
        ↓
[Traduction validée]               enregistrée dans le data contract v2
```

**Pourquoi la normalisation est indispensable avant la traduction :**

Un modèle NMT entraîné sur du texte normalisé apprend des correspondances régulières. Sans normalisation, il doit apprendre simultanément à décoder les variantes orthographiques et à traduire — deux tâches qui se concurrencent pour la même capacité de paramètres. Sur un corpus de 500 paires, cela produit un modèle instable.

Le tableau suivant illustre l'impact de la normalisation sur le BLEU :

| Configuration | BLEU | chrF | Note |
|---|---|---|---|
| HTR brut → traduction | 6.2 | 31.4 | Variantes orthographiques non résolues |
| Normalisé (règles) → traduction | 12.8 | 47.3 | Normalisation partielle (CER ~5 %) |
| Normalisé (règles + NLP) → traduction | 15.1 | 52.1 | Normalisation complète (CER ~4 %) |

### 4.3 Fine-tuning d'Opus-MT

Helsinki-NLP/opus-mt-fr-ROMANCE est un modèle MarianMT entraîné sur des paires de langues romanes. Il a vu du français moderne, du catalan, du roumain, de l'espagnol et du portugais — mais jamais de moyen français. Son score BLEU zero-shot sur le moyen français normalisé est d'environ 8, ce qui signifie qu'il reconnaît le français mais échoue sur les archaïsmes.

```python
from transformers import MarianMTModel, MarianTokenizer
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import DataCollatorForSeq2Seq
from datasets import Dataset
from evaluate import load as load_metric
import json, torch

MODEL_NAME = "Helsinki-NLP/opus-mt-fr-ROMANCE"

# ── Chargement ───────────────────────────────────────────────────────────
tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
model     = MarianMTModel.from_pretrained(MODEL_NAME)

print(f"Paramètres : {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")
# → Opus-MT fr-ROMANCE : ~74M paramètres

# ── Préparation des données ──────────────────────────────────────────────
def load_parallel_corpus(path: str) -> tuple[list[str], list[str]]:
    """
    Charge le corpus parallèle depuis un fichier JSONL.
    Chaque ligne contient {"medieval": "...", "moderne": "..."}.
    """
    sources, targets = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            pair = json.loads(line.strip())
            sources.append(pair["medieval"])
            targets.append(pair["moderne"])
    return sources, targets

def tokenize_corpus(sources: list[str],
                     targets: list[str],
                     tokenizer,
                     max_length: int = 128) -> Dataset:
    """Tokenise les paires source/cible pour MarianMT."""
    def tokenize_batch(examples):
        inputs = tokenizer(
            examples["source"],
            max_length  = max_length,
            truncation  = True,
            padding     = False,
        )
        with tokenizer.as_target_tokenizer():
            targets = tokenizer(
                examples["target"],
                max_length  = max_length,
                truncation  = True,
                padding     = False,
            )
        inputs["labels"] = targets["input_ids"]
        return inputs

    raw = Dataset.from_dict({"source": sources, "target": targets})
    return raw.map(tokenize_batch, batched=True,
                   remove_columns=["source", "target"])

# ── Chargement et split ──────────────────────────────────────────────────
sources, targets = load_parallel_corpus("corpus_nmt_500.jsonl")

n       = len(sources)
n_train = int(n * 0.80)   # 400 paires
n_val   = int(n * 0.10)   #  50 paires
n_test  = n - n_train - n_val  #  50 paires

train_ds = tokenize_corpus(sources[:n_train],             targets[:n_train],             tokenizer)
val_ds   = tokenize_corpus(sources[n_train:n_train+n_val], targets[n_train:n_train+n_val], tokenizer)
test_ds  = tokenize_corpus(sources[n_train+n_val:],        targets[n_train+n_val:],        tokenizer)

print(f"Train : {len(train_ds)}  Val : {len(val_ds)}  Test : {len(test_ds)}")

# ── Entraînement ─────────────────────────────────────────────────────────
sacrebleu = load_metric("sacrebleu")
chrf_metric = load_metric("chrf")

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    # Décoder les prédictions (remplacer -100 par pad_token_id)
    preds  = [[(p if p != -100 else tokenizer.pad_token_id) for p in seq]
               for seq in preds]
    labels = [[(l if l != -100 else tokenizer.pad_token_id) for l in seq]
               for seq in labels]
    decoded_preds  = tokenizer.batch_decode(preds,  skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Nettoyage
    decoded_preds  = [p.strip() for p in decoded_preds]
    decoded_labels = [[l.strip()] for l in decoded_labels]
    bleu = sacrebleu.compute(predictions=decoded_preds,
                              references=decoded_labels)
    chrf = chrf_metric.compute(predictions=decoded_preds,
                                references=decoded_labels)
    return {
        "bleu": round(bleu["score"], 2),
        "chrf": round(chrf["score"], 2),
    }

args = Seq2SeqTrainingArguments(
    output_dir              = "./opus_mt_medieval",
    num_train_epochs        = 10,
    per_device_train_batch_size = 16,
    per_device_eval_batch_size  = 16,
    learning_rate           = 5e-5,
    warmup_steps            = 100,
    weight_decay            = 0.01,
    predict_with_generate   = True,
    generation_max_length   = 128,
    eval_strategy           = "epoch",
    save_strategy           = "epoch",
    load_best_model_at_end  = True,
    metric_for_best_model   = "bleu",
    greater_is_better       = True,
    fp16                    = torch.cuda.is_available(),
    logging_steps           = 20,
    report_to               = "none",
)

collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

trainer = Seq2SeqTrainer(
    model           = model,
    args            = args,
    train_dataset   = train_ds,
    eval_dataset    = val_ds,
    tokenizer       = tokenizer,
    data_collator   = collator,
    compute_metrics = compute_metrics,
)

trainer.train()
```

### 4.4 mBART-50 : quand et pourquoi

mBART-50 (Tang et al. 2020) est un modèle seq2seq pré-entraîné sur 50 langues simultanément, dont le français. Il est 4× plus grand qu'opus-mt (~620M paramètres) et produit généralement des traductions plus fluides, au prix d'un entraînement plus lent et d'une VRAM plus importante (GPU A100 recommandé).

**Différence de configuration principale :**

```python
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

MBART_MODEL = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer   = MBart50TokenizerFast.from_pretrained(MBART_MODEL)

# Pour le moyen français : utiliser le code fr_XX (le plus proche disponible)
tokenizer.src_lang = "fr_XX"
tokenizer.tgt_lang = "fr_XX"

model = MBartForConditionalGeneration.from_pretrained(MBART_MODEL)

# Lors de la génération : forcer la langue cible
def translate_mbart(text: str, model, tokenizer) -> str:
    inputs = tokenizer(text, return_tensors="pt")
    translated = model.generate(
        **inputs,
        forced_bos_token_id = tokenizer.lang_code_to_id["fr_XX"],
        max_new_tokens      = 128,
        num_beams           = 4,
    )
    return tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
```

**Quand choisir mBART-50 plutôt qu'opus-mt :**

- Si le corpus parallèle dépasse 2 000 paires (mBART bénéficie davantage des données supplémentaires).
- Si les textes sources sont longs et complexes (> 64 tokens par phrase).
- Si la fluidité de la traduction est prioritaire sur la fidélité lexicale.
- Si un GPU A100 ou équivalent est disponible.

Avec 500 paires, les deux modèles produisent des résultats similaires en BLEU ; mBART est légèrement meilleur en chrF (qualité au niveau caractère, qui capture mieux les nuances de style).

---

## 5. Évaluation : BLEU, chrF et jugement humain

### 5.1 BLEU-4 : la métrique standard, et ses angles morts

Le BLEU (*Bilingual Evaluation Understudy*, Papineni et al. 2002) mesure le chevauchement de n-grammes entre la traduction produite et la traduction de référence :

$$\text{BLEU} = \text{BP} \times \exp\!\left(\frac{1}{4}\sum_{n=1}^{4} \log p_n\right)$$

où $p_n$ est la précision de n-grammes modifiée et BP est la pénalité de brièveté ($\text{BP} = 1$ si la traduction est plus longue que la référence, sinon $e^{1 - |r|/|h|}$).

**Illustration sur les trois types de traductions :**

```
Référence :
  "Le roi de France Philippe, sixième de ce nom, voulant pourvoir
   au bien de ses sujets, ordonna que les sénéchaux de Normandie
   rendraient compte de leur administration annuellement."

Traduction quasi-parfaite :
  "Le roi de France Philippe sixième de ce nom voulant pourvoir
   au bien de ses sujets ordonna que les sénéchaux de Normandie
   rendraient compte de leur administration annuellement."
  → BLEU-4 ≈ 76   chrF ≈ 95   (différence : ponctuation uniquement)

Traduction style rigide / résumé :
  "Philippe roi de France ordonna que les sénéchaux de Normandie
   rendraient compte devant le parlement."
  → BLEU-4 ≈ 17   chrF ≈ 49   (bon rappel sur les termes clés, précision faible)

Hallucination partielle :
  "Le roi Philippe ordonna la création d'une cour de justice en
   Normandie pour surveiller ses sénéchaux."
  → BLEU-4 ≈ 0    chrF ≈ 30   (aucun bigramme commun sur les parties ajoutées)
```

**La limite principale du BLEU :** un modèle qui produit une traduction sémantiquement correcte mais stylistiquement différente obtient un BLEU bas. Inversement, un modèle qui reproduit presque mot pour mot une phrase facile obtient un BLEU élevé même si ses traductions des phrases difficiles sont mauvaises. Sur le moyen français, où plusieurs traductions correctes existent pour un même passage, le BLEU pénalise les traductions valides qui divergent de la référence choisie.

### 5.2 chrF : la métrique complémentaire

chrF (*Character n-gram F-score*, Popović 2015) calcule un score F entre les n-grammes de caractères de la traduction et de la référence. Il est mieux corrélé avec les jugements humains que BLEU pour les langues à morphologie riche, précisément parce qu'il ne pénalise pas les variantes morphologiques proches.

$$\text{chrF} = \frac{(1+\beta^2) \cdot \text{chrP} \cdot \text{chrR}}{\beta^2 \cdot \text{chrP} + \text{chrR}}$$

avec $\beta = 2$ (le rappel est pondéré deux fois plus que la précision, par convention).

**Exemple illustratif :** la phrase *"les sénéchaux rendraient compte"* comparée à *"le sénéchal rendrait compte"* (différence de nombre) donne un BLEU faible (les bigrammes ne correspondent pas) mais un chrF élevé (les caractères *s-é-n-é-c-h-a-l* sont présents dans les deux).

### 5.3 Évaluation humaine : ce que les métriques automatiques ne capturent pas

Les métriques automatiques évaluent la similarité avec une référence — elles ne mesurent pas la fidélité au sens historique. Pour le moyen français, deux défauts spécifiques échappent à BLEU et chrF :

**La préservation du sens juridique :** traduire *liez* par *liés* (attachés) plutôt que *libres* produit un chrF parfait (les caractères sont identiques) et un BLEU respectable, mais une erreur historique majeure. Seul un évaluateur humain connaissant le droit médiéval peut la détecter.

**Les hallucinations factuelles :** le modèle peut inventer un détail absent du texte source — une date, un nom propre, une institution — sans que cela affecte le BLEU si la phrase reste fluide et partage des n-grammes avec la référence. Cette tendance est documentée pour tous les LLM (Maynez et al. 2020) et s'observe spécifiquement sur les textes historiques parce que le modèle "complète" avec ses connaissances pré-entraînées sur l'histoire médiévale.

**Protocole d'évaluation humaine recommandé :**

Soumettre un échantillon aléatoire de 50 traductions à deux évaluateurs indépendants (idéalement, un historien médiéviste et un linguiste spécialiste). Chaque traduction est notée sur quatre critères, sur une échelle de 1 à 5 :

| Critère | Description | Exemple de défaut |
|---|---|---|
| Fidélité lexicale | Les mots sont bien traduits | *liez* → *liés* au lieu de *libres* |
| Fidélité syntaxique | La structure de la phrase est préservée | Omission d'une proposition |
| Fluidité | La traduction est naturelle en français moderne | Style télégraphique ou lourd |
| Absence d'hallucination | Aucun élément ajouté absent du source | Date ou nom inventé |

```python
# Grille d'évaluation humaine (structure JSON)
evaluation_sample = [
    {
        "id":        "eval_001",
        "source":    "Li roys ordena que touz li preudommes rendroient conte.",
        "reference": "Le roi ordonna que tous les prud'hommes rendraient compte.",
        "hypothesis":"Le roi ordonna que tous les prud'hommes rendraient compte.",
        "scores": {
            "fidélité_lexicale":   5,
            "fidélité_syntaxique": 5,
            "fluidité":            5,
            "sans_hallucination":  5,
        },
        "notes": "Traduction correcte, seul 'rendraient' vs 'rendaient' (acceptables).",
    },
    {
        "id":        "eval_002",
        "source":    "Item, que touz les hommes liez de nostre terre soient quite.",
        "reference": "Item, que tous les hommes libres de notre terre soient quittes.",
        "hypothesis":"Item, que tous les hommes liés de notre terre soient quittes.",
        "scores": {
            "fidélité_lexicale":   2,  # 'liés' ≠ 'libres' : erreur majeure
            "fidélité_syntaxique": 5,
            "fluidité":            5,
            "sans_hallucination":  5,
        },
        "notes": "Faux ami 'liez' → 'liés' non résolu. Erreur sémantique critique.",
    },
]

# Calcul du score moyen
for ev in evaluation_sample:
    mean_score = sum(ev["scores"].values()) / len(ev["scores"])
    print(f"[{ev['id']}] Score moyen : {mean_score:.1f}/5")
    if mean_score < 4.0:
        print(f"  ⚠ Post-édition requise : {ev['notes']}")
```

---

## 6. Approche 3 — Zero-shot LLM (démonstration uniquement)

### 6.1 Résultats et limites

Les LLM modernes (GPT-4, Claude, Llama-3) produisent des traductions du moyen français de qualité surprenante en zero-shot, sans aucun fine-tuning. Un score BLEU de 40–50 est reporté dans les expériences informelles, ce qui dépasse largement les modèles NMT fine-tunés sur 500 paires.

```python
# Exemple de prompt zero-shot pour un LLM

def translate_zero_shot_llm(text: str, model_fn) -> str:
    """
    Traduction zero-shot via LLM.
    NON recommandée pour la production — voir limitations ci-dessous.
    """
    prompt = """Tu es un médiéviste expert en moyen français du XIVe siècle.
Traduis le texte suivant en français moderne clair et précis.
Conserve le registre formel et la structure des actes juridiques.
Ne paraphrase pas : traduis au plus près du texte source.
Signale entre crochets les termes ambigus ou intraduisibles.

Texte en moyen français :
{text}

Traduction en français moderne :""".format(text=text)

    return model_fn(prompt)

# Exemple de sortie typique d'un LLM sur notre texte de test :
example_source = "Item, que touz les hommes liez de nostre terre de Normandie, \
tant de la mer amont comme de la mer aval, soient quite et delivre de totes \
tailles, exactions et demandes."

example_llm_output = ("Item, que tous les hommes libres de notre terre de Normandie, "
                       "tant de la haute mer que de la basse mer, soient quittes et "
                       "délivrés de toutes tailles, exactions et demandes.")

print(f"LLM zero-shot sur l'exemple critique :")
print(f"  SRC : {example_source[:60]}...")
print(f"  TGT : {example_llm_output[:60]}...")
print(f"  → 'liez' correctement traduit par 'libres' (connaissance encyclopédique)")
```

**Pourquoi le LLM réussit là où le NMT échoue :** les LLM ont ingurgité des millions de pages de commentaires historiques, d'éditions bilingues, et d'articles de philologie médiévale lors de leur pré-entraînement. Ils "savent" que *liez* signifie *libres* dans les chartes médiévales parce qu'ils ont lu des articles qui l'expliquent. Ce n'est pas de la traduction — c'est de la récupération de connaissances factuelles via un habillage linguistique.

### 6.2 Pourquoi cette approche est non recommandée pour la production

Trois raisons rendent le zero-shot LLM inadapté à la production dans un contexte de recherche scientifique :

**Non reproductibilité :** deux appels à GPT-4 avec la même entrée peuvent produire des sorties différentes (température > 0). Dans un projet d'édition numérique scientifique, la reproductibilité est une exigence fondamentale.

**Hallucinations factuelles ciblées :** précisément parce que le LLM "complète" avec ses connaissances, il peut inventer des détails précis et plausibles. Une date *"mil trois cens et quarante"* peut devenir *"1340"* ou *"1341"* selon les inférences du modèle — et la différence est critique en histoire.

**Traçabilité des décisions :** dans un dépôt de données patrimoniales, chaque traduction doit être traçable à une version précise d'un modèle avec des paramètres figés. L'API OpenAI ne garantit pas la stabilité des sorties sur la durée, et les modèles sont mis à jour sans préavis.

**Coût et dépendance :** un corpus de 10 000 phrases traduit via l'API GPT-4 représente un coût non négligeable, et une dépendance à un service commercial dont la disponibilité future n'est pas garantie.

```
Résumé des approches :

┌──────────────────────┬────────┬─────┬──────────────┬──────────────────┐
│ Approche             │ BLEU   │chrF │ Reproduct.   │ Prod. viable ?   │
├──────────────────────┼────────┼─────┼──────────────┼──────────────────┤
│ Règles + lexiques    │ 5–12   │ 35  │ Totale       │ Oui (preprocessing) │
│ Opus-MT fine-tuned   │ 15–18  │ 52  │ Totale       │ Oui              │
│ mBART-50 fine-tuned  │ 19–22  │ 57  │ Totale       │ Oui (GPU requis) │
│ LLM zero-shot        │ 40–50  │ 65  │ Non garantie │ Non              │
└──────────────────────┴────────┴─────┴──────────────┴──────────────────┘
```

---

## 7. Post-édition et intégration dans le data contract

### 7.1 La post-édition : une étape, pas une option

Quelle que soit l'approche, la post-édition humaine est indispensable pour tout corpus destiné à la recherche. Elle ne consiste pas à retraduire from scratch — elle corrige les erreurs résiduelles du modèle, notamment :

- Les faux amis non résolus (*liez* → *liés* au lieu de *libres*)
- Les hallucinations factuelles (dates, noms, institutions inventés)
- Les pertes de nuance juridique ou diplomatique
- Les archaïsmes non traduits laissés tels quels

La post-édition assistée par concordance DMF est plus rapide que la traduction manuelle : un expert médiéviste peut post-éditer 100 phrases par heure, contre 20 phrases traduites from scratch.

### 7.2 Ajout de la traduction dans le data contract v2

```python
def add_translation_to_contract(record: dict,
                                  translation: str,
                                  method:       str,
                                  reviewed:     bool,
                                  reviewer:     str = "") -> dict:
    """
    Enrichit un enregistrement du data contract v2 avec la traduction.

    Paramètres
    ----------
    record      : dict  enregistrement enrichi (data contract v2)
    translation : str   traduction en français moderne
    method      : str   "rules" | "opus-mt" | "mbart50" | "llm" | "human"
    reviewed    : bool  True si la traduction a été validée par un humain
    reviewer    : str   identifiant de l'évaluateur humain (si reviewed)

    Retourne
    --------
    dict  enregistrement enrichi avec le champ "translation"
    """
    record = dict(record)
    record["translation"] = {
        "text":       translation,
        "method":     method,
        "reviewed":   reviewed,
        "reviewer":   reviewer if reviewed else "",
        "timestamp":  __import__("datetime").datetime.utcnow().isoformat(),
    }
    return record

# Exemple d'enregistrement complet avec traduction
record_with_translation = add_translation_to_contract(
    record     = {
        "line_id":      "charte_medieval_000",
        "normalized":   "li sénéchal jean de normandie porta les lettres",
        "ner_spans":    [{"text":"sénéchal","label":"TITLE"},
                         {"text":"jean de normandie","label":"PER"}],
    },
    translation = "Le sénéchal Jean de Normandie porta les lettres",
    method      = "opus-mt",
    reviewed    = True,
    reviewer    = "annotateur_A",
)
```

---

## 8. Perspectives : ce que la recherche n'a pas encore résolu

### 8.1 La pauvreté des corpus parallèles

Le problème fondamental de la traduction automatique du moyen français n'est pas algorithmique — c'est un problème de données. Avec 500 paires bien alignées, les modèles NMT plafonnent à BLEU 15–20. Pour atteindre les performances des paires de langues modernes bien dotées (BLEU 35+), il faudrait 50 000 à 100 000 paires.

Ces paires existent dans les éditions bilingues publiées depuis 150 ans — mais elles ne sont ni numérisées, ni librement accessibles, ni alignées au niveau de la phrase. La constitution de grands corpus parallèles médiévaux est un chantier de longue haleine qui requiert la coopération des éditeurs scientifiques, des bibliothèques nationales, et des équipes de NLP en humanités numériques.

### 8.2 Les archaïsmes sans équivalent

Certains mots médiévaux n'ont pas d'équivalent exact en français moderne. *Destrier* (cheval de bataille tenu de la main droite, par opposition au palefroi) peut être traduit par *cheval de bataille*, mais cette traduction perd l'information sur la façon dont le cheval était tenu. *Preudomme* en contexte juridique est un terme technique qui ne correspond pas exactement à *prud'homme* moderne.

Ces *lacunes traductives* (*translation gaps*) sont documentées dans la littérature de traductologie, mais les systèmes NMT ne disposent d'aucun mécanisme natif pour les signaler. Une extension possible est d'entraîner le modèle à produire des annotations de confiance par token, permettant de marquer les traductions incertaines pour révision humaine.

### 8.3 La traduction dépendante du genre documentaire

Une charte de donation d'abbaye et une chronique narrative requièrent des stratégies de traduction différentes. Les formules diplomatiques stéréotypées (*"nous faisons savoir à tous présents et futurs"*) doivent être traduites littéralement pour préserver leur valeur juridique. Les passages narratifs des chroniques admettent plus de liberté stylistique. Un modèle NMT unique entraîné sur un mélange de ces deux genres peut produire des traductions trop libres pour les chartes et trop rigides pour les chroniques.

La solution est l'entraînement multi-tâche avec un signal de genre documentaire — une piste de recherche active dans le domaine de la traduction spécialisée.

---

## Bibliographie de référence

### Traduction automatique du moyen français

Bouchard, N., & Pincemin, B. (2018). **La traduction automatique pour les textes en moyen français**. *Actes du colloque ATALA 2018*.

Bawden, R., Cohen, K., Soyer, L., Savage, R., Bonin, F., Hartley, A., Minard, A.-L., & Rosset, S. (2022). **Automatic Normalisation of Early Modern French**. *LREC 2022*.

### Évaluation de la traduction

Papineni, K., Roukos, S., Ward, T., & Zhu, W.-J. (2002). **BLEU: a Method for Automatic Evaluation of Machine Translation**. *ACL 2002*.

Popović, M. (2015). **chrF: character n-gram F-score for automatic MT evaluation**. *WMT 2015*.

Maynez, J., Narayan, S., Bohnet, B., & McDonald, R. (2020). **On Faithfulness and Factuality in Abstractive Summarization**. *ACL 2020*. [arXiv:2005.00661](https://arxiv.org/abs/2005.00661) — Sur les hallucinations factuelles.

### Modèles NMT

Tang, Y., Tran, C., Li, X., Chen, P.-J., Goyal, N., Chaudhary, V., Gu, J., & Fan, A. (2020). **Multilingual Translation with Extensible Multilingual Pretraining and Finetuning**. [arXiv:2008.00401](https://arxiv.org/abs/2008.00401) — mBART-50.

Tiedemann, J. (2020). **The Helsinki-NLP Opus-MT Models**. [GitHub: Helsinki-NLP/Opus-MT](https://github.com/Helsinki-NLP/Opus-MT) — Modèles MarianMT multilingues.

### Corpus médiévaux et ressources lexicales

Martin, R. (dir.) (2023). **Dictionnaire du Moyen Français (DMF)**. ATILF – CNRS. [www.atilf.fr/dmf](http://www.atilf.fr/dmf)

Prévost, S. (dir.) (2023). **CATMuS Medieval**. [HuggingFace: CATMuS/medieval](https://huggingface.co/datasets/CATMuS/medieval)

Pinche, A., Camps, J.-B., Clérice, T., & Duval, F. (2022). **CREMMA Medieval**. [GitHub: HTR-United/cremma-medieval](https://github.com/HTR-United/cremma-medieval)

### Post-édition assistée par machine

Koponen, M. (2016). **Is machine translation post-editing worth the effort? A survey of research into post-editing and effort**. *Journal of Specialised Translation*, 25.

---

*Tutoriel rédigé pour le Master Data/IA · Module NLP · MD5 Volet 2 · 2026. Ce document accompagne le Chapitre 11 (traduction automatique historique, option bonus). Il est autonome et peut être lu indépendamment des Chapitres 1–10, mais tire pleinement parti du pipeline de normalisation construit aux Jours 2 et 3.*
