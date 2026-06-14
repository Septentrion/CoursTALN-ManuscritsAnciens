"""
TP Guidé — Pipeline de normalisation orthographique du moyen français
         + Fine-tuning LoRA vs QLoRA sur mT5
Chapitres 3 & 4 — Module NLP · Master Data/IA · MD5 Volet 2 · 2026
──────────────────────────────────────────────────────────────────────
Pipeline implémenté :

  Texte brut HTR
       ↓
  [Étape 1]  Normalisation Unicode
       ↓
  [Étape 2]  Règles graphiques (u/v, i/j, terminaisons, graphies) → CONVENTIONS_NLP.md
       ↓
  [Étape 3]  Résolution abréviations déterministes → table_abreviations.json
       ↓
  [Étape 4]  Consultation cache DMF (formes connues)
       ↓
  [Étape 5]  Composition : pipeline complet par ligne
       ↓
  [Étape 6]  Évaluation (CER, BLEU-4, token accuracy)
       ↓
  [Étape 7]  Split train/val/test + journal d'expériences
       ↓
  [Étape 8]  Bilan mémoire LoRA vs QLoRA
       ↓
  [Étape 9]  Fine-tuning LoRA (CPU/GPU)
       ↓
  [Étape 10] Fine-tuning QLoRA (GPU recommandé, fallback CPU documenté)
       ↓
  [Étape 11] Tableau d'ablation comparatif

Durée estimée : 3 h 30
Livrables :
  - CONVENTIONS_NLP.md
  - table_abreviations.json
  - experiments/journal.jsonl
  - checkpoints LoRA et QLoRA (Étapes 9–10, GPU requis)

Instructions générales
──────────────────────
Les cellules  # TODO  contiennent des squelettes à compléter.
Les cellules  # FOURNI  sont à exécuter telles quelles.
Même convention que les TPs précédents : ne pas modifier les signatures.
"""

# %% [markdown]
# # TP — Pipeline de normalisation orthographique + LoRA vs QLoRA
#
# Ce TP réunit les Chapitres 3 et 4 en une chaîne de traitement continue.
# Vous partirez de transcriptions HTR brutes en moyen français,
# les normaliserez couche par couche (règles, abréviations, DMF),
# puis fine-tunerez mT5-small avec LoRA et QLoRA sur les paires
# ainsi produites.
#
# La comparaison LoRA / QLoRA est l'objectif central des Étapes 8 à 11 :
# vous mesurerez l'empreinte mémoire théorique, le CER de normalisation
# et le temps d'entraînement de chaque configuration.

# %% [markdown]
# ## Partie 0 — Imports et corpus de travail

# %% [FOURNI]
import re, json, unicodedata, math, hashlib, collections, random, os, datetime
import warnings
warnings.filterwarnings("ignore")

import editdistance          # pip install editdistance
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

print("Environnement prêt.")

# %% [markdown]
# ### Corpus de travail
#
# Quarante paires (transcription brute → forme normalisée) couvrant
# les phénomènes linguistiques traités dans le cours :
# alternances u/v et i/j, terminaisons variables, graphies médiévales,
# abréviations par tilde et signes souscrits, et formes rares
# nécessitant le DMF.
#
# En TP réel, remplacez ce corpus par vos paires issues de CREMMA.

# %% [FOURNI]
CORPUS_PAIRS = [
    # ── Alternances u/v ────────────────────────────────────────────────
    ("auoir este en son palays",        "avoir été en son palais"),
    ("sauoir la verite de l affaire",   "savoir la vérité de l affaire"),
    ("il deuoit rendre les lettres",    "il devoit rendre les lettres"),
    ("pour auancer le iugement",        "pour avancer le jugement"),
    ("la dame uouloit partir",          "la dame vouloit partir"),
    ("trouuer la verite est necessaire","trouver la vérité est nécessaire"),
    # ── Alternances i/j ────────────────────────────────────────────────
    ("iustice fut rendue au peuple",    "justice fut rendue au peuple"),
    ("le iour de mars li roys signa",   "le jour de mars li roys signa"),
    ("iusqu au terme fixe",             "jusqu au terme fixe"),
    ("le bailli rendit son iugement",   "le bailli rendit son jugement"),
    # ── Terminaisons -oit/-ait, -roit/-rait ────────────────────────────
    ("li chevaliers venoit de loin",    "li chevaliers venait de loin"),
    ("il pouroit savoir la verite",     "il pourrait savoir la vérité"),
    ("il faloit rendre les terres",     "il fallait rendre les terres"),
    ("la dame vouloit partir",          "la dame voulait partir"),
    # ── Graphies archaïques ─────────────────────────────────────────────
    ("li reis de france",               "li roi de france"),
    ("le chastel de gisors",            "le château de gisors"),
    ("au chastelain du louvre",         "au châtelain du louvre"),
    ("hault et puissant seigneur",      "haut et puissant seigneur"),
    ("au moys de mars li roys",         "au mois de mars le roi"),
    ("ledit seigneur porta les lettres","ledit seigneur porta les lettres"),
    # ── Abréviations : tildes de nasalité ──────────────────────────────
    ("au duc de norm~die",              "au duc de normandie"),
    ("le co~te de champagne",           "le conte de champagne"),
    ("q~ feist il li roys",             "que feist il li roys"),
    ("fra~ce et angleterre",            "france et angleterre"),
    ("te~ps de paix",                   "temps de paix"),
    ("co~bat singulier",                "combat singulier"),
    # ── Abréviations : signes souscrits ────────────────────────────────
    ("pñce de normandie",               "prince de normandie"),
    ("pour le paiement de vingt l~",    "pour le paiement de vingt livres"),
    ("li s^r de gisors signa",          "li seigneur de gisors signa"),
    ("en l an de grace m^e iean",       "en l an de grace messire jean"),
    # ── Formes rares : lookup DMF nécessaire ───────────────────────────
    ("ledit preudomme fu condamne",     "ledit prudhomme fu condamné"),
    ("la pucele estoit de grant beaute","la pucelle était de grande beauté"),
    ("li evesque de paris precha",      "l évêque de paris prêcha"),
    ("messire guillaumes escrivit",     "messire guillaume écrivit"),
    ("le seneschal porta les missives", "le sénéchal porta les missives"),
    ("la chartre fut seellee",          "la charte fut scellée"),
    ("les preudommes de la vile",       "les prudhommes de la ville"),
    ("en icele annee fu grant famine",  "en cette année fu grande famine"),
    ("li chevaliers arma son destrier", "le chevalier arma son destrier"),
    ("pour pouoir agir",                "pour pouvoir agir"),
]

print(f"Corpus : {len(CORPUS_PAIRS)} paires (brut → normalisé)")
print("Exemple :")
for src, tgt in CORPUS_PAIRS[:3]:
    print(f"  {src!r:45s} → {tgt!r}")

# %% [markdown]
# ## Étape 1 — Normalisation Unicode
#
# Avant toute règle linguistique, l'encodage doit être unifié.
# Deux problèmes coexistent dans les sorties HTR :
#
# - Le même caractère peut être encodé de plusieurs façons Unicode
#   (NFC vs NFD pour les lettres accentuées).
# - Certains modèles HTR retranscrivent les caractères médiévaux spéciaux
#   (p barré, i sans point, ligatures æ/œ) avec leurs codepoints Unicode
#   officiels, d'autres avec des approximations.
#
# **À vous de jouer.** Complétez `normalize_unicode`.

# %% [FOURNI]
MEDIEVAL_UNICODE_MAP = {
    'ꝑ': 'p',    # p barré (per/par/pro) — sera résolu à l'Étape 3
    'ꝓ': 'pro',  # p barré avec crochet
    'ꝕ': 'p',    # variante p barré
    'ȷ':  'j',   # j sans point
    'ı':  'i',   # i sans point (latin)
    'æ': 'ae',   # ligature ae
    'œ': 'oe',   # ligature oe
    '\u0304': '', # macron combinant (marque de nasale) → traité à l'Étape 3
}

# %% [TODO]
def normalize_unicode(text: str) -> str:
    """
    Étape 1 : normalise l'encodage Unicode d'une transcription HTR.

    Opérations dans l'ordre :
    1. Normalisation NFC : unicodedata.normalize('NFC', text)
       (unifie les représentations composées/décomposées des accents)
    2. Translittération des caractères médiévaux : parcourir
       MEDIEVAL_UNICODE_MAP et remplacer chaque clé par sa valeur.

    Paramètre
    ---------
    text : str  transcription brute

    Retourne
    --------
    str  texte avec encodage unifié

    Exemple
    -------
    normalize_unicode("ꝑar le ı")  →  "par le i"
    normalize_unicode("sœur")      →  "soeur"
    """
    # TODO : implémenter les deux opérations
    pass   # ← remplacer

# %% [markdown]
# **Validation 1**

# %% [FOURNI — validation]
assert normalize_unicode("ꝑar")  == "par",   "ꝑ (p barré) doit devenir p"
assert normalize_unicode("sœur") == "soeur", "œ doit devenir oe"
assert normalize_unicode("æge")  == "aege",  "æ doit devenir ae"
_nfc_test = "e\u0301"   # e + accent aigu combinant (NFD)
assert normalize_unicode(_nfc_test) == "é",  "NFD doit être normalisé en NFC"
print("Validation 1 : OK")

# %% [markdown]
# ## Étape 2 — Règles graphiques
#
# ### 2a — Alternance u/v
#
# En moyen français, *u* et *v* sont deux graphies de la même lettre.
# La règle phonologique est positionnelle :
#
# - *u* en position initiale devant voyelle → *v* (*uoir* → *voir*)
# - *u* après consonne devant voyelle → *v* (*sauoir* → *savoir*)
# - *u* après voyelle [a, e, i] devant voyelle → *v* (*auoir* → *avoir*)
# - *uu* (doublement médial) → *uv* (*trouuer* → *trouver*)
#
# **À vous de jouer.**

# %% [TODO]
def normalize_uv(word: str) -> str:
    """
    Normalise l'alternance u/v en moyen français.

    Règles (dans l'ordre d'application) :
    1. u initial devant voyelle  →  v
       Regex : r'^u(?=[aeiouyéèêëàâôùûîï])'
    2. u après consonne devant voyelle  →  v
       Regex : r'(?<=[bcdfghjklmnprst])u(?=[aeiouyéèêëàâôùûîï])'
    3. u après [a, e, i, j, é] devant voyelle  →  v  (cas intervocalique)
       Regex : r'(?<=[aeijé])u(?=[aeiouyéèêëàâôùûîï])'
    4. uu (doublement)  →  uv
       Re.sub simple : r'uu' → 'uv'

    Exemples
    --------
    "auoir"   → "avoir"
    "sauoir"  → "savoir"
    "trouuer" → "trouver"
    "deuoit"  → "devoit"
    "pour"    → "pour"   (pas de transformation)
    """
    # TODO : quatre substitutions regex dans l'ordre
    pass   # ← remplacer

# %% [markdown]
# ### 2b — Alternance i/j
#
# Même logique que u/v : *i* initial devant voyelle représente [j].
# *iustice* → *justice*, *iour* → *jour*, *iugement* → *jugement*.

# %% [TODO]
def normalize_ij(word: str) -> str:
    """
    Normalise l'alternance i/j en moyen français.

    Règles :
    1. i initial devant voyelle  →  j
       Regex : r'^i(?=[aeiouyéèêëàâôùûîï])'
    2. j médial devant consonne (rare)  →  i
       Regex : r'(?<=[aeiouyéèêëàâôùûîï])j(?=[bcdfghjklmnprst])'

    Exemples
    --------
    "iugement" → "jugement"
    "iour"     → "jour"
    "iustice"  → "justice"
    "iusqu"    → "jusqui"  (puis le pipeline DMF corrige)
    """
    # TODO
    pass   # ← remplacer

# %% [markdown]
# ### 2c — Règles graphémiques sur terminaisons et formes lexicales
#
# **À vous de jouer.** Complétez `apply_grapheme_rules` :
# elle applique séquentiellement la liste de règles fournie.

# %% [FOURNI]
GRAPHEME_RULES = [
    # Graphies archaïques : hault/aullt → haut
    (r'aullt?\b|ault\b',   'aut'),
    # Pluriels en -aulx
    (r'aulx\b',            'aux'),
    # rei → roi (forme picarde)
    (r'\brei\b',           'roi'),
    # ei initial devant lettre → oi
    (r'^ei(?=[a-z])',      'oi'),
    # Lexique des châteaux
    (r'\bchastel\b',       'château'),
    (r'chastelain\b',      'châtelain'),
    # Imparfait -oit → -ait
    (r'oit\b',             'ait'),
    (r'oient\b',           'aient'),
    # Conditionnel -roit → -rait
    (r'roit\b',            'rait'),
    # -our → -oir (verbes : savoir, pouvoir…) — SAUF 'pour' (préposition)
    (r'(?<=[^p\W])our\b',  'oir'),
    # Pluriels en -iax → -iaux
    (r'iax\b',             'iaux'),
    # -ele final → -elle
    (r'(?<!\w)ele\b',      'elle'),
    # Formes lexicales fréquentes
    (r'\bmoys\b',          'mois'),
    (r'\broys?\b',         'roi'),
    (r'palays\b',          'palais'),
]

# %% [TODO]
def apply_grapheme_rules(word: str, rules: list = GRAPHEME_RULES) -> str:
    """
    Applique les règles graphémiques dans l'ordre déclaré.

    L'ordre est linguistiquement significatif : les règles plus spécifiques
    (terminaisons longues) doivent précéder les règles plus générales
    (terminaisons courtes). Ne modifiez pas l'ordre de GRAPHEME_RULES.

    Paramètre
    ---------
    word  : str   un mot (sans espace)
    rules : list  liste de (pattern, replacement)

    Retourne
    --------
    str  mot normalisé

    Algorithme
    ----------
    Pour chaque (pattern, replacement) dans rules :
        word = re.sub(pattern, replacement, word)
    Retourner word.
    """
    # TODO
    pass   # ← remplacer

# %% [markdown]
# **Validation 2**

# %% [FOURNI — validation]
# u/v
assert normalize_uv("auoir")   == "avoir",   "auoir → avoir"
assert normalize_uv("sauoir")  == "savoir",  "sauoir → savoir"
assert normalize_uv("trouuer") == "trouver", "trouuer → trouver"
assert normalize_uv("deuoit")  == "devoit",  "deuoit → devoit"
assert normalize_uv("pour")    == "pour",    "'pour' ne doit pas changer"
# i/j
assert normalize_ij("iugement") == "jugement", "iugement → jugement"
assert normalize_ij("iour")     == "jour",     "iour → jour"
# règles graphiques
assert apply_grapheme_rules("venoit")  == "venait",  "-oit → -ait"
assert apply_grapheme_rules("faloit")  == "falait",  "-oit → -ait"
assert apply_grapheme_rules("hault")   == "haut",    "hault → haut"
assert apply_grapheme_rules("chastel") == "château", "chastel → château"
assert apply_grapheme_rules("roys")    == "roi",     "roys → roi"
assert apply_grapheme_rules("pouvoir") == "pouvoir", "pouvoir inchangé"
assert apply_grapheme_rules("pour")    == "pour",    "'pour' inchangé"
print("Validation 2 : OK")

# %% [markdown]
# **Question 2** : La règle `-our → -oir` exclut le mot *pour* via le
# lookbehind `(?<=[^p\W])`. Expliquez pourquoi cette exclusion est nécessaire
# linguistiquement. Quel autre mot courant cette règle pourrait-elle
# incorrectement transformer en l'absence de garde-fou ?

# %% [markdown]
# ## Étape 3 — Résolution des abréviations
#
# ### 3a — Tilde de nasalité
#
# Le tilde `~` dans les transcriptions HTR représente une consonne nasale
# supprimée par le scribe. La règle phonologique est déterministe :
# devant *b* ou *p*, la nasale est *m* ; devant toute autre consonne, c'est *n*.
#
# **À vous de jouer.**

# %% [TODO]
def resolve_nasal_tilde(word: str) -> str:
    """
    Résout les tildes de nasalité (~) selon les règles phonologiques
    du moyen français.

    Règles dans l'ordre :
    1. ~ devant b ou p  →  m
       Regex : r'~(?=[bBpP])'  →  'm'
    2. ~ devant toute autre consonne  →  n
       Regex : r'~(?=[cCdDfFgGhHjJkKlLnNrRsStTvVwWxXyYzZ])'  →  'n'
    3. ~ en fin de mot après consonne  →  on  (ex. : maso~ → mason)
       Regex : r'(?<=[bcdfghjklmnprst])~$'  →  'on'
    4. Forme isolée q~/Q~  →  que/Que  (traitement spécial)
       re.fullmatch(r'[Qq]~', word) pour détecter ce cas.

    Exemples
    --------
    "norm~die" → "normandie"
    "co~bat"   → "combat"
    "fra~ce"   → "france"
    "te~ps"    → "temps"
    "q~"       → "que"
    "Q~"       → "Que"
    """
    # TODO : quatre substitutions dans l'ordre
    pass   # ← remplacer

# %% [markdown]
# ### 3b — Table de résolution contextuelle

# %% [FOURNI]
ABBREV_TABLE = {
    # Mots grammaticaux
    "q~":  "que",        "Q~":  "Que",
    "no~": "nom",
    # Topographie
    "norm~die":  "normandie",  "Norm~die":  "Normandie",
    "co~te":     "conte",      "Co~te":     "Conte",
    "champ~e":   "champagne",  "Champ~e":   "Champagne",
    "bret~e":    "bretagne",
    "fra~ce":    "france",
    # Monnaies
    "l~":   "livres",    "s~":   "sous",    "d~":  "deniers",
    "l~t":  "livres tournois",
    # Signes souscrits : titres
    "pñce": "prince",    "Pñce": "Prince",
    "pñ":   "prison",
    "m^e":  "messire",   "M^e":  "Messire",
    "s^r":  "seigneur",  "S^r":  "Seigneur",
    "m^r":  "monseigneur",
}

# %% [TODO]
def resolve_abbreviations(word: str, table: dict = ABBREV_TABLE) -> tuple[str, bool]:
    """
    Résout une abréviation en combinant résolution phonologique et table.

    Stratégie en trois passes :
    1. Si word est directement dans table  →  retourner (table[word], True)
    2. Appliquer resolve_nasal_tilde(word)  →  phonological
       Si phonological est dans table       →  retourner (table[phonological], True)
    3. Si phonological != word (la règle a changé quelque chose)
                                            →  retourner (phonological, True)
    4. Sinon                                →  retourner (word, False)

    Le booléen indique si une résolution a été effectuée (True = résolu).
    Les abréviations non résolues alimentent les cas ambigus de
    CONVENTIONS_NLP.md.

    Exemples
    --------
    resolve_abbreviations("q~")       → ("que", True)
    resolve_abbreviations("norm~die") → ("normandie", True)
    resolve_abbreviations("pñce")     → ("prince", True)
    resolve_abbreviations("l~")       → ("livres", True)
    resolve_abbreviations("bonjour")  → ("bonjour", False)
    """
    phonological = resolve_nasal_tilde(word)
    # TODO : implémenter les 4 passes
    pass   # ← remplacer

# %% [markdown]
# **Validation 3**

# %% [FOURNI — validation]
# resolve_nasal_tilde insère la consonne nasale à la position du tilde.
# Les formes comme norm~die (lettres manquantes au-delà de la nasale)
# sont gérées par la table ABBREV_TABLE dans resolve_abbreviations.
assert resolve_nasal_tilde("co~bat")   == "combat",  "~ devant b → m"
assert resolve_nasal_tilde("fra~ce")   == "france",  "~ devant c → n"
assert resolve_nasal_tilde("te~ps")    == "temps",   "~ devant p → m"
assert resolve_nasal_tilde("q~")       == "que",     "q~ cas spécial"
assert resolve_nasal_tilde("Q~")       == "Que",     "Q~ casse préservée"

r, ok = resolve_abbreviations("q~")
assert r == "que" and ok,       "resolve_abbreviations('q~') doit retourner ('que', True)"
r, ok = resolve_abbreviations("pñce")
assert r == "prince" and ok,    "pñce → prince"
r, ok = resolve_abbreviations("l~")
assert r == "livres" and ok,    "l~ → livres"
r, ok = resolve_abbreviations("bonjour")
assert r == "bonjour" and not ok, "mot sans abréviation doit retourner (mot, False)"
print("Validation 3 : OK")

# %% [FOURNI]
# Sauvegarde de la table d'abréviations (livrable du TP)
with open("table_abreviations.json", "w", encoding="utf-8") as f:
    json.dump(ABBREV_TABLE, f, indent=2, ensure_ascii=False)
print("Livrable : table_abreviations.json sauvegardée")

# %% [markdown]
# ## Étape 4 — Cache DMF
#
# Le DMF couvre les formes médiévales inconnues des règles graphiques.
# Nous travaillons ici avec un stub offline qui simule les réponses
# que retournerait l'interface web du DMF (http://www.atilf.fr/dmf).
#
# En TP avec accès réseau, la fonction `query_dmf` du Chapitre 4
# alimente ce cache via scraping HTTP.
#
# **À vous de jouer.** Complétez `get_dmf_lemme`.

# %% [FOURNI]
DMF_STUB = {
    "preudomme":  "prudhomme",  "preudommes": "prudhommes",
    "pucele":     "pucelle",    "evesque":    "évêque",
    "seneschal":  "sénéchal",   "guillaumes": "guillaume",
    "seellee":    "scellée",    "chartre":    "charte",
    "iugement":   "jugement",   "vile":       "ville",
    "icele":      "cette",      "destrier":   "destrier",
    "missives":   "missives",   "bailli":     "bailli",
}

def load_dmf_cache(path: str = "dmf_cache.json") -> dict:
    """Charge le cache DMF depuis le disque, ou retourne le stub offline."""
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return DMF_STUB.copy()

def save_dmf_cache(cache: dict, path: str = "dmf_cache.json") -> None:
    """Persiste le cache DMF sur le disque."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

# %% [TODO]
def get_dmf_lemme(forme: str, cache: dict) -> str | None:
    """
    Retourne le lemme DMF d'une forme graphique, ou None si absent du cache.

    Paramètres
    ----------
    forme : str   forme médiévale (ex. "preudomme")
    cache : dict  cache DMF chargé

    Retourne
    --------
    str | None  lemme en français moderne (ex. "prudhomme"), ou None

    Indice
    ------
    La recherche doit être insensible à la casse :
    chercher forme.lower() dans le cache.
    """
    # TODO : une ligne
    pass   # ← remplacer

# %% [markdown]
# **Validation 4**

# %% [FOURNI — validation]
_cache = load_dmf_cache()
assert get_dmf_lemme("preudomme", _cache) == "prudhomme", "preudomme → prudhomme"
assert get_dmf_lemme("PUCELE",    _cache) == "pucelle",   "insensible à la casse"
assert get_dmf_lemme("inconnu",   _cache) is None,        "absent → None"
print("Validation 4 : OK")

# %% [markdown]
# ## Étape 5 — Pipeline complet
#
# **À vous de jouer.** Complétez `normalize_word` et `normalize_line`.
#
# L'ordre des opérations est délibéré :
# 1. Unicode en premier (corrige l'encodage avant toute regex).
# 2. Abréviations avant les règles graphiques (évite que `-oit → -ait`
#    s'applique à une forme abrégée avant qu'elle soit développée).
# 3. Règles graphiques sur la forme développée.
# 4. DMF en dernier (lookup sur la forme originale, plus stable).

# %% [TODO]
def normalize_word(word: str, dmf_cache: dict) -> str:
    """
    Normalise un mot en parcourant les quatre couches du pipeline.

    Ordre d'application :
    1. normalize_unicode(word)
    2. resolve_abbreviations(w)          → si résolu, utiliser la forme développée
       Sinon :                           → appliquer les règles graphiques :
         normalize_uv(w)
         normalize_ij(w)
         apply_grapheme_rules(w)
    3. get_dmf_lemme(word, dmf_cache)    → lookup sur le mot ORIGINAL
       Si trouvé : retourner le lemme (en préservant la majuscule initiale).
       Sinon : retourner la forme normalisée par les étapes précédentes.

    Paramètres
    ----------
    word      : str   un seul mot (sans espace)
    dmf_cache : dict  cache DMF

    Note sur la casse
    -----------------
    Si word[0].isupper() et le lemme est en minuscules,
    retourner lemme.capitalize().

    Exemples
    --------
    normalize_word("auoir",      cache) → "avoir"
    normalize_word("iugement",   cache) → "jugement"
    normalize_word("q~",         cache) → "que"
    normalize_word("pñce",       cache) → "prince"
    normalize_word("norm~die",   cache) → "normandie"
    normalize_word("preudomme",  cache) → "prudhomme"
    normalize_word("chastel",    cache) → "château"
    normalize_word("hault",      cache) → "haut"
    """
    w = normalize_unicode(word)
    # TODO : implémenter les étapes 2, 3 et 4
    pass   # ← remplacer


def normalize_line(text: str, dmf_cache: dict) -> str:
    """
    Normalise une ligne complète mot par mot.

    Tokenisation naïve : split() sur les espaces.
    Appliquer normalize_word à chaque token et rejoindre avec " ".join().
    """
    # TODO : une ligne
    pass   # ← remplacer

# %% [markdown]
# **Validation 5**

# %% [FOURNI — validation]
_cache = load_dmf_cache()
_tests = [
    ("auoir",      "avoir"),
    ("iugement",   "jugement"),
    ("q~",         "que"),
    ("pñce",       "prince"),
    ("norm~die",   "normandie"),
    ("preudomme",  "prudhomme"),
    ("chastel",    "château"),
    ("hault",      "haut"),
    ("pour",       "pour"),
    ("pouvoir",    "pouvoir"),
]
for word, expected in _tests:
    result = normalize_word(word, _cache)
    assert result == expected, \
        f"normalize_word({word!r}) → {result!r}, attendu {expected!r}"

# Test ligne complète
_line = "au duc de norm~die et au co~te de champagne"
_normalized = normalize_line(_line, _cache)
assert "normandie" in _normalized, "norm~die doit être résolu dans la ligne"
assert "conte"     in _normalized, "co~te doit être résolu dans la ligne"
print("Validation 5 : OK")
print(f"  {_line!r}")
print(f"  → {_normalized!r}")

# %% [markdown]
# ## Étape 6 — Métriques d'évaluation
#
# Trois métriques complémentaires mesurent différents aspects
# de la normalisation :
#
# - **CER** (*Character Error Rate*) : erreurs au niveau caractère —
#   la plus fine et la plus standard pour les textes historiques.
# - **BLEU-4** : chevauchement de n-grammes jusqu'à 4 —
#   capture la fidélité des séquences de mots.
# - **Token Accuracy** : pourcentage de mots exactement corrects.
#
# **À vous de jouer.** Complétez les trois fonctions.

# %% [TODO]
def compute_cer(hypotheses: list[str], references: list[str]) -> float:
    """
    Calcule le CER moyen sur un ensemble de paires.

    CER = Σ edit_distance(hyp, ref) / Σ len(ref)

    Utiliser editdistance.eval(hyp, ref) pour la distance d'édition.
    Protéger contre la division par zéro avec max(total_chars, 1).

    Retourne un float entre 0 et 1 (0 = parfait).
    """
    total_dist, total_chars = 0, 0
    for hyp, ref in zip(hypotheses, references):
        # TODO
        pass   # ← remplacer
    return total_dist / max(total_chars, 1)


def compute_bleu4(hypotheses: list[str], references: list[str]) -> float:
    """
    Calcule le BLEU-4 simplifié (sans brevity penalty).

    Algorithme pour n de 1 à 4 :
    1. Extraire les n-grammes de hyp et ref (tuples de n mots consécutifs).
    2. Compter les correspondances : pour chaque n-gramme de hyp,
       min(count_in_hyp, count_in_ref).
    3. precision_n = total_matches / max(total_ngrams_hyp, 1)

    BLEU-4 = exp( (1/4) * Σ log(precision_n) )
             avec log(0) remplacé par -100 pour éviter les erreurs.

    Retourne un float entre 0 et 100 (score en %).

    Indice
    ------
    from collections import Counter
    ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    """
    scores = []
    for n in range(1, 5):
        match_total, cand_total = 0, 0
        for hyp, ref in zip(hypotheses, references):
            # TODO
            pass   # ← remplacer
        scores.append(match_total / max(cand_total, 1))
    bleu = math.exp(sum(math.log(s) if s > 0 else -100 for s in scores) / 4)
    return round(bleu * 100, 2)


def compute_token_accuracy(hypotheses: list[str], references: list[str]) -> float:
    """
    Calcule le taux de tokens exactement corrects.

    Token Acc = Σ (tokens corrects dans la paire) / Σ max(len(hyp), len(ref))

    Indice : zip(hyp.split(), ref.split()) pour aligner les tokens.
    """
    correct, total = 0, 0
    for hyp, ref in zip(hypotheses, references):
        # TODO
        pass   # ← remplacer
    return correct / max(total, 1)

# %% [markdown]
# **Validation 6**

# %% [FOURNI — validation]
_h = ["le roi de france", "le roi de france"]
_r = ["le roi de france", "le duc de france"]
assert compute_cer(_h, _r) < 0.20,      "CER < 20% sur ces paires proches"
assert compute_token_accuracy(_h, _r) > 0.50, "Token acc > 50%"
assert 0 < compute_bleu4(_h, _r) < 100,       "BLEU entre 0 et 100"
assert compute_cer(["abc"], ["abc"]) == 0.0,   "CER parfait = 0"
print("Validation 6 : OK")

# %% [FOURNI]
# Évaluation du pipeline de règles sur le corpus complet
_sources = [s for s, _ in CORPUS_PAIRS]
_targets = [t for _, t in CORPUS_PAIRS]
_cache   = load_dmf_cache()
_hyps    = [normalize_line(s, _cache) for s in _sources]

cer_brut  = compute_cer(_sources, _targets)
cer_rules = compute_cer(_hyps,    _targets)
bleu_rules= compute_bleu4(_hyps,  _targets)
tok_rules = compute_token_accuracy(_hyps, _targets)

print(f"\nBaseline pipeline de règles ({len(CORPUS_PAIRS)} paires) :")
print(f"  CER brut         : {cer_brut:.4f}")
print(f"  CER après règles : {cer_rules:.4f}  "
      f"(réduction : {(cer_brut-cer_rules)/cer_brut*100:.1f} %)")
print(f"  BLEU-4           : {bleu_rules:.2f}")
print(f"  Token Accuracy   : {tok_rules:.4f}")

# %% [markdown]
# **Question 6** : Quel pourcentage du CER initial les règles seules permettent-elles
# de réduire ? Ce chiffre correspond-il à la répartition que vous aviez observée
# dans l'inventaire d'abréviations du TP Chapitre 2 ?
# Quels types de paires restent les plus éloignés de la cible après les règles ?

# %% [markdown]
# ## Étape 7 — Split train/val/test et journal d'expériences
#
# **À vous de jouer.** Complétez `build_split`, `compute_split_hash`
# et `log_experiment`.

# %% [TODO]
def build_split(pairs: list[tuple],
                train_ratio: float = 0.70,
                val_ratio:   float = 0.15,
                seed:        int   = 42) -> dict[str, list]:
    """
    Partitionne les paires en train/val/test par mélange aléatoire.

    Paramètres
    ----------
    pairs       : list  liste de (source, target)
    train_ratio : float fraction entraînement (défaut 0.70)
    val_ratio   : float fraction validation    (défaut 0.15)
    seed        : int   graine pour reproductibilité

    Retourne
    --------
    dict avec clés "train", "val", "test" → listes de paires

    Algorithme
    ----------
    1. Copier et mélanger pairs avec random.Random(seed).shuffle().
    2. n_train = int(n * train_ratio)
       n_val   = int(n * val_ratio)
    3. train = shuffled[:n_train]
       val   = shuffled[n_train:n_train+n_val]
       test  = shuffled[n_train+n_val:]
    """
    # TODO
    pass   # ← remplacer


def compute_split_hash(split: dict) -> str:
    """
    Hash SHA-256 déterministe du split courant.

    Calculé sur la liste triée de (split_name, source_text),
    sérialisée en JSON. Le tri garantit l'idempotence quelle que
    soit l'ordre des paires dans le dict.

    Retourne
    --------
    str  hash hexadécimal de 64 caractères
    """
    # TODO : même logique que le TP Chapitre 2
    pass   # ← remplacer


def log_experiment(config: dict, metrics: dict, split_hash: str,
                   path: str = "experiments/journal.jsonl") -> None:
    """
    Enregistre une expérience dans le journal JSONL.

    Format d'une entrée :
    {
      "timestamp":  "2026-...",
      "split_hash": "abc123...",
      "config":     { ... },
      "metrics":    { ... }
    }

    Crée le répertoire parent si nécessaire (os.makedirs(..., exist_ok=True)).
    Mode append ("a") pour ne pas écraser les expériences précédentes.
    """
    # TODO
    pass   # ← remplacer

# %% [markdown]
# **Validation 7**

# %% [FOURNI — validation]
_split  = build_split(CORPUS_PAIRS)
assert set(_split.keys()) == {"train", "val", "test"}, "Trois clés requises"
assert len(_split["train"]) + len(_split["val"]) + len(_split["test"]) \
       == len(CORPUS_PAIRS), "Aucune paire ne doit être perdue"
# Pas de source commune entre train et test
_src_train = {s for s,_ in _split["train"]}
_src_test  = {s for s,_ in _split["test"]}
assert not (_src_train & _src_test), "Train et test ne doivent pas se chevaucher"
# Idempotence du hash
_h1 = compute_split_hash(_split)
_h2 = compute_split_hash(build_split(CORPUS_PAIRS))
assert _h1 == _h2 and len(_h1) == 64, "Hash doit être idempotent et de 64 chars"
print(f"Validation 7 : OK — SHA-256 = {_h1[:24]}...")

SPLIT      = _split
SPLIT_HASH = _h1

# Enregistrer la baseline règles dans le journal
log_experiment(
    config  = {"step": "rules", "rules_version": "v1.0", "abbrev_table": len(ABBREV_TABLE)},
    metrics = {"cer_brut": round(cer_brut,4), "cer_rules": round(cer_rules,4),
               "bleu4": round(bleu_rules,2), "token_acc": round(tok_rules,4)},
    split_hash = SPLIT_HASH,
)
print("Journal mis à jour : experiments/journal.jsonl")

# %% [markdown]
# ## Étape 8 — Bilan mémoire théorique : LoRA vs QLoRA
#
# Avant de lancer un entraînement, calculez l'empreinte mémoire de chaque
# configuration. C'est l'exercice central du Chapitre 3 : comprendre ce que
# vous allouez sur le GPU avant d'appuyer sur "train".
#
# **Rappel architectural de mT5-small** (le modèle que vous utiliserez) :
# $d_{\text{model}} = 512$, $d_{kv} = 64$ (dimension des têtes Q/V),
# $n_{\text{layers}} = 8$ (4 encodeur + 4 décodeur), paramètres totaux ≈ 77 M.
#
# **À vous de jouer.** Complétez les deux fonctions de bilan mémoire.

# %% [TODO]
def compute_lora_params(d_model:        int,
                         d_kv:           int,
                         n_layers:       int,
                         r:              int,
                         target_modules: list[str]) -> dict:
    """
    Calcule le bilan paramétrique et mémoire d'une configuration LoRA.

    Pour chaque matrice adaptée par LoRA (Q ou V), on entraîne :
      - Matrice A : r × d_model  (projection descendante)
      - Matrice B : d_kv × r     (projection montante)
      soit  r × (d_model + d_kv)  paramètres par matrice.

    Le full fine-tuning de la même matrice coûterait d_model × d_kv.

    Paramètres
    ----------
    d_model        : int   dimension du modèle (ex. 512 pour mT5-small)
    d_kv           : int   dimension des têtes (ex. 64 pour mT5-small)
    n_layers       : int   nombre de couches (enc + dec)
    r              : int   rang LoRA
    target_modules : list  ex. ["q", "v"]

    Retourne
    --------
    dict avec :
        "lora_trainable"   : int    paramètres LoRA entraînables
        "full_ft_params"   : int    paramètres équivalents en full FT
        "pct_trainable"    : float  lora / full * 100
        "mem_full_adam_gb" : float  mémoire Adam pour full FT (Go)
        "mem_lora_adam_gb" : float  mémoire Adam pour LoRA seul (Go)
        "ratio_mem"        : float  mem_full / mem_lora

    Note mémoire
    ------------
    Adam stocke 4 valeurs par paramètre en float32 (poids + gradient
    + 2 moments) = 16 octets/paramètre.
    Convertir en Go : n_params * 16 / 1e9.
    """
    n_matrices        = n_layers * len(target_modules)
    params_per_matrix = r * (d_model + d_kv)
    lora_params       = n_matrices * params_per_matrix
    full_ft_params    = n_matrices * d_model * d_kv
    pct               = lora_params / max(full_ft_params, 1) * 100

    # TODO : calculer mem_full_adam_gb, mem_lora_adam_gb, ratio_mem
    # et retourner le dict complet
    pass   # ← remplacer


def compute_qlora_memory(n_params:       int,
                          d_model:        int,
                          d_kv:           int,
                          n_layers:       int,
                          r:              int,
                          target_modules: list[str]) -> dict:
    """
    Estime la mémoire GPU totale en configuration QLoRA (NF4 + LoRA).

    Composantes :
    - Poids du modèle en NF4 (4 bits = 0.5 octet/paramètre) :
        model_nf4_gb = n_params * 0.5 / 1e9
    - Optimiseur Adam sur les seuls paramètres LoRA (float32) :
        optim_gb = lora_trainable * 16 / 1e9
      (récupérer lora_trainable depuis compute_lora_params)

    Retourne
    --------
    dict avec "model_nf4_gb", "optim_lora_gb", "total_est_gb"
    """
    lora_info = compute_lora_params(d_model, d_kv, n_layers, r, target_modules)
    # TODO
    pass   # ← remplacer

# %% [markdown]
# **Validation 8**

# %% [FOURNI — validation]
# mT5-small : d_model=512, d_kv=64, 8 couches, ~77M params
_r8  = compute_lora_params(512, 64, 8, 8,  ["q", "v"])
_r16 = compute_lora_params(512, 64, 8, 16, ["q", "v"])
assert _r8  is not None, "compute_lora_params doit retourner un dict"
assert _r8["lora_trainable"] < _r16["lora_trainable"], "r=16 > r=8 en paramètres"
assert _r8["pct_trainable"]  < 20,                     "LoRA < 20% du full FT"
assert _r8["mem_full_adam_gb"] > _r8["mem_lora_adam_gb"], "Full FT > LoRA en mémoire"

_q8 = compute_qlora_memory(77_000_000, 512, 64, 8, 8, ["q", "v"])
assert _q8 is not None,             "compute_qlora_memory doit retourner un dict"
assert _q8["model_nf4_gb"] < 0.10,  "mT5-small en NF4 < 100 Mo"
assert _q8["total_est_gb"] < 0.5,   "QLoRA total < 500 Mo pour mT5-small"
print("Validation 8 : OK")

# %% [FOURNI]
# Tableau comparatif LoRA r=8 vs r=16 vs QLoRA r=8
print("\nBilan mémoire — mT5-small (77M params, d_model=512, d_kv=64, 8 couches)")
print(f"{'Configuration':30s} {'Params entr.':>14} {'% LoRA/FT':>10} "
      f"{'Adam LoRA (Go)':>16} {'Adam Full FT (Go)':>18}")
for label, cfg in [
    ("LoRA r=8,  Q+V",  compute_lora_params(512,64,8, 8,["q","v"])),
    ("LoRA r=16, Q+V",  compute_lora_params(512,64,8,16,["q","v"])),
    ("LoRA r=8,  Q+K+V",compute_lora_params(512,64,8, 8,["q","k","v"])),
]:
    print(f"  {label:28s} {cfg['lora_trainable']:>14,} {cfg['pct_trainable']:>9.2f}%"
          f" {cfg['mem_lora_adam_gb']:>16.5f} {cfg['mem_full_adam_gb']:>18.4f}")

q8 = compute_qlora_memory(77_000_000, 512, 64, 8, 8, ["q","v"])
print(f"\nQLoRA r=8, Q+V (NF4):")
print(f"  Modèle NF4 : {q8['model_nf4_gb']:.4f} Go")
print(f"  Adam LoRA  : {q8['optim_lora_gb']:.5f} Go")
print(f"  Total est. : {q8['total_est_gb']:.4f} Go")

# %% [markdown]
# **Question 8** : Comparez la colonne "Adam LoRA" entre r=8 et r=16.
# Doublez-vous vraiment la consommation mémoire en doublant r ?
# Pourquoi QLoRA est-il si avantageux pour les très grands modèles
# (ex. LLaMA-7B = 7 milliards de paramètres) mais moins décisif
# pour mT5-small ?

# %% [markdown]
# ## Étape 9 — Fine-tuning LoRA sur mT5-small
#
# Cette étape entraîne mT5-small avec LoRA sur le corpus de normalisation.
# Le modèle est suffisamment petit pour tourner sur CPU (lentement)
# ou sur GPU T4/A100 (en quelques minutes).
#
# **Scénario CPU :** `fp16=False`, `batch_size=2`, `num_epochs=3`.
# L'entraînement prend ~10-15 min. Le résultat est indicatif.
#
# **Scénario GPU :** `fp16=True`, `batch_size=8`, `num_epochs=10`.
# Résultats exploitables pour le tableau d'ablation.

# %% [FOURNI]
# Détection GPU
import torch
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
USE_FP16   = DEVICE == "cuda"
BATCH_SIZE = 8 if DEVICE == "cuda" else 2
N_EPOCHS   = 10 if DEVICE == "cuda" else 3

print(f"Dispositif : {DEVICE.upper()}")
print(f"  fp16={USE_FP16}, batch={BATCH_SIZE}, epochs={N_EPOCHS}")
if DEVICE == "cpu":
    print("  Mode CPU : entraînement indicatif (~10 min). "
          "Pour des résultats de production, utiliser un GPU.")

# %% [markdown]
# ### 9.1 Préparation des données

# %% [FOURNI]
from transformers import AutoTokenizer

MODEL_NAME = "google/mt5-small"   # 77M params — entraînable sur CPU
PREFIX     = "normalise moyen français: "

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"Tokeniseur chargé : {MODEL_NAME}")
except Exception as e:
    print(f"Erreur de chargement ({e}).")
    print("Vérifiez votre connexion réseau ou utilisez un cache local.")
    raise

# %% [TODO]
def prepare_dataset(pairs: list[tuple],
                    tokenizer,
                    prefix:     str = PREFIX,
                    max_length: int = 128) -> list[dict]:
    """
    Tokenise les paires (source, target) pour l'entraînement mT5.

    Pour chaque paire (src, tgt) :
    1. Construire l'entrée : prefix + src
    2. Tokeniser source (avec truncation, max_length)
       → input_ids, attention_mask
    3. Tokeniser target (avec truncation, max_length)
       → labels = target_ids["input_ids"]
    4. Retourner un dict avec "input_ids", "attention_mask", "labels"

    Note : ne pas appliquer de padding ici — le DataCollator s'en charge.

    Paramètres
    ----------
    pairs      : list  liste de (source_brut, target_normalisé)
    tokenizer  : tokenizer HuggingFace
    prefix     : str   préfixe de tâche T5
    max_length : int   longueur maximale

    Retourne
    --------
    list[dict]  un dict par exemple
    """
    dataset = []
    for src, tgt in pairs:
        # TODO : tokeniser src et tgt, construire le dict
        pass   # ← remplacer
    return dataset

# %% [markdown]
# **Validation 9.1**

# %% [FOURNI — validation]
_sample = prepare_dataset(CORPUS_PAIRS[:3], tokenizer)
assert len(_sample) == 3,                    "Un exemple par paire"
assert "input_ids"      in _sample[0],       "Clé input_ids manquante"
assert "attention_mask" in _sample[0],       "Clé attention_mask manquante"
assert "labels"         in _sample[0],       "Clé labels manquante"
assert isinstance(_sample[0]["input_ids"], list), "input_ids doit être une liste"
assert len(_sample[0]["input_ids"]) <= 128,  "Longueur max 128"
print("Validation 9.1 : OK")

# %% [markdown]
# ### 9.2 Configuration LoRA et entraînement

# %% [TODO]
def build_lora_config(r:              int   = 8,
                       alpha:          int   = 16,
                       dropout:        float = 0.1,
                       target_modules: list  = None) -> "LoraConfig":
    """
    Construit une configuration LoRA pour mT5.

    Paramètres
    ----------
    r              : rang LoRA (défaut 8)
    alpha          : facteur d'échelle (défaut 16 → ratio α/r = 2)
    dropout        : taux de dropout LoRA (défaut 0.1)
    target_modules : modules à adapter (défaut ["q", "v"])

    Retourne
    --------
    LoraConfig (peft)

    Indice
    ------
    from peft import LoraConfig, TaskType
    LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules or ["q", "v"],
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    """
    from peft import LoraConfig, TaskType
    # TODO
    pass   # ← remplacer

# %% [FOURNI]
def train_lora(split:        dict,
               tokenizer,
               model_name:   str   = MODEL_NAME,
               r:            int   = 8,
               alpha:        int   = 16,
               dropout:      float = 0.1,
               n_epochs:     int   = N_EPOCHS,
               batch_size:   int   = BATCH_SIZE,
               lr:           float = 3e-4,
               output_dir:   str   = "./lora_checkpoint") -> dict:
    """
    Entraîne mT5-small avec LoRA sur le corpus de normalisation.
    Retourne les métriques de validation finales.
    """
    from transformers import (AutoModelForSeq2SeqLM, Seq2SeqTrainer,
                               Seq2SeqTrainingArguments,
                               DataCollatorForSeq2Seq, EarlyStoppingCallback)
    from peft import get_peft_model
    from datasets import Dataset

    lora_config = build_lora_config(r=r, alpha=alpha, dropout=dropout)
    if lora_config is None:
        print("build_lora_config non implémenté — skip entraînement.")
        return {}

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_data = Dataset.from_list(prepare_dataset(split["train"], tokenizer))
    val_data   = Dataset.from_list(prepare_dataset(split["val"],   tokenizer))

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    args = Seq2SeqTrainingArguments(
        output_dir              = output_dir,
        num_train_epochs        = n_epochs,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size  = batch_size,
        learning_rate           = lr,
        lr_scheduler_type       = "cosine",
        warmup_ratio            = 0.1,
        fp16                    = USE_FP16,
        predict_with_generate   = True,
        generation_max_length   = 128,
        eval_strategy           = "epoch",
        save_strategy           = "epoch",
        load_best_model_at_end  = True,
        metric_for_best_model   = "eval_loss",
        greater_is_better       = False,
        logging_steps           = 5,
        report_to               = "none",
    )

    trainer = Seq2SeqTrainer(
        model           = model,
        args            = args,
        train_dataset   = train_data,
        eval_dataset    = val_data,
        data_collator   = collator,
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=3)],
    )
    trainer.train()
    metrics = trainer.evaluate()
    model.save_pretrained(output_dir)
    return metrics

# %% [FOURNI]
print("\nLancement de l'entraînement LoRA r=8...")
try:
    lora_r8_metrics = train_lora(SPLIT, tokenizer, r=8, output_dir="./lora_r8")
    log_experiment(
        config     = {"method":"lora","model":MODEL_NAME,"r":8,"alpha":16,
                      "dropout":0.1,"target_modules":["q","v"],"epochs":N_EPOCHS},
        metrics    = lora_r8_metrics,
        split_hash = SPLIT_HASH,
    )
    print(f"LoRA r=8 : eval_loss = {lora_r8_metrics.get('eval_loss','N/A'):.4f}")
except Exception as e:
    print(f"Entraînement non disponible ({e}).")
    lora_r8_metrics = {}

# %% [FOURNI]
print("\nLancement de l'entraînement LoRA r=16...")
try:
    lora_r16_metrics = train_lora(SPLIT, tokenizer, r=16, output_dir="./lora_r16")
    log_experiment(
        config     = {"method":"lora","model":MODEL_NAME,"r":16,"alpha":16,
                      "dropout":0.1,"target_modules":["q","v"],"epochs":N_EPOCHS},
        metrics    = lora_r16_metrics,
        split_hash = SPLIT_HASH,
    )
    print(f"LoRA r=16 : eval_loss = {lora_r16_metrics.get('eval_loss','N/A'):.4f}")
except Exception as e:
    print(f"Entraînement non disponible ({e}).")
    lora_r16_metrics = {}

# %% [markdown]
# ## Étape 10 — Fine-tuning QLoRA (GPU recommandé)
#
# QLoRA combine la quantisation NF4 (poids du modèle en 4 bits)
# avec LoRA (seuls les adaptateurs sont entraînés en float32).
# Le gain est maximal pour les grands modèles ; sur mT5-small,
# il reste pédagogiquement instructif.
#
# **Sur CPU :** `bitsandbytes` requiert CUDA. En l'absence de GPU,
# la cellule affiche le bilan théorique et saute l'entraînement.
# Le code est valide et peut être exécuté tel quel sur Colab T4.

# %% [TODO]
def build_qlora_model(model_name: str = MODEL_NAME,
                       r:          int = 8) -> tuple:
    """
    Charge mT5-small en quantisation NF4 et l'enveloppe avec LoRA.

    Étapes :
    1. Construire BitsAndBytesConfig :
         load_in_4bit=True,
         bnb_4bit_quant_type="nf4",
         bnb_4bit_compute_dtype=torch.float16,
         bnb_4bit_use_double_quant=True
    2. Charger AutoModelForSeq2SeqLM.from_pretrained(
           model_name,
           quantization_config=bnb_config,
           device_map="auto"
       )
    3. Appeler prepare_model_for_kbit_training(model)
       (cast des LayerNorms en float32 pour la stabilité)
    4. Appeler get_peft_model(model, build_lora_config(r=r))
    5. Retourner (model, tokenizer) où tokenizer est déjà chargé.

    Si bitsandbytes n'est pas disponible (pas de GPU CUDA),
    lever ImportError avec un message explicatif.

    Indice
    ------
    from transformers import BitsAndBytesConfig
    from peft import prepare_model_for_kbit_training
    """
    if DEVICE == "cpu":
        raise ImportError(
            "QLoRA requiert CUDA (bitsandbytes ne supporte pas le CPU). "
            "Exécutez ce TP sur Colab T4 ou équivalent pour cette étape."
        )
    from transformers import BitsAndBytesConfig, AutoModelForSeq2SeqLM
    from peft import get_peft_model, prepare_model_for_kbit_training
    # TODO : implémenter les 5 étapes
    pass   # ← remplacer

# %% [FOURNI]
print("\nTentative de chargement QLoRA...")
try:
    qlora_model, _ = build_qlora_model(MODEL_NAME, r=8)
    # Entraînement QLoRA (même routine que LoRA, le modèle est déjà quantisé)
    qlora_metrics = train_lora(SPLIT, tokenizer, r=8, output_dir="./qlora_r8")
    log_experiment(
        config     = {"method":"qlora","model":MODEL_NAME,"r":8,"quantization":"nf4",
                      "double_quant":True,"compute_dtype":"float16"},
        metrics    = qlora_metrics,
        split_hash = SPLIT_HASH,
    )
    print(f"QLoRA r=8 : eval_loss = {qlora_metrics.get('eval_loss','N/A'):.4f}")
except ImportError as e:
    print(f"[INFO] {e}")
    print("Bilan théorique affiché à l'Étape 8. Entraînement QLoRA non exécuté.")
    qlora_metrics = {}
except Exception as e:
    print(f"Erreur inattendue QLoRA : {e}")
    qlora_metrics = {}

# %% [markdown]
# ## Étape 11 — Tableau d'ablation comparatif
#
# **À vous de jouer.** Complétez `build_ablation_table` :
# elle rassemble les résultats de toutes les configurations
# dans un DataFrame pandas pour l'analyse finale.

# %% [TODO]
def build_ablation_table(results: dict,
                          lora_r8_cfg:  dict,
                          lora_r16_cfg: dict,
                          qlora_cfg:    dict) -> pd.DataFrame:
    """
    Construit le tableau d'ablation comparatif.

    Paramètres
    ----------
    results      : dict  {config_name: {"cer", "bleu4", "tok_acc", "eval_loss"}}
    lora_r8_cfg  : dict  retour de compute_lora_params  pour r=8
    lora_r16_cfg : dict  retour de compute_lora_params  pour r=16
    qlora_cfg    : dict  retour de compute_qlora_memory pour r=8

    Retourne
    --------
    pd.DataFrame avec colonnes :
        Configuration | CER ↓ | BLEU-4 ↑ | Token Acc ↑ | Params entr. | Mémoire Adam

    Algorithme
    ----------
    Construire une liste de dicts, un par ligne du tableau.
    La ligne "Règles seules" utilise cer_rules, bleu_rules, tok_rules
    calculés à l'Étape 6 (0 paramètres entraînés).
    """
    rows = []
    # TODO : construire rows avec au minimum :
    # - "Texte brut (baseline)" : cer_brut, —, —, 0 params
    # - "Règles seules"         : cer_rules, bleu_rules, tok_rules, 0 params
    # - "LoRA r=8,  Q+V"        : depuis results["lora_r8"]  + lora_r8_cfg
    # - "LoRA r=16, Q+V"        : depuis results["lora_r16"] + lora_r16_cfg
    # - "QLoRA r=8, Q+V (NF4)"  : depuis results["qlora"]    + qlora_cfg
    pass   # ← remplacer
    return pd.DataFrame(rows)

# %% [FOURNI]
_results = {
    "lora_r8":  lora_r8_metrics,
    "lora_r16": lora_r16_metrics,
    "qlora":    qlora_metrics,
}
_lora_r8_cfg  = compute_lora_params(512, 64, 8,  8,  ["q","v"]) or {}
_lora_r16_cfg = compute_lora_params(512, 64, 8,  16, ["q","v"]) or {}
_qlora_cfg    = compute_qlora_memory(77_000_000, 512, 64, 8, 8, ["q","v"]) or {}

ablation = build_ablation_table(_results, _lora_r8_cfg, _lora_r16_cfg, _qlora_cfg)
if ablation is not None and len(ablation) > 0:
    print("\nTableau d'ablation :")
    print(ablation.to_string(index=False))
else:
    print("Tableau d'ablation non disponible (TODO non complété ou modèles non entraînés).")

# %% [markdown]
# **Question 11** :
#
# Analysez votre tableau d'ablation en répondant aux trois questions suivantes.
#
# Premièrement : l'hypothèse du faible rang intrinsèque est-elle vérifiée
# sur votre corpus de normalisation ? Autrement dit, r=8 donne-t-il des
# résultats significativement inférieurs à r=16, ou les deux configurations
# convergent-elles vers les mêmes métriques ?
#
# Deuxièmement : le pipeline hybride (règles → mT5-LoRA) est-il meilleur
# que mT5-LoRA seul ? Quelle fraction du gain est attribuable aux règles,
# et quelle fraction au modèle neural ?
#
# Troisièmement : QLoRA produit-il une dégradation mesurable des métriques
# par rapport à LoRA sur ce modèle de taille modeste (77 M paramètres) ?
# À partir de quelle taille de modèle QLoRA devient-il indispensable ?

# %% [markdown]
# ## Étape 12 — CONVENTIONS_NLP.md
#
# Générez le fichier de conventions. Ce livrable documente chaque décision
# de normalisation non entièrement déterminée par le DMF.

# %% [FOURNI]
_conventions = f"""# CONVENTIONS_NLP.md

## Version : 1.0 | Date : {datetime.date.today()} | Split SHA-256 : {SPLIT_HASH[:32]}

### Principes généraux
- Normalisation vers le français moderne orthographié standard.
- Toute décision non couverte par le DMF est documentée ici.
- Le module de règles est versionné (v1.0) ; toute modification incrémente
  le numéro de version et invalide les comparaisons antérieures.

### Règles graphiques appliquées (v1.0)
| Règle | Avant | Après | Justification |
|---|---|---|---|
| u/v positionnel | auoir | avoir | Alternance phonographique systématique |
| i/j initial | iour | jour | Alternance phonographique systématique |
| -oit → -ait | venoit | venait | Imparfait médiéval → moderne |
| -roit → -rait | pouroit | pourrait | Conditionnel médiéval → moderne |
| chastel → château | chastel | château | Forme dialectale → standard |
| hault → haut | hault | haut | Graphie archaïque |
| rei → roi | rei | roi | Forme picarde → standard |
| roys → roi | roys | roi | Pluriel cas sujet → forme moderne |

### Cas ambigus résolus
| Forme brute | Expansion choisie | Alternative | Justification |
|---|---|---|---|
| q~ | que | qui (devant voyelle) | Convention t9n |
| l~ | livres | livres tournois | Contexte : sans suffixe 't' |
| p~ | par | per / pro | Contexte : devant substantif → par |

### Formes rares transmises au modèle neural
- Noms propres abrégés non couverts par la table
- Abréviations hapax sans contexte suffisant
- Graphies scripto-régionales non référencées dans le DMF stub
"""

with open("CONVENTIONS_NLP.md", "w", encoding="utf-8") as f:
    f.write(_conventions)
print("Livrable : CONVENTIONS_NLP.md généré.")

# %% [markdown]
# ---
#
# ## Récapitulatif — Ce que vous avez implémenté
#
# | Étape | Fonction(s) | Lien avec le cours |
# |---|---|---|
# | 1 Unicode | `normalize_unicode` | Chapitre 4 §2.2 |
# | 2 Règles | `normalize_uv`, `normalize_ij`, `apply_grapheme_rules` | Chapitre 4 §1.4 + §2.3 |
# | 3 Abréviations | `resolve_nasal_tilde`, `resolve_abbreviations` | Chapitre 4 §4 |
# | 4 DMF | `get_dmf_lemme` | Chapitre 4 §3 |
# | 5 Pipeline | `normalize_word`, `normalize_line` | Chapitre 4 §2.1 |
# | 6 Métriques | `compute_cer`, `compute_bleu4`, `compute_token_accuracy` | Chapitre 4 §6.2 |
# | 7 Split + journal | `build_split`, `compute_split_hash`, `log_experiment` | Chapitre 2 §6 + Chapitre 3 §6.3 |
# | 8 Bilan mémoire | `compute_lora_params`, `compute_qlora_memory` | Chapitre 3 §2.3 + §4.4 |
# | 9 LoRA | `prepare_dataset`, `build_lora_config` | Chapitre 3 §6.1 |
# | 10 QLoRA | `build_qlora_model` | Chapitre 3 §4 |
# | 11 Ablation | `build_ablation_table` | Chapitre 3 §8.1 |
#
# **Livrables produits :**
# `table_abreviations.json`, `CONVENTIONS_NLP.md`,
# `experiments/journal.jsonl`, checkpoints LoRA/QLoRA (si GPU disponible).
