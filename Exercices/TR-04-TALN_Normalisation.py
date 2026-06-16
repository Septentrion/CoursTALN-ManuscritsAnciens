"""
TP Guidé — Normalisation orthographique du moyen français
         Règles graphiques · Abréviations · DMF · Arbitrage MLM · mT5 LoRA
Chapitre 4 — Module NLP · Master Data/IA · MD5 Volet 2 · 2026
─────────────────────────────────────────────────────────────────────────────
Pipeline implémenté dans ce TP :

  Transcription HTR brute
        │
  [Étape 1]  Normalisation Unicode + règles graphémiques (u/v, i/j, -oit→-ait)
             → mesure du CER avant/après
        │
  [Étape 2]  Résolution des abréviations (tilde nasal, signes suprascripts)
             → table ABBREV_TABLE + règles phonologiques
        │
  [Étape 3]  Lookup DMF (cache JSON pré-rempli)
             → normalize_word_with_dmf : combinaison règles + lexique
        │
  [Étape 4]  Arbitrage CamemBERT MLM sur les positions de faible confiance
             → score_text_mlm (PLL), arbitrate_candidate, ablation A/B/C
        │
  [Étape 5]  Construction des paires d'entraînement + fine-tuning mT5 LoRA
             → build_normalization_pairs, WeightedSeq2SeqTrainer
        │
  [Étape 6]  Évaluation CER / BLEU / Token Accuracy
             → tableau d'ablation complet
        │
  [Étape 7]  Traçabilité : CONVENTIONS_NLP.md + journal d'expériences

Durée estimée : 3 h 30
Livrables :
  - CONVENTIONS_NLP.md           (décisions de normalisation documentées)
  - dmf_cache.json               (cache DMF pre-rempli)
  - normalization_ablation.json  (tableau d'ablation)
  - experiments/journal.jsonl    (mis à jour)

Instructions générales
──────────────────────
Les cellules  # TODO  contiennent des squelettes à compléter.
Les cellules  # FOURNI  sont à exécuter telles quelles.
Ne pas modifier les signatures des fonctions.
"""

# %% [markdown]
# # TP Guidé — Normalisation orthographique du moyen français
#
# Ce TP implémente le pipeline complet de normalisation orthographique
# décrit dans le Chapitre 4. Vous partirez de transcriptions HTR brutes
# contenant les graphies médiévales non standardisées et produirez des
# formes normalisées en français moderne, en quatre étapes emboîtées.
#
# À chaque étape, vous mesurerez le gain en CER (Character Error Rate)
# par rapport à la baseline HTR brute, pour construire le tableau
# d'ablation final.

# %% [markdown]
# ## Partie 0 — Corpus et imports

# %% [FOURNI]
import re, unicodedata, editdistance, json, random, math
import hashlib, datetime, os
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np

random.seed(42)

# %% [FOURNI]
# ── Corpus de travail : 40 paires (transcription HTR brute, normalisé, confidence) ──
# Chaque entrée est un tuple (transcription_brute, reference_normalisee, confidence_htr).
# En TP réel, ces données viennent de votre enriched_corpus.jsonl du Jour 2.

CORPUS = [
    # charte royale
    ("li roys de france signa l acte",           "le roi de france signa l acte",              0.94),
    ("au duc de norm~die",                        "au duc de normandie",                         0.71),
    ("la chartre fut seellee du seel royal",      "la charte fut scellée du sceau royal",        0.89),
    ("en l an de grace mil trois cent quarante",  "en l an de grâce mil trois cent quarante",    0.92),
    ("q~ li cheualier ait restitue les terres",   "que le chevalier ait restitué les terres",    0.67),
    ("don de la terre de gisors au chastelain",   "don de la terre de gisors au châtelain",      0.88),
    ("ledit seigneur porta les lettres patentes", "ledit seigneur porta les lettres patentes",   0.91),
    ("pñce de normandie fit don a l eglise",      "prince de normandie fit don à l église",      0.73),
    # registre comptable
    ("item le bailli rendit son jugement",         "item le bailli rendit son jugement",          0.85),
    ("co~te des deniers receus au tresor",         "conte des deniers reçus au trésor",           0.68),
    ("messire guillaumes de villehardouin",        "messire Guillaume de Villehardouin",          0.87),
    ("au moys de mars de l an susdit",             "au mois de mars de l an susdit",              0.93),
    ("pour le paiement de vingt l~ de rente",     "pour le paiement de vingt livres de rente",  0.66),
    ("ledit preudomme fu condamne a l amende",    "ledit prud'homme fut condamné à l amende",    0.82),
    ("no~ du seneschal de champagne",              "nom du sénéchal de champagne",                0.70),
    # texte narratif
    ("li chevaliers arma son destrier",            "le chevalier arma son destrier",              0.78),
    ("et prist sa lance et son escu",              "et prit sa lance et son écu",                 0.91),
    ("la dame du chastel vit le cheualier",        "la dame du châtel vit le chevalier",          0.83),
    ("q~ feist il dist li roys",                   "que fit-il dit le roi",                       0.62),
    ("lors s en ala vers le palays roial",         "lors s en alla vers le palais royal",         0.79),
    ("messire gauvain respondi au roi",            "messire Gauvain répondit au roi",             0.88),
    ("la pucele estoit de grant beaute",           "la pucelle était de grande beauté",           0.85),
    # documents divers
    ("le seneschal porta les lettres au roi",      "le sénéchal porta les lettres au roi",        0.91),
    ("iustice fu rendue au palays",                "justice fut rendue au palais",                0.80),
    ("auoir fait seruice au seigneur",             "avoir fait service au seigneur",              0.86),
    ("norm~die fut conq~ise par les angloys",      "normandie fut conquise par les anglais",      0.74),
    ("le co~te de champagne signa l acte",         "le conte de champagne signa l acte",          0.79),
    ("vn cheualier de grand valeur",               "un chevalier de grand valeur",                0.84),
    ("le seel du roy de france",                   "le sceau du roi de france",                   0.90),
    ("fra~ce et norm~die sont au roy",             "france et normandie sont au roi",             0.76),
    ("la chartre fut faite en l an",               "la charte fut faite en l an",                 0.88),
    ("iour de saint iehan baptiste",               "jour de saint jean baptiste",                 0.81),
    ("le s^r de gisors fit seruice",               "le seigneur de gisors fit service",           0.83),
    ("m^r de normandie porta l acte",              "monseigneur de normandie porta l acte",       0.75),
    ("co~te de deniers en l an courant",           "conte de deniers en l an courant",            0.68),
    ("le pñce signa de sa main",                   "le prince signa de sa main",                  0.73),
    ("iuger les cas par deuant le bailli",         "juger les cas par devant le bailli",          0.82),
    ("auoit este fait prisonnier",                 "avoit été fait prisonnier",                   0.77),
    ("le moys de iuillet en l an",                 "le mois de juillet en l an",                  0.86),
    ("sauoir faire et dire en iustice",            "savoir faire et dire en justice",             0.89),
]

SOURCES = [c[0] for c in CORPUS]
REFS    = [c[1] for c in CORPUS]
CONFS   = [c[2] for c in CORPUS]

# Génération du split_hash (SHA-256 sur les sources)
SPLIT_HASH = hashlib.sha256(
    "\n".join(SOURCES).encode("utf-8")
).hexdigest()[:16]

print(f"Corpus : {len(CORPUS)} paires · split_hash = {SPLIT_HASH}")
print(f"Confidence moyenne : {sum(CONFS)/len(CONFS):.2f}")
print(f"Lignes needs_review (conf < 0.75) : "
      f"{sum(1 for c in CONFS if c < 0.75)}")

# %% [markdown]
# ---
# ## Étape 1 — Règles graphémiques

# %% [markdown]
# ### 1.1 Normalisation Unicode et table de caractères médiévaux
#
# Avant d'appliquer toute règle linguistique, les textes doivent être
# homogénéisés au niveau de l'encodage.
#
# **À vous de jouer.** Complétez `normalize_unicode` en appliquant :
# 1. `unicodedata.normalize('NFC', text)` — unification NFC
# 2. Translittération des caractères médiévaux de `MEDIEVAL_UNICODE`

# %% [FOURNI]
MEDIEVAL_UNICODE = {
    'ꝑ': 'p',    # p barré (per/par/pro)
    'ꝓ': 'pro',  # p barré avec crochet
    'ꝕ': 'p',    # variante p barré
    'ȷ': 'j',    # j sans point
    'ı': 'i',    # i sans point (latin)
}

# %% [TODO]
def normalize_unicode(text: str) -> str:
    """
    Étape 1a — Normalisation Unicode et translittération médiévale.

    Opérations dans l'ordre :
    1. unicodedata.normalize('NFC', text)
    2. Pour chaque (src, tgt) dans MEDIEVAL_UNICODE : text.replace(src, tgt)

    Paramètre
    ---------
    text : str  texte brut (une ligne HTR)

    Retourne
    --------
    str  texte avec encodage unifié et caractères spéciaux translittérés
    """
    # TODO
    pass


# %% [markdown]
# ### 1.2 Alternance u/v
#
# En moyen français, *u* et *v* représentent indifféremment [u] ou [v]
# selon leur position. Trois règles positionnelles couvrent tous les cas :
#
# **Règle 1** — *u* en position **initiale** devant voyelle → *v*
#   ex. : *uoir* → *voir*, *uenir* → *venir*
#
# **Règle 2** — *u* en position **intervocalique** (après voyelle ou consonne,
#   devant voyelle) → *v*
#   ex. : *auoir* → *avoir*, *sauoir* → *savoir*, *trouuer* → *trouver*
#
# **Règle 3** — *vn* en position initiale → *un*
#   ex. : *vn cheualier* → *un chevalier*
#
# **À vous de jouer.** Complétez `normalize_uv` avec ces trois règles.
# Utilisez les constantes `VOWELS` et `CONSONANTS` ci-dessous.

# %% [FOURNI]
VOWELS     = 'aeiouyéèêëàâôùûîï'
CONSONANTS = 'bcdfghjklmnprst'

# %% [TODO]
def normalize_uv(word: str) -> str:
    """
    Normalise l'alternance u/v.

    Règles (dans l'ordre) :
    1. ^u(?=[VOWELS]) → v         (u initial devant voyelle)
    2. (?<=[VOWELS|CONS])u(?=[VOWELS]) → v  (u intervocalique ou post-consonne)
    3. ^vn\\b → un                 (vn initial)

    Indice : re.sub(r'^u(?=[' + VOWELS + r'])', 'v', word)
    """
    # TODO
    pass


# %% [markdown]
# ### 1.3 Alternance i/j
#
# Symétrique à u/v. Deux règles :
#
# **Règle 1** — *i* en position **initiale** devant voyelle → *j*
#   ex. : *iuger* → *juger*, *iour* → *jour*, *iuillet* → *juillet*
#
# **Règle 2** — *j* en position **médiale** après voyelle devant consonne → *i*
#   ex. : *aijde* → *aide* (rare)

# %% [TODO]
def normalize_ij(word: str) -> str:
    """
    Normalise l'alternance i/j.

    Règles :
    1. ^i(?=[VOWELS]) → j
    2. (?<=[VOWELS])j(?=[CONSONANTS]) → i
    """
    # TODO
    pass


# %% [markdown]
# ### 1.4 Règles graphémiques : table de correspondances
#
# Ces règles s'appliquent **après** u/v et i/j, sur les terminaisons
# et graphies morphologiques variables.
#
# **À vous de jouer.** Complétez `apply_grapheme_rules` qui applique les
# règles de `GRAPHEME_RULES` dans l'ordre (des plus spécifiques aux plus
# générales).

# %% [FOURNI]
GRAPHEME_RULES = [
    # Imparfait et conditionnel (du plus long au plus court)
    (r'oient\b',         'aient'),    # savoient → savaient
    (r'roit\b',          'rait'),     # sauroit → saurait
    (r'oit\b',           'ait'),      # savoit → savait
    # -our → -oir (sauf pour, tour, amour)
    (r'(?<![p])our\b',   'oir'),      # pouvoir (garder "pour")
    # Finales en -x médiéval
    (r'iax\b',           'iaux'),     # chevax → chevaux (via iaux)
    (r'aus\b',           'aux'),      # haus → hauts
    (r'aullt?\b',        'aut'),      # haullt → haut
    (r'aulx\b',          'aux'),      # aulx → aux
    # ei → oi
    (r'\bei\b',          'oi'),       # dei → doi
    (r'^ei',             'oi'),       # eiritage → oiritage
    # -ele finale → -elle
    (r'(?<=[a-z])ele\b', 'elle'),     # bele → belle, pucele → pucelle
]

# %% [TODO]
def apply_grapheme_rules(word: str,
                          rules: list = GRAPHEME_RULES) -> str:
    """
    Applique les règles graphémiques dans l'ordre déclaré.

    Pour chaque (pattern, remplacement) dans rules :
        word = re.sub(pattern, remplacement, word)

    L'ordre des règles est une décision linguistique : les règles
    plus spécifiques (terminaisons longues) doivent précéder les
    règles plus générales (terminaisons courtes).
    """
    # TODO
    pass


# %% [markdown]
# **Validation 1**

# %% [FOURNI — validation]
assert normalize_unicode("test\u0065\u0301") == "testé",          "NFC attendu"
assert normalize_uv("auoir")    == "avoir",                        "auoir → avoir"
assert normalize_uv("sauoir")   == "savoir",                       "sauoir → savoir"
assert normalize_uv("uoir")     == "voir",                         "uoir → voir"
assert normalize_uv("vn")       == "un",                           "vn → un"
assert normalize_ij("iuger")    == "juger",                        "iuger → juger"
assert normalize_ij("iour")     == "jour",                         "iour → jour"
assert normalize_ij("iuillet")  == "juillet",                      "iuillet → juillet"
assert apply_grapheme_rules("savoit")    == "savait",              "-oit → -ait"
assert apply_grapheme_rules("vouloit")   == "voulait",             "-oit → -ait"
assert apply_grapheme_rules("savoient")  == "savaient",            "-oient → -aient"
assert apply_grapheme_rules("sauroit")   == "saurait",             "-roit → -rait"
assert apply_grapheme_rules("pucele")    == "pucelle",             "-ele → -elle"
print("Validation 1 : OK")

# %% [markdown]
# ### 1.5 Mesure du CER avant et après les règles graphémiques
#
# **À vous de jouer.** Complétez `compute_cer` et `pipeline_step1`.

# %% [TODO]
def compute_cer(hypotheses: list[str], references: list[str]) -> float:
    """
    Calcule le CER (Character Error Rate) moyen.

    CER = Σ editdistance(hyp, ref) / Σ len(ref)

    Paramètres
    ----------
    hypotheses : list[str]  sorties du pipeline
    references : list[str]  formes attendues (vérité terrain)

    Retourne
    --------
    float  CER moyen (0 = parfait, 1 = entièrement faux)

    Indice : editdistance.eval(hyp, ref)
    """
    # TODO
    pass


def pipeline_step1(text: str) -> str:
    """
    Applique les règles graphémiques de l'Étape 1 à chaque token.

    Opérations dans l'ordre pour chaque mot :
    1. normalize_unicode(mot)
    2. normalize_uv(mot)
    3. normalize_ij(mot)
    4. apply_grapheme_rules(mot)

    Retourne le texte normalisé (espace comme séparateur).
    """
    # TODO
    pass


# %% [FOURNI — mesure CER Étape 1]
CER_BASELINE = compute_cer(SOURCES, REFS)
CER_STEP1    = compute_cer([pipeline_step1(s) for s in SOURCES], REFS)

print(f"CER baseline (HTR brut)     : {CER_BASELINE:.4f}")
print(f"CER après règles graphiques : {CER_STEP1:.4f}")
print(f"Réduction                   : {(CER_BASELINE - CER_STEP1) / CER_BASELINE * 100:.1f} %")

assert CER_STEP1 < CER_BASELINE, "Les règles doivent améliorer le CER"

# %% [markdown]
# ---
# ## Étape 2 — Résolution des abréviations

# %% [markdown]
# ### 2.1 Règle phonologique du tilde de nasalité
#
# Le tilde `~` dans les transcriptions HTR signale une consonne nasale
# supprimée. La règle de résolution dépend de la consonne qui suit :
#
# - *~* devant *b* ou *p* → **m** (assimilation labiale)
#   ex. : *co~bat* → *combat*, *te~ps* → *temps*
# - *~* devant toute autre consonne → **n**
#   ex. : *norm~die* → *normandie*, *fra~ce* → *france*
# - *~* en fin de mot après consonne → **on**
#   ex. : *bo~* → *bon* (cas rare, contextuel)
#
# **À vous de jouer.** Complétez `resolve_nasal_tilde`.

# %% [TODO]
def resolve_nasal_tilde(word: str) -> str:
    """
    Résout les tildes de nasalité (~) selon les règles phonologiques.

    Règles (dans l'ordre) :
    1. ~(?=[bp]) → m
    2. ~(?=[cdfghjklnrstvwxyz]) → n
    3. (?<=[bcdfghjklmnprst])~$ → on  (position finale)

    Exemples attendus :
        norm~die → normandie
        fra~ce   → france
        co~bat   → combat
        te~ps    → temps
    """
    # TODO
    pass


# %% [markdown]
# ### 2.2 Table de résolution contextuelle
#
# Les abréviations non résolvables par règle phonologique seule
# (signes suprascripts, mots grammaticaux, titres) sont couverts
# par la table `ABBREV_TABLE`.
#
# **À vous de jouer.** Complétez `resolve_abbreviations`.

# %% [FOURNI]
ABBREV_TABLE = {
    # Mots grammaticaux
    "q~":    "que",     "Q~":    "Que",    "q~e":   "que",
    "n~":    "nom",     "N~":    "Nom",
    "p~":    "par",     "p~r":   "par",

    # Titres et dignités
    "m^e":   "messire",      "M^e":   "Messire",
    "s^r":   "seigneur",     "S^r":   "Seigneur",
    "m^r":   "monseigneur",  "M^r":   "Monseigneur",
    "pñce":  "prince",       "Pñce":  "Prince",
    "pñ":    "prison",

    # Toponymes fréquents
    "norm~die":  "normandie",  "Norm~die":  "Normandie",
    "co~te":     "conte",      "Co~te":     "Conte",
    "fra~ce":    "france",     "Fra~ce":    "France",
    "champ~e":   "champagne",  "Champ~e":   "Champagne",
    "conq~ise":  "conquise",

    # Monnaies
    "l~":    "livres",   "s~":    "sous",
    "d~":    "deniers",  "l~t":   "livres tournois",
}

# %% [TODO]
def resolve_abbreviations(word: str,
                           table: dict = ABBREV_TABLE) -> tuple[str, bool]:
    """
    Résout une abréviation via table contextuelle puis règle phonologique.

    Algorithme (dans l'ordre) :
    1. Calculer phonological = resolve_nasal_tilde(word)
    2. Si word est dans table → retourner (table[word], True)
    3. Si phonological est dans table → retourner (table[phonological], True)
    4. Si phonological != word → retourner (phonological, True)
    5. Sinon → retourner (word, False)

    Retourne
    --------
    (forme_développée: str, was_resolved: bool)
    """
    # TODO
    pass


# %% [markdown]
# ### 2.3 Pipeline Étape 1 + Étape 2

# %% [TODO]
def pipeline_step2(text: str) -> str:
    """
    Applique les règles graphémiques (Étape 1) puis la résolution
    des abréviations (Étape 2) à chaque token.

    Pour chaque mot :
    1. normalize_unicode, normalize_uv, normalize_ij, apply_grapheme_rules
    2. resolve_abbreviations sur la forme obtenue

    Retourne le texte normalisé.
    """
    # TODO
    pass


# %% [FOURNI — validation et CER Étape 2]
assert resolve_nasal_tilde("norm~die")  == "normandie",  "norm~die → normandie"
assert resolve_nasal_tilde("fra~ce")    == "france",     "fra~ce → france"
assert resolve_nasal_tilde("te~ps")     == "temps",      "te~ps → temps"
assert resolve_nasal_tilde("co~bat")    == "combat",     "co~bat → combat"
assert resolve_abbreviations("s^r")[0]  == "seigneur",   "s^r → seigneur"
assert resolve_abbreviations("pñce")[0] == "prince",     "pñce → prince"
assert resolve_abbreviations("l~")[0]   == "livres",     "l~ → livres"
assert resolve_abbreviations("m^r")[0]  == "monseigneur","m^r → monseigneur"

CER_STEP2 = compute_cer([pipeline_step2(s) for s in SOURCES], REFS)
print(f"CER après abréviations      : {CER_STEP2:.4f}")
print(f"Gain vs Étape 1             : {(CER_STEP1 - CER_STEP2) / CER_STEP1 * 100:.1f} %")
assert CER_STEP2 <= CER_STEP1 + 0.01, "Étape 2 ne doit pas régresser le CER"
print("Validation 2 : OK")

# %% [markdown]
# ---
# ## Étape 3 — Lookup DMF (cache pré-rempli)

# %% [markdown]
# ### 3.1 Cache DMF
#
# Le DMF est accessible via `http://www.atilf.fr/dmf` mais requiert
# un accès réseau. Le cache JSON pré-rempli ci-dessous contient les
# 40 lemmes les plus fréquents du corpus synthétique.
#
# En TP avec réseau disponible, vous pouvez enrichir ce cache avec
# `query_dmf(forme, delay=1.0)` (cf. Chapitre 4 §3.3).

# %% [FOURNI — cache DMF]
DMF_CACHE = {
    "roys":{"lemme":"roi","found":True}, "roy":{"lemme":"roi","found":True},
    "rois":{"lemme":"roi","found":True}, "chartre":{"lemme":"charte","found":True},
    "seellee":{"lemme":"sceller","found":True}, "seel":{"lemme":"sceau","found":True},
    "cheualier":{"lemme":"chevalier","found":True},
    "chastelain":{"lemme":"châtelain","found":True},
    "chastel":{"lemme":"châtel","found":True},
    "palays":{"lemme":"palais","found":True}, "roial":{"lemme":"royal","found":True},
    "moys":{"lemme":"mois","found":True}, "receus":{"lemme":"recevoir","found":True},
    "preudomme":{"lemme":"prud'homme","found":True},
    "seneschal":{"lemme":"sénéchal","found":True},
    "destrier":{"lemme":"destrier","found":True}, "escu":{"lemme":"écu","found":True},
    "pucele":{"lemme":"pucelle","found":True}, "beaute":{"lemme":"beauté","found":True},
    "estoit":{"lemme":"être","found":True}, "fu":{"lemme":"être","found":True},
    "prist":{"lemme":"prendre","found":True}, "dist":{"lemme":"dire","found":True},
    "respondi":{"lemme":"répondre","found":True}, "ala":{"lemme":"aller","found":True},
    "iustice":{"lemme":"justice","found":True}, "iuger":{"lemme":"juger","found":True},
    "iuillet":{"lemme":"juillet","found":True},
    "iour":{"lemme":"jour","found":True}, "iehan":{"lemme":"jean","found":True},
    "deuant":{"lemme":"devant","found":True}, "auoit":{"lemme":"avoir","found":True},
    "auoir":{"lemme":"avoir","found":True}, "seruice":{"lemme":"service","found":True},
    "condamne":{"lemme":"condamner","found":True},
    "angloys":{"lemme":"anglais","found":True},
    "sauoir":{"lemme":"savoir","found":True}, "vn":{"lemme":"un","found":True},
    "restitue":{"lemme":"restituer","found":True},
}

# %% [markdown]
# ### 3.2 Fonction de lookup et pipeline complet Étapes 1+2+3
#
# **À vous de jouer.** Complétez `get_dmf_lemme` et
# `normalize_word_with_dmf`.

# %% [TODO]
def get_dmf_lemme(forme: str, cache: dict) -> str | None:
    """
    Retourne le lemme DMF d'une forme depuis le cache.

    Cherche `forme.lower()` dans cache.
    Retourne cache[forme.lower()]["lemme"] si trouvé, sinon None.
    """
    # TODO
    pass


def normalize_word_with_dmf(word: str, cache: dict) -> tuple[str, str]:
    """
    Normalise un mot par combinaison de règles graphémiques et lookup DMF.

    Algorithme :
    1. Appliquer normalize_unicode, normalize_uv, normalize_ij,
       apply_grapheme_rules sur word.
    2. Appliquer resolve_abbreviations sur le résultat.
    3. Chercher le lemme DMF sur la forme ORIGINALE (word.lower()).
       Si trouvé → retourner (lemme, "dmf").
    4. Sinon → retourner (forme après règles+abrév, "rules").

    Retourne
    --------
    (forme_normalisée: str, source: str)
        source vaut "dmf" si le lemme vient du DMF, "rules" sinon.

    Note : le lookup DMF s'effectue sur la forme ORIGINALE, avant toute
    transformation, car le DMF indexe les formes médiévales attestées.
    """
    # TODO
    pass


def pipeline_step3(text: str, cache: dict = DMF_CACHE) -> str:
    """
    Applique le pipeline complet Étapes 1+2+3 à chaque token.
    """
    return ' '.join(normalize_word_with_dmf(w, cache)[0]
                    for w in text.split())


# %% [FOURNI — validation et CER Étape 3]
w, s = normalize_word_with_dmf("roys", DMF_CACHE)
assert w == "roi" and s == "dmf",      f"roys → roi (dmf), obtenu : {w}, {s}"
w, s = normalize_word_with_dmf("s^r", DMF_CACHE)
assert w == "seigneur" and s == "rules", f"s^r → seigneur (rules)"
w, s = normalize_word_with_dmf("iustice", DMF_CACHE)
assert w == "justice",                 f"iustice → justice, obtenu : {w}"

CER_STEP3 = compute_cer([pipeline_step3(s) for s in SOURCES], REFS)
print(f"CER après DMF               : {CER_STEP3:.4f}")
print(f"Gain vs Étape 2             : {(CER_STEP2 - CER_STEP3) / CER_STEP2 * 100:.1f} %")
print("Validation 3 : OK")

# %% [markdown]
# ---
# ## Étape 4 — Arbitrage CamemBERT MLM

# %% [markdown]
# ### 4.1 Pseudo-log-vraisemblance (PLL) sous CamemBERT
#
# Pour les positions de faible confiance où le HTR hésite entre deux
# caractères, CamemBERT peut arbitrer en calculant la PLL de chaque
# version candidate.
#
# La PLL masque chaque token un par un et somme les log-probabilités :
#
# $$\text{PLL}(x) = \sum_{i} \log P(x_i \mid x_{-i} ; \theta)$$
#
# Référence : Salazar et al. (2020), *Masked Language Model Scoring*, ACL.
#
# **Avertissement durée :** `score_text_mlm` itère sur chaque token,
# ce qui prend 20–40 secondes par ligne sur CPU. Sur le corpus complet
# de 40 lignes, prévoir 15–25 minutes. Lancez l'Étape 5 en parallèle.
#
# **À vous de jouer.** Complétez `score_text_mlm` et
# `arbitrate_candidate`.

# %% [FOURNI — imports MLM]
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

CAMEMBERT_MODEL      = "almanach/camembert-base"
CONFIDENCE_THRESHOLD = 0.75   # seuil en dessous duquel on arbitre

_tokenizer_mlm = None
_model_mlm     = None

def _get_mlm_model():
    """Charge CamemBERT MLM une seule fois (lazy loading)."""
    global _tokenizer_mlm, _model_mlm
    if _tokenizer_mlm is None:
        print("Chargement CamemBERT MLM…")
        _tokenizer_mlm = AutoTokenizer.from_pretrained(CAMEMBERT_MODEL)
        _model_mlm     = AutoModelForMaskedLM.from_pretrained(CAMEMBERT_MODEL)
        _model_mlm.eval()
        print("ok")
    return _tokenizer_mlm, _model_mlm

# %% [TODO]
def score_text_mlm(text: str) -> float:
    """
    Calcule la pseudo-log-vraisemblance (PLL) d'un texte sous CamemBERT.

    Algorithme :
    1. Tokeniser text avec le tokeniseur CamemBERT.
    2. Pour chaque position i (sauf [CLS] et [SEP]) :
       a. Créer une copie masked des input_ids avec masked[i] = mask_token_id.
       b. Passer masked à model_mlm pour obtenir les logits.
       c. Calculer log_softmax(logits[0, i]) → log_prob.
       d. Accumuler log_prob[input_ids[i]].
    3. Retourner la somme accumulée (float).

    Indice :
        inputs = tokenizer(text, return_tensors="pt", max_length=128,
                           truncation=True)
        input_ids = inputs["input_ids"][0]
        with torch.no_grad():
            logits = model(masked.unsqueeze(0),
                           attention_mask=inputs["attention_mask"]).logits
    """
    tokenizer, model = _get_mlm_model()
    # TODO
    pass


def arbitrate_candidate(transcription: str,
                         pos: int,
                         alt_a: str,
                         alt_b: str,
                         score_fn=score_text_mlm) -> tuple[str, float, float]:
    """
    Arbitre entre deux caractères candidats à la position pos.

    Algorithme :
    1. Construire version_a = transcription[:pos] + alt_a + transcription[pos+len(alt_a):]
    2. Construire version_b de façon similaire avec alt_b.
    3. Normaliser phonologiquement les deux versions (resolve_nasal_tilde).
    4. Scorer avec score_fn.
    5. Choisir la version avec le score le plus élevé.

    Retourne
    --------
    (alternative_choisie: str, score_a: float, score_b: float)
    """
    # TODO
    pass


# %% [markdown]
# ### 4.2 Ablation : comparaison des trois stratégies d'arbitrage
#
# **À vous de jouer.** Complétez la comparaison des trois stratégies
# sur les 12 lignes de faible confiance (conf < 0.75) du corpus.

# %% [TODO]
def apply_strategy_a(corpus: list, pipeline_fn) -> list[str]:
    """
    Stratégie A — Confiance seule.
    Applique pipeline_fn à toutes les lignes sans arbitrage MLM.
    """
    # TODO
    pass


def apply_strategy_b(corpus: list, pipeline_fn,
                     score_fn=score_text_mlm,
                     threshold: float = CONFIDENCE_THRESHOLD) -> list[str]:
    """
    Stratégie B — MLM seul.
    Pour chaque ligne, si la ligne contient un candidat (~ ou ^),
    arbitre TOUJOURS avec MLM, sans tenir compte de la confiance HTR.

    Simplification pour le TP : détecter les candidats par présence
    de '~' ou '^' dans la transcription.
    """
    # TODO
    pass


def apply_strategy_c(corpus: list, pipeline_fn,
                     score_fn=score_text_mlm,
                     threshold: float = CONFIDENCE_THRESHOLD) -> list[str]:
    """
    Stratégie C — Combinée (recommandée).
    N'arbitre avec MLM que si confidence < threshold ET la ligne
    contient un candidat ambigu ('~' ou '^').
    Pour les autres lignes, applique pipeline_fn directement.
    """
    # TODO
    pass


# %% [FOURNI — mesure ablation stratégies]
# ATTENTION : cellule coûteuse (~15 min CPU). Exécuter pendant l'Étape 5.
# Décommenter pour lancer :
#
# hyps_a = apply_strategy_a(CORPUS, pipeline_step3)
# hyps_b = apply_strategy_b(CORPUS, pipeline_step3)
# hyps_c = apply_strategy_c(CORPUS, pipeline_step3)
# cer_a  = compute_cer(hyps_a, REFS)
# cer_b  = compute_cer(hyps_b, REFS)
# cer_c  = compute_cer(hyps_c, REFS)
# print(f"Stratégie A (confiance seule) : CER={cer_a:.4f}")
# print(f"Stratégie B (MLM seul)        : CER={cer_b:.4f}")
# print(f"Stratégie C (combinée)        : CER={cer_c:.4f}")

# %% [markdown]
# ---
# ## Étape 5 — Construction des paires d'entraînement
#
# **À vous de jouer.** Complétez `build_normalization_pairs`.

# %% [TODO]
def build_normalization_pairs(corpus:         list,
                               apply_fn:       callable,
                               min_change:     bool  = True,
                               min_confidence: float = 0.0) -> list[dict]:
    """
    Construit les paires (source → normalisé) pour le fine-tuning mT5.

    Pour chaque (src, ref, conf) dans corpus :
    - Appliquer apply_fn(src) pour obtenir la cible.
    - Exclure si min_change=True et que la cible == src (aucune modification).
    - Exclure si conf < min_confidence.
    - Inclure avec le champ sample_weight = conf.

    Retourne
    --------
    list[dict] avec pour chaque paire :
        {"source": str, "target": str, "reference": str, "sample_weight": float}

    Note : "target" est la sortie du pipeline de règles (pseudo-référence
    automatique). "reference" est la vraie vérité terrain humaine (utilisée
    pour l'évaluation uniquement, pas pour l'entraînement).
    """
    pairs = []
    for src, ref, conf in corpus:
        # TODO
        pass
    return pairs


# %% [FOURNI — affichage des paires]
pairs_train = build_normalization_pairs(CORPUS, pipeline_step3,
                                         min_change=True, min_confidence=0.65)
pairs_all   = build_normalization_pairs(CORPUS, pipeline_step3, min_change=False)

print(f"Paires totales              : {len(pairs_all)}")
print(f"Paires avec changement      : {len(pairs_train)}")
print(f"Confidence moyenne (paires) : "
      f"{sum(p['sample_weight'] for p in pairs_train)/max(len(pairs_train),1):.2f}")
print("\nExemples de paires :")
for p in pairs_train[:3]:
    print(f"  SRC    : {p['source'][:60]}")
    print(f"  TARGET : {p['target'][:60]}")
    print(f"  WEIGHT : {p['sample_weight']:.2f}")
    print()

# %% [markdown]
# ### 5.2 Fine-tuning mT5 LoRA (référence au Chapitre 3)
#
# Le fine-tuning de mT5 avec LoRA a été implémenté en détail au Chapitre 3.
# La cellule ci-dessous adapte ce code au problème de normalisation.
# Elle requiert un GPU — lancer après avoir validé les Étapes 1–5.

# %% [FOURNI — fine-tuning mT5]
def finetune_mt5_normalization(pairs:      list[dict],
                                model_name: str   = "google/mt5-small",
                                r:          int   = 8,
                                n_epochs:   int   = 10,
                                output_dir: str   = "./mt5_normalisation_lora",
                                device:     str   = "auto") -> dict:
    """
    Fine-tune mT5-small avec LoRA r=8 sur les paires de normalisation.

    Utilise WeightedSeq2SeqTrainer pour pondérer les exemples par
    la confiance HTR (sample_weight).

    Note CPU : avec 28 paires et 10 epochs, ~20 minutes sur CPU.
    Le code est identique à celui du TP Chapitre 3.
    """
    try:
        from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                                   Seq2SeqTrainingArguments,
                                   DataCollatorForSeq2Seq)
        from peft import LoraConfig, TaskType, get_peft_model
        from datasets import Dataset
        import torch

        device   = ("cuda" if torch.cuda.is_available() else "cpu") \
                   if device == "auto" else device
        use_fp16 = device == "cuda"
        PREFIX   = "normalise moyen français: "

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        lora_config = LoraConfig(
            r               = r,
            lora_alpha      = r * 2,
            target_modules  = ["q", "v"],
            lora_dropout    = 0.1,
            bias            = "none",
            task_type       = TaskType.SEQ_2_SEQ_LM,
        )
        model = get_peft_model(base_model, lora_config)
        model.print_trainable_parameters()

        def tokenize(examples):
            model_inputs = tokenizer(
                [PREFIX + s for s in examples["source"]],
                max_length=128, truncation=True,
            )
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(
                    examples["target"], max_length=128, truncation=True,
                )
            model_inputs["labels"]         = labels["input_ids"]
            model_inputs["sample_weight"]  = examples["sample_weight"]
            return model_inputs

        dataset = Dataset.from_list(pairs).map(
            tokenize, batched=True, remove_columns=["source","target","reference"])

        # Split 80/20
        split   = dataset.train_test_split(test_size=0.2, seed=42)
        train_d = split["train"]
        val_d   = split["test"]

        args = Seq2SeqTrainingArguments(
            output_dir=output_dir, num_train_epochs=n_epochs,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            learning_rate=2e-4, warmup_steps=50,
            predict_with_generate=True, generation_max_length=128,
            eval_strategy="epoch", save_strategy="epoch",
            load_best_model_at_end=True, fp16=use_fp16,
            logging_steps=10, report_to="none",
        )
        from transformers import Trainer

        class WeightedSeq2SeqTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kw):
                weights = inputs.pop("sample_weight", None)
                outputs = model(**inputs)
                loss    = outputs.loss
                if weights is not None:
                    w = torch.tensor(weights, dtype=torch.float,
                                     device=loss.device)
                    loss = (loss * w.mean()).mean()
                return (loss, outputs) if return_outputs else loss

        trainer = WeightedSeq2SeqTrainer(
            model=model, args=args,
            train_dataset=train_d, eval_dataset=val_d,
            data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
            tokenizer=tokenizer,
        )
        trainer.train()
        model.save_pretrained(output_dir)
        return {"status": "ok", "output_dir": output_dir}

    except Exception as e:
        print(f"Fine-tuning non disponible ({e}). Passer à l'Étape 6.")
        return {"status": "skipped", "error": str(e)}

# %% [markdown]
# ---
# ## Étape 6 — Évaluation CER / BLEU / Token Accuracy

# %% [markdown]
# ### 6.1 Token Accuracy
#
# **À vous de jouer.** Complétez `token_accuracy`.

# %% [TODO]
def token_accuracy(hypotheses: list[str], references: list[str]) -> float:
    """
    Calcule le pourcentage de tokens correctement normalisés.

    Pour chaque paire (hyp, ref) :
    - Comparer token par token (hyp.split() vs ref.split()).
    - Compter les tokens identiques.
    - Le total inclut les tokens supplémentaires si les longueurs diffèrent.

    Retourne
    --------
    float  proportion de tokens corrects (entre 0 et 1)
    """
    # TODO
    pass


# %% [FOURNI — calcul des métriques complètes]
hyps_pipeline = [pipeline_step3(s) for s in SOURCES]

cer_final  = compute_cer(hyps_pipeline, REFS)
tok_acc    = token_accuracy(hyps_pipeline, REFS)

# BLEU via sacrebleu (si disponible)
try:
    from evaluate import load as load_metric
    bleu_metric = load_metric("sacrebleu")
    bleu_score  = bleu_metric.compute(
        predictions=hyps_pipeline,
        references=[[r] for r in REFS],
    )["score"]
except Exception:
    bleu_score = None
    print("sacrebleu non disponible — BLEU non calculé")

print(f"\nMétriques finales (pipeline règles + DMF) :")
print(f"  CER          : {cer_final:.4f}")
print(f"  Token Acc    : {tok_acc:.4f}")
if bleu_score is not None:
    print(f"  BLEU-4       : {bleu_score:.2f}")

# %% [markdown]
# ### 6.2 Tableau d'ablation complet
#
# **À vous de jouer.** Complétez `build_ablation_table` avec les résultats
# de toutes les étapes mesurées.

# %% [TODO]
def build_ablation_table(cer_baseline: float,
                          cer_step1:    float,
                          cer_step2:    float,
                          cer_step3:    float,
                          tok_acc:      float,
                          bleu:         float = None,
                          cer_mt5:      float = None,
                          cer_hybrid:   float = None) -> pd.DataFrame:
    """
    Construit le tableau d'ablation complet.

    Retourne un DataFrame avec les colonnes :
        Configuration | CER ↓ | Réduction | BLEU ↑ | Token Acc ↑ | Params | Note

    La réduction est calculée par rapport à la baseline.
    Pour les configurations sans mT5, Params = 0.
    """
    rows = [
        {
            "Configuration": "HTR brut (baseline)",
            "CER ↓": round(cer_baseline, 4),
            "Réduction": "—",
            "BLEU ↑": "—",
            "Token Acc ↑": "—",
            "Params": 0,
            "Note": "Point de départ",
        },
        # TODO : ajouter les lignes pour étapes 1, 2, 3, et optionnellement 4, 5
    ]
    return pd.DataFrame(rows)


# %% [FOURNI — affichage tableau]
df_ablation = build_ablation_table(
    CER_BASELINE, CER_STEP1, CER_STEP2, CER_STEP3,
    tok_acc, bleu_score
)
print("\nTableau d'ablation :")
print(df_ablation.to_string(index=False))

# %% [markdown]
# ---
# ## Étape 7 — Traçabilité

# %% [markdown]
# ### 7.1 CONVENTIONS_NLP.md
#
# **À vous de jouer.** Complétez le document `CONVENTIONS_NLP.md` avec
# vos décisions de normalisation. Le squelette ci-dessous couvre les
# cas documentés dans le cours ; ajoutez vos propres cas ambigus.

# %% [TODO]
def generate_conventions_doc(version:    str = "1.0.0",
                              split_hash: str = SPLIT_HASH,
                              extra_cases: list = None) -> str:
    """
    Génère le contenu de CONVENTIONS_NLP.md.

    Paramètres
    ----------
    version    : str   version du module de règles (sémantique : MAJEUR.MINEUR.PATCH)
    split_hash : str   SHA-256 du split (pour reproductibilité)
    extra_cases: list  cas ambigus supplémentaires, format :
                       [{"forme":"...","choix":"...","alt":"...","justif":"..."}]

    Retourne
    --------
    str  contenu Markdown du fichier CONVENTIONS_NLP.md
    """
    base_cases = [
        {"forme":"`p~`", "choix":"`par`",    "alt":"`per`, `pro`",
         "justif":"Devant substantif dans les chartes → `par` dominant"},
        {"forme":"`no~`","choix":"`nom`",    "alt":"`non`, `notre`",
         "justif":"DMF : `no~` devant consonne = `nom` dans les chartes normandes"},
        {"forme":"`q~`", "choix":"`que`",    "alt":"`qui` (devant voyelle)",
         "justif":"Convention t9n — `que` par défaut"},
        {"forme":"`s^r`","choix":"`seigneur`","alt":"`sir` (anglicisme)",
         "justif":"Corpus francophone → `seigneur`"},
    ]
    if extra_cases:
        base_cases.extend(extra_cases)

    # TODO : construire la chaîne Markdown du document
    # Structure attendue :
    # # CONVENTIONS_NLP.md
    # ## Version : {version} | Date : {date} | Split SHA-256 : {split_hash}
    # ### Règles appliquées
    # ### Cas ambigus résolus (tableau Markdown)
    # ### Formes non résolues
    pass


# %% [FOURNI — sauvegarde]
conventions_content = generate_conventions_doc()
if conventions_content:
    with open("CONVENTIONS_NLP.md", "w", encoding="utf-8") as f:
        f.write(conventions_content)
    print("Livrable : CONVENTIONS_NLP.md")

# %% [markdown]
# ### 7.2 Journal d'expériences
#
# **À vous de jouer.** Complétez `log_experiment` et enregistrez les
# résultats des étapes 1 à 3.

# %% [TODO]
def log_experiment(config:     dict,
                   metrics:    dict,
                   split_hash: str,
                   path:       str = "experiments/journal.jsonl") -> None:
    """
    Ajoute une entrée au journal d'expériences.

    Chaque entrée est un objet JSON sur une seule ligne (JSONL) contenant :
    - "timestamp" : datetime UTC ISO
    - "split_hash": str
    - "config"    : dict  paramètres de l'expérience
    - "metrics"   : dict  résultats mesurés

    Crée le dossier `experiments/` si absent.
    Appende au fichier (ne pas écraser les entrées précédentes).
    """
    # TODO
    pass


# %% [FOURNI — logging des étapes]
if log_experiment.__doc__:  # skip si non implémenté
    for step, cer, method in [
        (1, CER_STEP1, "rules_graphemic"),
        (2, CER_STEP2, "rules_abbrev"),
        (3, CER_STEP3, "rules_dmf"),
    ]:
        log_experiment(
            config  = {"step": step, "method": method,
                        "n_rules": len(GRAPHEME_RULES),
                        "abbrev_table_size": len(ABBREV_TABLE),
                        "dmf_cache_size": len(DMF_CACHE)},
            metrics = {"cer": cer,
                        "reduction_vs_baseline":
                            round((CER_BASELINE - cer) / CER_BASELINE, 4)},
            split_hash = SPLIT_HASH,
        )
    print("Journal d'expériences mis à jour.")

# %% [markdown]
# ---
# ## Récapitulatif — Ce que vous avez implémenté
#
# | Étape | Fonction(s) | Lien avec le cours |
# |---|---|---|
# | 1a Unicode | `normalize_unicode` | Ch4 §2.2 |
# | 1b u/v | `normalize_uv` | Ch4 §1.4 |
# | 1c i/j | `normalize_ij` | Ch4 §1.4 |
# | 1d Graphèmes | `apply_grapheme_rules` | Ch4 §2.3 |
# | 1e CER | `compute_cer`, `pipeline_step1` | Ch4 §2.4 |
# | 2a Tilde | `resolve_nasal_tilde` | Ch4 §4.2 |
# | 2b Abrév. | `resolve_abbreviations` | Ch4 §4.3 |
# | 3a DMF | `get_dmf_lemme`, `normalize_word_with_dmf` | Ch4 §3.3–3.5 |
# | 4 MLM | `score_text_mlm`, `arbitrate_candidate` | Ch4 §5.2 |
# | 4b Ablation | Stratégies A/B/C | Ch4 §5.3 |
# | 5 Paires | `build_normalization_pairs` | Ch4 §6.1 |
# | 6a Acc | `token_accuracy` | Ch4 §6.2 |
# | 6b Ablation | `build_ablation_table` | Ch4 §6.3 |
# | 7a Conv. | `generate_conventions_doc` | Ch4 §7.1 |
# | 7b Journal | `log_experiment` | Ch4 §7.3 |
#
# **Livrables attendus :**
# `CONVENTIONS_NLP.md`, `experiments/journal.jsonl`,
# et le tableau d'ablation affiché en Étape 6.
