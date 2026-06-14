# Chapitre 4 — Normalisation orthographique du moyen français : règles, lexiques et fine-tuning

**Module NLP · Master Data/IA · MD5 Volet 2 · 2026**  
TP autonome — 3 heures 30

---

## Avant-propos : la priorité des règles

Il est tentant, dans un module centré sur les modèles de langage, de passer directement au fine-tuning. Le Chapitre 3 vous en a montré les rouages. Mais avant de toucher à mT5, vous allez passer plusieurs heures à écrire des règles. Ce n'est pas un détour — c'est la séquence correcte, pour trois raisons qui s'enchaînent.

La première est empirique : les règles graphiques couvrent une fraction déterministe et documentable du problème. Sur un corpus de chartes médiévales, 60 à 75 % des divergences entre une transcription HTR et sa forme normalisée attendue sont des alternances graphiques systématiques — *u/v*, *i/j*, *roy/roi*, terminaisons déclinées — que l'on peut résoudre sans jamais faire appel à un modèle. Un neural qui apprend ce que les règles savent déjà brûle de la capacité de généralisation sur du bruit structuré.

La deuxième est épistémique : les règles sont traçables. Dans le contexte des humanités numériques, chaque décision de normalisation doit pouvoir être justifiée, auditée, et réversible. Le fichier `CONVENTIONS_NLP.md` que vous produirez est un artefact scientifique — les philologues qui utiliseront vos données dans cinq ans doivent pouvoir reconstruire exactement ce que vous avez fait et pourquoi.

La troisième est instrumentale : les règles servent de baseline pour évaluer le neural. Si votre modèle mT5 ne fait pas mieux que les règles sur les cas simples, quelque chose ne va pas — dans le corpus, dans l'entraînement, ou dans les deux.

Ce chapitre est organisé en quatre étapes correspondant exactement au TP de l'après-midi.

---

## 1. Éléments de linguistique du moyen français pour ingénieurs

Cette section ne prétend pas remplacer un cours de philologie romane. Son objectif est précis : vous donner les outils conceptuels strictement nécessaires pour écrire des règles de normalisation justes, sans commettre des erreurs que vous ne sauriez même pas repérer.

### 1.1 Qu'est-ce que le moyen français ?

Le moyen français désigne le français écrit entre environ 1330 et 1500, entre la fin de l'ancien français et le début du français préclassique. C'est la langue des grandes chroniques (Froissart, Commynes), des premières grandes compilations juridiques, et des documents administratifs — chartes, comptes, registres — qui constituent l'essentiel des corpus HTR que vous traitez.

Ses deux caractéristiques les plus importantes pour votre travail sont les suivantes.

**L'orthographe n'est pas standardisée.** Il n'existe pas d'Académie française au XIVe siècle. Un même mot peut s'écrire de cinq façons différentes dans le même document, selon le scribe, sa région d'origine, sa fatigue, ou la place disponible sur la ligne. *Roi* s'écrit *roi*, *roy*, *roÿ*, *roys* (au pluriel), *rei*, *rois* selon les textes et les scribes. Ce n'est pas des fautes — c'est la norme de l'époque.

**La morphologie est encore flexionnelle.** Le moyen français conserve des traces du système de déclinaison latin — notamment le cas sujet (*li roys*) opposé au cas régime (*le roi*) pour les noms masculins. Ces alternances sont linguistiquement significatives et ne doivent pas toutes être "corrigées" de façon automatique. La distinction entre *li* (article cas sujet masculin singulier) et *le* (article cas régime) est une information morphosyntaxique, pas une erreur.

### 1.2 Le signe linguistique et la variation graphique

Ferdinand de Saussure distinguait le *signifiant* (la forme sonore ou graphique) du *signifié* (le concept). Dans votre corpus, vous avez affaire à une situation où le même signifié peut avoir des dizaines de signifiants graphiques différents. La normalisation consiste à ramener ces variantes de signifiant vers une forme canonique — ce que les linguistes appellent le *lemme*.

Le **lemme** est la forme de citation d'un mot : la forme infinitive pour un verbe (*signer*, pas *signa*), la forme du cas régime singulier pour un nom en moyen français (*roi*, pas *li roys*), la forme masculine singulier pour un adjectif. C'est la forme sous laquelle vous trouverez le mot dans le Dictionnaire du Moyen Français.

Le **morphème** est la plus petite unité porteuse de sens. Dans *signait*, on distingue le morphème lexical *sign-* (la racine), le morphème de temps *-ait* (imparfait), et la désinence de personne. Pour la normalisation, ce qui intéresse est d'abord la **racine graphique** — la partie qui permet d'identifier le lemme — et les **affixes flexionnels** qui encodent les informations grammaticales.

La **variation graphique** en moyen français opère à plusieurs niveaux que l'on peut hiérarchiser par régularité décroissante :

**Niveau 1 — Alternances phonographiques systématiques :** certains sons peuvent s'écrire de plusieurs façons sans règle contextuellement déterminée. C'est le cas de *u/v* (les deux lettres représentent indifféremment le son [u] ou [v] selon leur position), de *i/j* (même logique), et de *c/k/qu* devant voyelle vélaire. Ces alternances sont entièrement résolues par des règles de position.

**Niveau 2 — Variantes morphographiques stables :** des terminaisons qui correspondent à un même morphème fonctionnel prennent des formes différentes selon les scribes ou les régions. *-oit* et *-ait* sont deux graphies de l'imparfait ; *-eur* et *-eur* et *-uer* pour la même terminaison verbale. Ces variantes sont résolvables par une table de correspondances.

**Niveau 3 — Variantes lexicales scripto-régionales :** le même mot a une forme entièrement différente selon la région d'origine du scribe. *Chastel* (picard) vs *château* (francs), *mès* (normand) vs *mais* (général). Ces variantes nécessitent un lexique de référence.

**Niveau 4 — Abréviations conventionnelles :** les scribes utilisent des signes abréviatifs normalisés dans leur pratique professionnelle. Ce niveau est traité séparément en section 3.

### 1.3 Lemme, forme et type : vocabulaire opérationnel

Ces trois concepts sont directement utilisés dans le code du TP.

Un **type** (*type*) est une forme graphique distincte telle qu'elle apparaît dans le texte. *Roys*, *roy*, *roi*, *rois* sont quatre types différents.

Une **forme** (*token*) est une occurrence d'un type dans le texte. Si *roys* apparaît 17 fois dans votre corpus, c'est un type mais 17 occurrences (tokens).

Un **lemme** (*lemma*) est la forme canonique qui regroupe tous ces types. *Roi* est le lemme des types *roys*, *roy*, *roi*, *rois*, *rei*, etc.

La normalisation consiste à mapper des types médiévaux vers leurs lemmes canoniques (ou, dans votre cas, vers leurs équivalents en français moderne standardisé). Ce mapping est ce que votre module de règles doit produire.

### 1.4 Phonologie historique : les alternances à connaître

Voici les alternances phonographiques les plus fréquentes dans les corpus médiévaux, avec leur explication phonologique, et la règle qui en découle.

**Alternance u/v :** en latin médiéval et en moyen français, *u* et *v* sont deux graphies de la même lettre, utilisées selon la position (initiale de mot ou de syllabe pour *v*, médiale pour *u*). Le son [v] en position médiale s'écrit *u* : *auoir* (avoir), *sauoir* (savoir), *iuger* (juger). La règle de normalisation est positionnelle : *u* précédé d'une consonne ou en position intervocalique représente [v] et doit être normalisé en *v* ; *u* en position finale ou après voyelle représente [u].

```python
import re

def normalize_uv(word: str) -> str:
    """
    Normalise l'alternance u/v en moyen français.
    Règle : u en position initiale devant voyelle → v ;
            v en position médiale après voyelle → u (rare mais possible).
    """
    # u initial devant voyelle (auoir → avoir, uoir → voir)
    word = re.sub(r'^u(?=[aeiouéèêëàâôùûîï])', 'v', word)
    # u médial entre consonnes ou après consonne devant voyelle
    # Ex : sauoir → savoir, trouuer → trouver
    word = re.sub(r'(?<=[bcdfghjklmnprst])u(?=[aeiouéèêëàâôùûîï])', 'v', word)
    return word

# Test
assert normalize_uv("auoir") == "avoir"
assert normalize_uv("sauoir") == "savoir"
assert normalize_uv("uoir")   == "voir"
```

**Alternance i/j :** symétrique à u/v. *i* en position initiale devant voyelle représente [j] et doit être normalisé en *j* : *iuger* → *juger*, *iour* → *jour*, *iustice* → *justice*.

```python
def normalize_ij(word: str) -> str:
    """Normalise l'alternance i/j."""
    # i initial devant voyelle
    word = re.sub(r'^i(?=[aeiouéèêëàâôùûîï])', 'j', word)
    # j médial devant consonne (rare) → i : ex. 'aijde' → 'aide'
    word = re.sub(r'(?<=[aeiouéèêëàâôùûîï])j(?=[bcdfghjklmnprst])', 'i', word)
    return word

assert normalize_ij("iuger")  == "juger"
assert normalize_ij("iour")   == "jour"
```

**Terminaisons désinentielles variables :** les finales de l'imparfait (*-oit/-ait*), du conditionnel (*-roit/-rait*), et de nombreux substantifs (*-eur/-eur*, *-our/-oir*) varient selon les scribes. Ces alternances sont listées dans une table de correspondances.

**Alternance c/k/qu :** le son [k] s'écrit *c* devant *a*, *o*, *u* (latin), *qu* devant *e*, *i*, et parfois *k* dans des textes d'influence nordique. La normalisation préfère la graphie *c/qu* selon la voyelle suivante.

---

## 2. Étape 1 — Module de règles graphiques

### 2.1 Architecture du module

Le module de règles est un pipeline séquentiel. Chaque règle est une fonction pure : elle prend une chaîne de caractères, retourne une chaîne normalisée, et est testable indépendamment. Les règles sont appliquées dans un ordre précis, car certaines sont dépendantes du résultat des précédentes.

```
texte brut
    │
    ▼
[1] Normalisation Unicode (NFC, caractères spéciaux médiévaux)
    │
    ▼
[2] Alternances graphiques systématiques (u/v, i/j, c/k/qu)
    │
    ▼
[3] Terminaisons variables (table de correspondances morphographiques)
    │
    ▼
[4] Lookup DMF / LGeRM (vérification de la forme normalisée)
    │
    ▼
[5] Mesure CER avant/après
```

### 2.2 Normalisation Unicode

Le premier problème est souvent invisible : les caractères médiévaux spéciaux ne sont pas toujours encodés de façon cohérente par les différents outils HTR. La normalisation Unicode NFC (*Canonical Decomposition, followed by Canonical Composition*) résout les ambiguïtés d'encodage les plus courantes : un *é* peut être encodé comme un seul codepoint U+00E9 ou comme *e* + accent aigu combinant (U+0065 + U+0301) — NFC unifie les deux.

```python
import unicodedata, re

# Table de translittération des caractères médiévaux spéciaux
# vers leurs équivalents Unicode standard ou leur expansion
MEDIEVAL_UNICODE = {
    'ꝑ': 'p',    # p barré (per/par/pro) — résolu en section 3
    'ꝓ': 'pro',  # p barré avec crochet
    'ꝕ': 'p',    # variante p barré
    '\u0304': '', # macron combinant (souvent marque de nasale) → traité en section 3
    'ȷ':  'j',   # j sans point
    'ı':  'i',   # i sans point (latin)
    '꜀':  'c',   # c superscrit
    '꜁':  'e',   # variante
}

def normalize_unicode(text: str) -> str:
    """
    Étape 1 : normalisation Unicode et translittération des
    caractères médiévaux spéciaux.
    """
    # Normalisation NFC
    text = unicodedata.normalize('NFC', text)
    # Translittération des caractères médiévaux
    for src, tgt in MEDIEVAL_UNICODE.items():
        text = text.replace(src, tgt)
    return text
```

### 2.3 Table de correspondances graphiques

La table de correspondances est le cœur du module de règles. Elle mappe des terminaisons ou des formes entières vers leurs équivalents normalisés. Elle est construite sur la base du DMF et de la convention *t9n* (voir section 5) :

```python
# Substitutions de terminaisons — ordre important : du plus long au plus court
GRAPHEME_RULES = [
    # Imparfait et conditionnel : -oit → -ait
    (r'oit\b',   'ait'),
    (r'oient\b', 'aient'),
    (r'roit\b',  'rait'),
    # Suffixe -our → -oir (vouloir, pouvoir, savoir)
    (r'our\b',   'oir'),
    # Terminaison -x pour -s (vieux, eaux)
    (r'iax\b',   'iaux'),
    (r'aus\b',   'aux'),
    # Graphie -oi- médiévale pour -oi- moderne (souvent stable)
    # mais : -ei- → -oi- (rei → roi)
    (r'\bei\b',  'oi'),
    (r'^ei',     'oi'),
    # Doublement de consonne devant e muet final : belle/bele → belle
    (r'ele\b',   'elle'),
    # Finale -z pour -s dans certains paradigmes verbaux
    (r'ez\b',    'ez'),   # conserver -ez (forme verbale 2e pers pl)
    # -aulx, -aullt → -aut (hault → haut)
    (r'aullt?\b', 'aut'),
    (r'aulx\b',  'aux'),
]

def apply_grapheme_rules(word: str,
                          rules: list = GRAPHEME_RULES) -> str:
    """
    Applique les règles graphémiques dans l'ordre déclaré.
    Les règles plus longues (plus spécifiques) doivent précéder
    les règles plus courtes (plus générales).
    """
    for pattern, replacement in rules:
        word = re.sub(pattern, replacement, word)
    return word
```

**Avertissement sur l'ordre des règles :** appliquer `-oit → -ait` avant `-oit → -oit` n'est pas le même résultat qu'à l'envers. Dans ce module, l'ordre des règles est une décision linguistique documentée dans `CONVENTIONS_NLP.md`, pas un détail d'implémentation.

### 2.4 Mesure du CER avant et après normalisation

Le *Character Error Rate* (CER) mesure la distance d'édition entre la transcription normalisée et la référence attendue, normalisée par la longueur de la référence :

$$\text{CER} = \frac{S + D + I}{N}$$

où $S$ est le nombre de substitutions, $D$ le nombre de suppressions, $I$ le nombre d'insertions de caractères, et $N$ la longueur de la référence.

```python
import editdistance   # pip install editdistance

def compute_cer(hypotheses: list[str], references: list[str]) -> float:
    """
    Calcule le CER moyen sur un ensemble de paires (hypothèse, référence).

    Paramètres
    ----------
    hypotheses : list[str]  transcriptions normalisées (sorties du pipeline)
    references : list[str]  formes attendues (vérité terrain)

    Retourne
    --------
    float  CER moyen (entre 0 et 1 ; 0 = parfait, 1 = entièrement faux)
    """
    total_dist  = 0
    total_chars = 0
    for hyp, ref in zip(hypotheses, references):
        total_dist  += editdistance.eval(hyp, ref)
        total_chars += len(ref)
    return total_dist / max(total_chars, 1)

# Baseline avant normalisation (HTR brut)
cer_avant = compute_cer(transcriptions_brutes, references)
# Après application du module de règles
cer_apres = compute_cer(
    [apply_grapheme_rules(normalize_uv(normalize_ij(normalize_unicode(t))))
     for t in transcriptions_brutes],
    references
)
print(f"CER avant normalisation : {cer_avant:.4f}")
print(f"CER après règles        : {cer_apres:.4f}")
print(f"Réduction               : {(cer_avant - cer_apres)/cer_avant*100:.1f} %")
```

La réduction de CER obtenue par les règles seules est votre **baseline**. Sur un corpus de chartes normandes, une réduction de 15 à 30 % est typique. Si vous observez moins de 10 %, vos règles sont trop conservatrices ou le CER initial est déjà bas. Si vous observez plus de 40 %, vérifiez que votre référence est bien du français moderne normalisé et non du moyen français déjà "propre".

---

## 3. Le Dictionnaire du Moyen Français (DMF) et LGeRM

### 3.1 Qu'est-ce que le DMF

Le Dictionnaire du Moyen Français est un dictionnaire de référence scientifique élaboré par l'ATILF (Analyse et Traitement Informatique de la Langue Française, CNRS – Université de Lorraine). Il couvre le français des XIVe et XVe siècles (1330–1500) et représente l'équivalent pour le moyen français de ce qu'est le *Trésor de la Langue Française* pour le français moderne. À sa version 2023, il comprend plus de 65 000 entrées avec 470 000 exemples contextuels, avec des définitions en français moderne.

Le DMF est accessible à l'adresse `http://www.atilf.fr/dmf`. C'est la seule adresse pérenne à retenir pour accéder à toutes les données liées aux projets et sous-projets du Dictionnaire du Moyen Français.

Son rôle dans votre pipeline est double : il fournit la **forme lemmatisée canonique** de chaque mot médiéval, et il documente les **variantes graphiques** attestées pour chaque lemme. Avant d'écrire une règle, vérifiez qu'elle est cohérente avec le DMF : si vous normalisez *chastel* en *château*, vérifiez que le lemme DMF est bien *château* (il l'est), et que *chastel* est bien une variante attestée de ce lemme.

### 3.2 LGeRM : le lemmatiseur intégré au DMF

LGeRM (*Lemmes, Graphies et Règles Morphologiques*) est le moteur de lemmatisation développé par Gilles Souvay (ATILF) qui est intégré au DMF. Il propose une solution s'appuyant sur une base de formes connues lemmatisées et sur un ensemble de règles graphémiques et morphologiques spécifiques de la langue médiévale. LGeRM permet de faciliter la consultation d'un dictionnaire, l'interrogation et la lemmatisation de textes médiévaux, et trouve des applications dans l'édition électronique de manuscrits et la construction automatique de glossaires.

LGeRM est initialement un lemmatiseur conçu pour gérer la variation graphique des états anciens du français. Il a été développé pour le moyen français (1330-1500) puis adapté au français du XVIe et XVIIe siècles.

L'interface en ligne du DMF intègre LGeRM : si vous entrez *roys* dans le champ de recherche, LGeRM proposera le lemme *roi* avec ses variantes graphiques attestées et les définitions correspondantes. Cette fonctionnalité est accessible via l'URL de recherche du DMF.

### 3.3 Accès programmatique au DMF

Le DMF n'expose pas d'API REST publique documentée. L'accès programmatique se fait par scraping HTML de l'interface web, en respectant les conditions d'utilisation de l'ATILF et en limitant le débit des requêtes.

```python
import requests
from bs4 import BeautifulSoup
import time, re

DMF_SEARCH_URL = "http://www.atilf.fr/dmf/definition/{forme}"

def query_dmf(forme: str, delay: float = 1.0) -> dict:
    """
    Interroge le DMF pour une forme graphique médiévale.

    Paramètres
    ----------
    forme : str   forme graphique à rechercher (ex. "roys", "chastel")
    delay : float délai en secondes entre deux requêtes (respect du serveur)

    Retourne
    --------
    dict avec :
        "lemme"     : str   lemme canonique proposé par LGeRM (ou None)
        "definition": str   première définition trouvée (ou None)
        "variantes" : list  formes graphiques attestées dans le DMF
        "found"     : bool  True si une entrée a été trouvée
    """
    url = DMF_SEARCH_URL.format(forme=forme.lower())
    time.sleep(delay)   # politesse vis-à-vis du serveur ATILF

    try:
        resp = requests.get(url, timeout=10,
                            headers={"User-Agent": "NLP-medieval-research/1.0"})
        if resp.status_code != 200:
            return {"lemme": None, "definition": None,
                    "variantes": [], "found": False}

        soup = BeautifulSoup(resp.text, "html.parser")

        # Extraction du lemme (balise <h2> ou <span class="vedette">)
        vedette = soup.find("span", class_="vedette") or soup.find("h2")
        lemme = vedette.get_text(strip=True) if vedette else None

        # Extraction de la première définition
        defn_tag = soup.find("div", class_="definition") or \
                   soup.find("p", class_="def")
        definition = defn_tag.get_text(strip=True)[:200] if defn_tag else None

        # Extraction des variantes graphiques
        variantes_tags = soup.find_all("span", class_="forme-variante") or \
                         soup.find_all("span", class_="graphie")
        variantes = [t.get_text(strip=True) for t in variantes_tags]

        return {
            "lemme":      lemme,
            "definition": definition,
            "variantes":  variantes,
            "found":      lemme is not None,
        }

    except requests.RequestException as e:
        return {"lemme": None, "definition": None,
                "variantes": [], "found": False, "error": str(e)}

# Exemple d'utilisation
result = query_dmf("roys")
print(f"Lemme     : {result['lemme']}")
print(f"Définition: {result['definition']}")
print(f"Variantes : {result['variantes']}")
```

**Avertissement pratique :** l'interface du DMF est une application web CGI héritée. La structure HTML peut varier entre les versions du DMF (DMF2012, DMF2020, DMF2023). Si les sélecteurs CSS ci-dessus ne retournent rien, inspecter la page manuellement avec `soup.prettify()` et adapter.

### 3.4 Constitution du cache DMF

Pour éviter de solliciter le serveur ATILF à chaque run du pipeline, constituez un cache local JSON des résultats DMF. Ce cache est aussi un artefact de traçabilité : il documente exactement quelles requêtes ont produit quels résultats à quelle date.

```python
import json
from pathlib import Path

DMF_CACHE_PATH = Path("data/dmf_cache.json")

def load_dmf_cache() -> dict:
    if DMF_CACHE_PATH.exists():
        with open(DMF_CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_dmf_cache(cache: dict) -> None:
    DMF_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DMF_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

def get_dmf_lemme(forme: str, cache: dict) -> str | None:
    """
    Retourne le lemme DMF d'une forme, en utilisant le cache si disponible.
    Met à jour le cache en cas de requête réseau.
    """
    if forme in cache:
        return cache[forme].get("lemme")

    result = query_dmf(forme)
    cache[forme] = result
    save_dmf_cache(cache)
    return result.get("lemme")

# Utilisation dans le pipeline
dmf_cache = load_dmf_cache()

# Pré-remplissage du cache sur les types les plus fréquents du corpus
types_freq = sorted(
    {w for line in df["transcription"] for w in line.split()},
    key=lambda w: -sum(1 for l in df["transcription"] if w in l.split())
)[:200]   # top 200 types seulement — assez pour couvrir 80 % des tokens

for forme in types_freq:
    if forme not in dmf_cache:
        lemme = get_dmf_lemme(forme, dmf_cache)
        print(f"  {forme:20s} → {lemme or '(non trouvé)'}")
```

### 3.5 Intégration DMF dans le module de règles

Le lookup DMF est appliqué après les règles graphémiques automatiques : il valide et complète les normalisations que les règles ne couvrent pas.

```python
def normalize_word_with_dmf(word: str, dmf_cache: dict,
                              fallback_rules: bool = True) -> str:
    """
    Normalise un mot en combinant règles graphémiques et lookup DMF.

    Stratégie :
    1. Appliquer les règles graphémiques (u/v, i/j, terminaisons).
    2. Chercher dans le cache DMF.
    3. Si trouvé → retourner le lemme DMF.
    4. Si non trouvé et fallback_rules → retourner la forme après règles.
    5. Si non trouvé et pas fallback → retourner la forme originale.

    Le fallback sur les règles est recommandé en production :
    le DMF ne couvre pas tous les hapax ni les noms propres.
    """
    # Étape 1 : règles graphémiques
    normalized = normalize_unicode(word)
    normalized = normalize_uv(normalized)
    normalized = normalize_ij(normalized)
    normalized = apply_grapheme_rules(normalized)

    # Étape 2 : lookup DMF sur la forme originale
    lemme = get_dmf_lemme(word.lower(), dmf_cache)

    if lemme:
        # Préserver la casse : si le mot original est en majuscule
        if word[0].isupper():
            return lemme.capitalize()
        return lemme

    # Fallback
    return normalized if fallback_rules else word
```

---

## 4. Étape 2 — Résolution des abréviations résiduelles

### 4.1 Taxonomie des abréviations médiévales

Le Chapitre 2 a inventorié les abréviations présentes dans votre corpus. Ce chapitre explique leur structure linguistique pour que vous puissiez les résoudre correctement.

Les abréviations médiévales manuscrites suivent des conventions calligraphiques héritées de la pratique latine. Elles se répartissent en quatre catégories selon leur mode de formation.

**Suspension :** la fin du mot est coupée, signalée par un signe abréviatif. *S.* pour *saint*, *m.* pour *monsieur*. Dans les transcriptions HTR, la suspension apparaît comme une troncature sans marqueur explicite.

**Contraction :** des lettres médianes sont supprimées, les lettres restantes sont reliées par un trait ou un tilde. *norm~die* pour *normandie* (lettres *an* supprimées), *co~te* pour *conte* (lettre *n* supprimée). C'est le cas le plus fréquent dans votre corpus.

**Lettre suscrite :** une lettre est écrite en exposant au-dessus d'une autre pour indiquer la syllabe manquante. *m^r* pour *monseigneur*, *s^r* pour *seigneur*, *p^e* pour *prince* (dans certaines graphies). Les modèles HTR retranscrivent ces exposants avec le caractère `ñ` ou un tilde.

**Signes suprascripts spéciaux :** le tilde de nasalité indique une consonne nasale supprimée. Devant *b* ou *p*, la nasale supprimée est *m* (*co~pte* → *compte*) ; devant toute autre consonne, c'est *n* (*norm~die* → *normandie*, *fra~ce* → *france*). Le *p barré* (`ꝑ`) représente *par*, *per*, ou *pro* selon le contexte.

### 4.2 Règles de résolution par contexte phonologique

La résolution des tildes de nasalité est régie par des règles phonologiques déterministes dans la grande majorité des cas :

```python
import re

def resolve_nasal_tilde(word: str) -> str:
    """
    Résout les tildes de nasalité (~) selon les règles phonologiques
    du moyen français.

    Règles appliquées dans l'ordre :
    1. Devant b ou p : la nasale est m (co~bat → combat, te~ps → temps)
    2. Devant toute autre consonne : la nasale est n
    3. En position finale (q~) : résolution via table contextuelle

    Exemples :
      norm~die → normandie
      co~te    → conte
      te~ps    → temps
      co~bat   → combat
      q~       → que  (via table, voir section suivante)
    """
    # Règle 1 : ~ suivi de b ou p → m
    word = re.sub(r'~(?=[bp])', 'm', word)

    # Règle 2 : ~ suivi d'une autre consonne → n
    word = re.sub(r'~(?=[cdfghjklnrstvwxyz])', 'n', word)

    # Règle 3 : ~ en fin de mot (après une consonne) → on (son, bon, don...)
    #           — cas ambigu, traité par table contextuelle
    word = re.sub(r'(?<=[bcdfghjklmnprst])~$', 'on', word)

    return word

# Tests
assert resolve_nasal_tilde("norm~die")  == "normandie"
assert resolve_nasal_tilde("co~te")     == "conte"
assert resolve_nasal_tilde("te~ps")     == "temps"
assert resolve_nasal_tilde("co~bat")    == "combat"
assert resolve_nasal_tilde("fra~ce")    == "france"
```

### 4.3 Table de résolution contextuelle pour les cas ambigus

Certaines abréviations ne sont pas résolvables par règle phonologique seule : leur expansion dépend du mot entier ou du contexte. La table contextuelle est construite à partir du DMF et des conventions de votre corpus, en distinguant les formules latines, les titres, les termes monétaires, et les mots courants.

```python
# Table principale : formes abrégées → forme développée
# Organisée par domaine pour faciliter l'audit
ABBREV_TABLE = {
    # ── Pronoms et mots grammaticaux ─────────────────────────────────────
    "q~":    "que",       "Q~":    "Que",
    "q~e":   "que",
    "ql":    "quel",      "qls":   "quels",

    # ── Formules latines fréquentes dans les chartes ──────────────────────
    "n~":    "nom",       "N~":    "Nom",
    "p~":    "par",
    "p~r":   "par",       "p~mi":  "parmi",
    "p~ch":  "parce",

    # ── Titres et dignités ───────────────────────────────────────────────
    "m^e":   "messire",   "M^e":   "Messire",
    "s^r":   "seigneur",  "S^r":   "Seigneur",
    "m^r":   "monseigneur","M^r":  "Monseigneur",
    "pñce":  "prince",    "Pñce":  "Prince",
    "pñ":    "prison",

    # ── Termes géographiques fréquents ───────────────────────────────────
    "norm~die": "normandie", "Norm~die": "Normandie",
    "co~te":    "conte",     "Co~te":    "Conte",
    "champ~e":  "champagne", "Champ~e":  "Champagne",
    "bret~e":   "bretagne",  "Bret~e":   "Bretagne",

    # ── Monnaies et mesures ──────────────────────────────────────────────
    "l~":    "livres",
    "s~":    "sous",
    "d~":    "deniers",
    "l~t":   "livres tournois",
    "l~p":   "livres parisis",
}

def resolve_abbreviations(word: str,
                           table: dict = ABBREV_TABLE) -> tuple[str, bool]:
    """
    Résout une abréviation via la table contextuelle.

    Retourne
    --------
    (forme_développée, was_resolved) :
        - forme_développée : str  mot résolu ou mot original si non résolu
        - was_resolved     : bool True si une résolution a été trouvée

    La distinction résolu/non résolu est utile pour l'annotation
    des cas ambigus dans CONVENTIONS_NLP.md.
    """
    # Résolution phonologique d'abord
    phonological = resolve_nasal_tilde(word)

    # Puis lookup dans la table
    if word in table:
        return table[word], True
    if phonological in table:
        return table[phonological], True
    if phonological != word:
        # La résolution phonologique a changé quelque chose
        return phonological, True

    return word, False
```

### 4.4 Annotation des cas ambigus dans CONVENTIONS_NLP.md

Les abréviations pour lesquelles plusieurs expansions sont possibles doivent être documentées et tranchées explicitement. Ce document est un livrable du Jour 2 — il garantit la traçabilité scientifique.

```markdown
# CONVENTIONS_NLP.md — Décisions de normalisation

## Version : 1.0 | Date : AAAA-MM-JJ | Split SHA-256 : [hash]

### Principes généraux
- La normalisation vise la forme du français moderne orthographié standard.
- Les formes médiévales spécifiques conservant une valeur morphosyntaxique
  (ex. distinction cas sujet/cas régime) sont signalées mais normalisées.
- Toute décision non couverte par les règles est documentée ici.

### Cas ambigus résolus

| Forme brute | Expansion choisie | Expansion alternative | Justification |
|---|---|---|---|
| `p~` | `par` | `per`, `pro`, `pré` | Contexte : devant substantif → `par` dominant |
| `no~` | `nom` | `non`, `notre` | DMF : `no~` devant consonne = `nom` dans les chartes |
| `q~` | `que` | `qui` (devant voyelle) | Convention t9n |
| `s^r` | `seigneur` | `sir` (anglicisme) | Corpus francophone → `seigneur` |

### Formes non résolues (à traiter par CamemBERT MLM, Étape 3)
- Lettres suprascrites rares non couvertes par la table
- Abréviations de noms propres géographiques inconnus
- Hapax abréviés sans contexte suffisant
```

---

## 5. Étape 3 — Correction guidée par la confiance HTR et CamemBERT MLM

### 5.1 Le problème : positions de faible confiance avec candidats

Le Chapitre 2 a introduit le champ `candidates` du data contract HTR : pour les positions où le modèle HTR a hésité entre deux caractères, ce champ liste les alternatives et leurs scores. Exemple :

```json
{
  "transcription": "au duc de norm~die",
  "candidates": [
    {"position": 13, "alternatives": ["~", "a"], "scores": [0.60, 0.40]}
  ]
}
```

Ici, le modèle HTR a transcrit `~` avec 60 % de confiance, mais `a` était l'alternative à 40 %. Après la résolution phonologique, `norm~die` devient `normandie`. Mais si le modèle avait transcrit `a` à la place de `~`, on obtiendrait `normadie` — une transcription fausse qui ne se trouve pas dans le DMF.

Pour les positions de faible confiance, on peut utiliser CamemBERT en mode MLM pour arbitrer : quel est le caractère le plus probable étant donné le contexte complet de la ligne ?

### 5.2 Algorithme d'arbitrage MLM

L'arbitrage MLM fonctionne en trois étapes. D'abord, on identifie les positions candidates (confidence < seuil). Ensuite, on construit deux versions de la ligne — une avec chaque alternative — et on demande à CamemBERT de scorer les deux versions via pseudo-log-likelihood. Enfin, on choisit la version qui a le score le plus élevé.

```python
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

CAMEMBERT_MODEL = "almanach/camembert-base"
CONFIDENCE_THRESHOLD = 0.70   # seuil d'arbitrage (voir Chapitre 2, section 4.2)

tokenizer_mlm = AutoTokenizer.from_pretrained(CAMEMBERT_MODEL)
model_mlm     = AutoModelForMaskedLM.from_pretrained(CAMEMBERT_MODEL)
model_mlm.eval()

def score_text_mlm(text: str) -> float:
    """
    Calcule la pseudo-log-vraisemblance d'un texte sous CamemBERT.

    La pseudo-log-vraisemblance (PLL) est calculée en masquant chaque token
    un par un et en sommant les log-probabilités assignées au token original.
    C'est une approximation tractable de la probabilité du texte entier.

    PLL(x) = Σ_i log P(x_i | x_{-i} ; θ)

    Référence : Salazar et al. (2020), "Masked Language Model Scoring"
    """
    inputs = tokenizer_mlm(text, return_tensors="pt",
                            max_length=256, truncation=True)
    input_ids = inputs["input_ids"][0]
    log_prob_sum = 0.0

    with torch.no_grad():
        for i in range(1, len(input_ids) - 1):   # exclure [CLS] et [SEP]
            masked = input_ids.clone()
            masked[i] = tokenizer_mlm.mask_token_id
            logits = model_mlm(
                masked.unsqueeze(0),
                attention_mask=inputs["attention_mask"]
            ).logits
            log_prob = torch.log_softmax(logits[0, i], dim=-1)
            log_prob_sum += log_prob[input_ids[i]].item()

    return log_prob_sum

def arbitrate_candidate(line: dict,
                         candidate: dict,
                         threshold: float = CONFIDENCE_THRESHOLD) -> str:
    """
    Arbitre entre deux caractères candidats en utilisant CamemBERT MLM.

    Paramètres
    ----------
    line      : dict  ligne du data contract (avec "transcription" et "candidates")
    candidate : dict  un élément de line["candidates"]
    threshold : float seuil de confiance en dessous duquel on arbitre

    Retourne
    --------
    str  transcription corrigée (avec le caractère retenu)
    """
    conf_htр = candidate["scores"][0]
    if conf_htр >= threshold:
        return line["transcription"]   # confiance suffisante, pas d'arbitrage

    trans   = line["transcription"]
    pos     = candidate["position"]
    alt_a   = candidate["alternatives"][0]   # transcription HTR
    alt_b   = candidate["alternatives"][1]   # alternative

    # Construire les deux versions de la ligne
    version_a = trans[:pos] + alt_a + trans[pos + len(alt_a):]
    version_b = trans[:pos] + alt_b + trans[pos + len(alt_b):]

    # Résoudre phonologiquement les deux versions
    version_a_norm = resolve_nasal_tilde(version_a)
    version_b_norm = resolve_nasal_tilde(version_b)

    # Scorer avec CamemBERT
    score_a = score_text_mlm(version_a_norm)
    score_b = score_text_mlm(version_b_norm)

    chosen = version_a_norm if score_a >= score_b else version_b_norm
    return chosen
```

### 5.3 Ablation : trois stratégies comparées

Le TP vous demande de comparer trois stratégies d'arbitrage. Cette ablation est directement connectée aux décisions architecturales du Chapitre 3.

**Stratégie A — Confiance seule :** on fait confiance au HTR pour les positions dont la confidence est au-dessus du seuil, et on applique la règle phonologique sans vérification pour les positions en dessous.

**Stratégie B — MLM seul :** on applique l'arbitrage MLM à toutes les positions candidates, sans tenir compte des scores de confiance HTR.

**Stratégie C — Combinée (recommandée) :** on utilise la confiance HTR comme premier filtre (évite de faire tourner MLM sur des positions déjà claires), et MLM seulement pour les positions sous le seuil.

Le CER mesuré après chaque stratégie sur vos 200 lignes de référence est la métrique d'ablation. La stratégie combinée devrait dominer les deux autres, mais ce n'est pas garanti : si votre seuil est mal calibré, B peut être meilleur que C.

---

## 6. Étape 4 — Fine-tuning mT5 LoRA sur les paires normalisées

### 6.1 Construction des paires d'entraînement

Après les trois premières étapes, vous disposez de paires *(transcription brute, forme normalisée)* produites par le pipeline de règles. Ces paires constituent le corpus d'entraînement du modèle neural. La normalisation du Chapitre 3 a expliqué la mécanique LoRA ; ce qui est spécifique ici est la construction des données.

```python
from datasets import Dataset

def build_normalization_pairs(df,
                               apply_pipeline_fn,
                               split: str = "train") -> Dataset:
    """
    Construit le dataset de paires (brut → normalisé) pour le fine-tuning.

    Paramètres
    ----------
    df               : DataFrame  corpus aplati avec colonnes split, transcription,
                                  confidence, needs_review
    apply_pipeline_fn: callable   fonction de normalisation complète (règles + abbrev)
    split            : str        "train", "val" ou "test"

    Retourne
    --------
    HuggingFace Dataset avec colonnes :
        source        : str    transcription brute (entrée du modèle)
        target        : str    forme normalisée (cible)
        sample_weight : float  poids = confidence globale de la ligne
        document_type : str    type de document (pour analyse par genre)
    """
    subset = df[df["split"] == split].copy()
    records = []
    for _, row in subset.iterrows():
        normalized = apply_pipeline_fn(row["transcription"])
        if normalized == row["transcription"]:
            continue   # Skip les lignes que le pipeline n'a pas changées
        records.append({
            "source":        row["transcription"],
            "target":        normalized,
            "sample_weight": float(row["confidence"]),
            "document_type": row["document_type"],
        })
    return Dataset.from_list(records)

# Pipeline complet : règles + abréviations
def full_pipeline(text: str) -> str:
    words = text.split()
    normalized = []
    for word in words:
        # Étape 1 : Unicode
        w = normalize_unicode(word)
        # Étape 2 : u/v, i/j
        w = normalize_uv(w)
        w = normalize_ij(w)
        # Étape 3 : règles graphémiques
        w = apply_grapheme_rules(w)
        # Étape 4 : abréviations (résolution phonologique + table)
        w, _ = resolve_abbreviations(w)
        # Étape 5 : DMF lookup (si cache disponible)
        lemme = get_dmf_lemme(word.lower(), dmf_cache)
        if lemme and not word[0].isupper():
            w = lemme
        normalized.append(w)
    return " ".join(normalized)

train_dataset = build_normalization_pairs(df, full_pipeline, split="train")
val_dataset   = build_normalization_pairs(df, full_pipeline, split="val")
print(f"Paires entraînement : {len(train_dataset)}")
print(f"Paires validation   : {len(val_dataset)}")
```

### 6.2 Métriques d'évaluation

Le TP requiert trois métriques d'évaluation. Elles mesurent des choses différentes et sont complémentaires.

**Token accuracy** mesure le pourcentage de mots correctement normalisés, sans tenir compte des erreurs de voisinage :

$$\text{Token Acc} = \frac{\text{mots correctement normalisés}}{\text{mots totaux}}$$

**CER normalisé** (introduit en section 2.4) mesure les erreurs au niveau caractère. C'est la métrique la plus fine et la plus standard pour la normalisation de textes historiques.

**BLEU** (*Bilingual Evaluation Understudy*) mesure le chevauchement de n-grammes entre la sortie du modèle et la référence. Pour la normalisation, le BLEU-4 (n-grammes jusqu'à 4) est la convention :

$$\text{BLEU-4} = \text{BP} \times \exp\!\left(\sum_{n=1}^{4} \frac{1}{4} \log p_n\right)$$

où $p_n$ est la précision de n-grammes modifiée et BP est la pénalité de brièveté.

```python
from evaluate import load as load_metric

cer_metric  = load_metric("cer")
bleu_metric = load_metric("sacrebleu")

def evaluate_normalization(model, tokenizer, dataset,
                            prefix: str = "normalise moyen français: ") -> dict:
    """Évalue le modèle de normalisation sur un dataset."""
    hypotheses, references = [], []

    for example in dataset:
        input_text = prefix + example["source"]
        inputs = tokenizer(input_text, return_tensors="pt",
                           max_length=128, truncation=True)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=128)
        hyp = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        hypotheses.append(hyp)
        references.append(example["target"])

    # Token accuracy
    token_matches = sum(
        sum(h == r for h, r in zip(hyp.split(), ref.split()))
        for hyp, ref in zip(hypotheses, references)
    )
    token_total = sum(len(ref.split()) for ref in references)

    return {
        "cer":            cer_metric.compute(predictions=hypotheses,
                                              references=references),
        "bleu":           bleu_metric.compute(predictions=hypotheses,
                                               references=[[r] for r in references
                                               ])["score"],
        "token_accuracy": token_matches / max(token_total, 1),
    }
```

### 6.3 Tableau d'ablation complet

Le tableau d'ablation final réunit toutes les configurations du Jour 2. Sa structure est directement liée aux décisions architecturales du Chapitre 3 :

| Configuration | CER ↓ | BLEU ↑ | Token Acc ↑ | Params entr. | Note |
|---|---|---|---|---|---|
| HTR brut (baseline) | — | — | — | 0 | Point de départ |
| Règles seules | — | — | — | 0 | Étape 1 |
| Règles + abbrev. | — | — | — | 0 | Étapes 1+2 |
| Règles + MLM conf. | — | — | — | 0 | Étapes 1+2+3 |
| mT5 LoRA r=8, Q+V | — | — | — | ~300K | Étape 4 |
| mT5 LoRA r=16, Q+V | — | — | — | ~600K | Ablation r |
| Pipeline hybride | — | — | — | ~300K | Règles → mT5 |

La colonne "Pipeline hybride" est souvent la meilleure configuration : les règles traitent les cas déterministes (et réduisent la fragmentation BPE, rappel Chapitre 1), le modèle neural traite les cas ambigus restants. L'hypothèse est que la combinaison est meilleure que chaque approche isolément — votre tableau le vérifiera.

---

## 7. Traçabilité : CONVENTIONS_NLP.md et le journal d'expériences

### 7.1 Le rôle de CONVENTIONS_NLP.md

`CONVENTIONS_NLP.md` n'est pas un fichier de documentation optionnel — c'est un livrable de même importance que le checkpoint du modèle. Dans un projet de humanités numériques, les décisions de normalisation ont des conséquences scientifiques : si vous normalisez *li roys* en *le roi*, vous effacez une distinction morphosyntaxique (cas sujet vs cas régime) qui peut être signifiante pour un historien ou un linguiste qui utilisera vos données.

La règle est simple : toute décision que vous prenez et qui n'est pas entièrement déterminée par le DMF doit apparaître dans `CONVENTIONS_NLP.md` avec sa justification.

### 7.2 Versionnement du module de règles

Le module de règles doit être versionné. Chaque modification d'une règle change le corpus d'entraînement, change les métriques, et invalide les comparaisons antérieures. La convention recommandée est un numéro de version sémantique : `v1.0.0` pour la version initiale, `v1.0.1` pour un ajout de règle sans modification d'existant, `v1.1.0` pour une modification d'une règle existante.

### 7.3 Mise à jour du journal d'expériences

Le journal `experiments/journal.jsonl` (introduit au Chapitre 3) est mis à jour après chaque étape mesurée :

```python
# Après l'Étape 1 (règles seules)
log_experiment(
    config={
        "step":             1,
        "method":           "rules",
        "rules_version":    "v1.0.0",
        "grapheme_rules":   len(GRAPHEME_RULES),
        "abbrev_table_size": len(ABBREV_TABLE),
    },
    metrics={
        "cer_avant":  cer_avant,
        "cer_apres":  cer_apres,
        "reduction":  (cer_avant - cer_apres) / cer_avant,
    },
    split_hash=SPLIT_HASH,
)
```

---

## Bibliographie de référence

### Linguistique du moyen français

Marchello-Nizia, C. (1997). *La langue française aux XIVe et XVe siècles*. Nathan. — La référence grammaticale standard pour le moyen français.

Greimas, A. J., & Keane, T. M. (1992). *Dictionnaire du moyen français : la Renaissance*. Larousse. — Complément lexicographique.

Picoche, J., & Marchello-Nizia, C. (1989). *Histoire de la langue française*. Nathan. — Introduction à la phonologie et à la morphologie historique du français.

Pope, M. K. (1934). *From Latin to Modern French with especial consideration of Anglo-Norman*. Manchester University Press. — Ouvrage de référence sur les évolutions phonologiques.

### DMF et LGeRM

Souvay, G., & Pierrel, J.-M. (2009). **LGeRM : lemmatisation des mots en moyen français**. *Traitement Automatique des Langues*, 50(2), 149–172. — Article fondateur décrivant l'architecture de LGeRM.

Martin, R. (dir.) (2012, v. 2020). **Dictionnaire du Moyen Français (DMF)**. ATILF — CNRS & Université de Lorraine. [www.atilf.fr/dmf](http://www.atilf.fr/dmf). — La ressource lexicographique de référence.

### Normalisation de textes historiques

Bollmann, M. (2019). **A Large-Scale Comparison of Historical Text Normalization Systems**. *NAACL 2019*. [arXiv:1904.02036](https://arxiv.org/abs/1904.02036) — Comparaison systématique des approches règles vs neural sur plusieurs langues et périodes.

Domingo, M., & Casacuberta, F. (2018). **Spelling Normalization of Historical Documents by Using a Machine Translation System**. *IWSLT 2018*.

Bollmann, M., & Søgaard, A. (2016). **Improving Historical Spelling Normalization with Bi-directional LSTMs and Multi-Task Learning**. *COLING 2016*. [arXiv:1610.07004](https://arxiv.org/abs/1610.07004)

Clérice, T. (2023). **Pie Extended** (lemmatisation et normalisation pour le français médiéval). [Zenodo](https://doi.org/10.5281/zenodo.3883589)

### Normalisation médiévale francophone

Camps, J.-B., Vinsonneau, A., & Clérice, T. (2021). **Corpus and Models for Lemmatisation and POS-tagging of Old French**. *Journal of Data Mining & Digital Humanities*.

Gabay, S., & Diwersy, S. (2020). **Normalisation de l'orthographe en français classique : quelques pistes**. *Humanités Numériques*, 1.

### Évaluation MLM comme modèle de langue

Salazar, J., Liang, D., Nguyen, T. Q., & Kirchhoff, K. (2020). **Masked Language Model Scoring**. *ACL 2020*. [arXiv:1910.14659](https://arxiv.org/abs/1910.14659) — Fondement théorique de la pseudo-log-vraisemblance utilisée à l'Étape 3.

### Outils

Ott, M., Edunov, S., Baevski, A., Fan, A., Gross, S., Ng, N., Grangier, D., & Auli, M. (2019). **fairseq: A Fast, Extensible Toolkit for Sequence Modeling**. *NAACL 2019 (démo)*. [arXiv:1904.01038](https://arxiv.org/abs/1904.01038)

*t9n — Outil de normalisation médiévale.* [GitHub](https://github.com/PonteIneptique/tei-to-training) — Outil de référence pour les choix de normalisation en humanités numériques.

---

*Support de cours rédigé pour le Master Data/IA · Module NLP · MD5 Volet 2 · 2026. Ce document accompagne le TP autonome du Jour 2 (13h30–17h00). Les livrables attendus en fin de séance sont : le module de règles documenté (`CONVENTIONS_NLP.md`), la table d'abréviations résolues, le checkpoint du modèle de normalisation, le tableau d'ablation, et le journal d'expériences mis à jour.*
