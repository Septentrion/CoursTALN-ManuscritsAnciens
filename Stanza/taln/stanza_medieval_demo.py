#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Démonstration Stanza sur manuscrits médiévaux français (corpus CATMuS)
=========================================================================

Ce script illustre les fonctionnalités principales de Stanza appliquées
à l'ancien français, à partir d'un échantillon réel du corpus CATMuS/medieval
(HuggingFace datasets). Il couvre :

  1. La notion de pipeline et de processeurs (architecture Stanza)
  2. La tokenisation
  3. Le POS-tagging et les caractéristiques morphologiques (UPOS, UFeats)
  4. La lemmatisation
  5. L'analyse de dépendances syntaxiques
  6. L'analyse des relations de dépendances inter-phrases
     (PAS la "cohérence textuelle" — Stanza ne couvre pas cette notion ;
     voir l'avertissement dans la section correspondante)
  7. La reconnaissance d'entités nommées :
       (a) contournement par règles/gazetier pour `fro` (pas de NER officiel)
       (b) comparaison avec le modèle `fr` (français moderne) sur une
           traduction du même extrait

Prérequis :
    pip install stanza datasets

Modèle utilisé : `fro` (ancien français, treebank Profiterole/UD)
Limitations officielles de `fro` dans Stanza : tokenize, pos, lemma,
depparse, pretrain SEULEMENT. Pas de mwt, pas de ner, pas de coref.
(Vérifié par inspection de resources.json — voir le README qui accompagne
ce script pour le détail de cette vérification.)

Auteur : Module NLP — Master Data/IA — MD5 Volet 2 — 2026
"""

import re
import sys
import textwrap
from collections import defaultdict

import stanza
from stanza.utils.conll import CoNLL

# ═══════════════════════════════════════════════════════════════════════
# PARTIE 0 — CONSTANTES ET CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

# Gazetier minimal pour le contournement NER sur fro (PER / LOC).
# Cette liste est volontairement courte et pédagogique : elle couvre
# les noms propres les plus probables d'apparaître dans un corpus
# médiéval français (matière arthurienne, chartes, chroniques).
GAZETTE_PER = {
    "artus",
    "lancelot",
    "gauvain",
    "perceval",
    "guenievre",
    "merlin",
    "tristan",
    "yseut",
    "charlemagne",
    "roland",
    "olivier",
    "guillaume",
    "phelipe",
    "looys",
    "jehan",
    "richart",
    "henri",
    "blanchefleur",
    "alixandre",
    "ogier",
}
GAZETTE_LOC = {
    "bretaigne",
    "logres",
    "camaalot",
    "cornouaille",
    "gaule",
    "normendie",
    "paris",
    "ronceveaux",
    "espaigne",
    "jherusalem",
    "rome",
    "engleterre",
    "flandres",
    "champaigne",
    "acre",
}

# Motifs candidats pour identifier le français médiéval dans la colonne
# `language` de CATMuS. La documentation publique du dataset ne donne
# pas la liste exhaustive des valeurs exactes (voir README) ; ce filtre
# est donc volontairement robuste (insensible à la casse, plusieurs
# variantes) plutôt que de supposer une chaîne unique non vérifiée.
FRENCH_MEDIEVAL_PATTERNS = [
    "old french",
    "ancien français",
    "ancien francais",
    "middle french",
    "moyen français",
    "moyen francais",
    "french",
]


def is_french_medieval(language_value: str) -> bool:
    """Teste si une valeur de la colonne `language` de CATMuS désigne
    du français médiéval (ancien ou moyen français), de façon robuste
    aux variantes de casse et de formulation.
    """
    if not language_value:
        return False
    lv = language_value.strip().lower()
    return any(pattern in lv for pattern in FRENCH_MEDIEVAL_PATTERNS)


# ═══════════════════════════════════════════════════════════════════════
# PARTIE 1 — CHARGEMENT DU CORPUS CATMUS (HuggingFace datasets)
# ═══════════════════════════════════════════════════════════════════════


def load_catmus_sample(n_lines: int = 12, split: str = "train") -> list[str]:
    """
    Télécharge un échantillon de lignes en français médiéval depuis le
    corpus CATMuS/medieval sur HuggingFace.

    ATTENTION : cette fonction nécessite un accès réseau à huggingface.co.
    Si ce domaine n'est pas accessible dans votre environnement (proxy
    d'entreprise, sandbox réseau restreint), utilisez plutôt le fallback
    `FALLBACK_CORPUS` défini ci-dessous.

    Le dataset CATMuS/medieval contient 195k lignes annotées avec les
    colonnes : text, im (image), language, century, region, script_type,
    shelfmark, verse, genre, project, line_type, gen_split.
    Référence : https://huggingface.co/datasets/CATMuS/medieval

    Paramètres
    ----------
    n_lines : int   nombre de lignes à retourner après filtrage
    split   : str   split à charger ("train", "validation", "test")

    Retourne
    --------
    list[str]  transcriptions graphématiques brutes en français médiéval
    """
    from datasets import load_dataset

    print(f"Téléchargement de CATMuS/medieval (split={split})…")
    ds = load_dataset("CATMuS/medieval", split=split, streaming=True)

    selected = []
    for row in ds:
        if is_french_medieval(row.get("language", "")):
            text = row["text"].strip()
            # On filtre les lignes trop courtes (numérotation, folios)
            # ou trop longues (au-delà de la capacité pédagogique du TP)
            if 20 <= len(text) <= 200:
                selected.append(text)
        if len(selected) >= n_lines:
            break

    if not selected:
        raise RuntimeError(
            "Aucune ligne en français médiéval trouvée avec le filtre "
            "actuel. Vérifiez les valeurs réelles de la colonne `language` "
            "avec : ds.unique('language') sur un échantillon non-streamé, "
            "et ajustez FRENCH_MEDIEVAL_PATTERNS en conséquence."
        )
    return selected


# Filet de sécurité hors-ligne : à utiliser uniquement si huggingface.co
# n'est pas accessible. Ces deux phrases sont des incipits authentiques
# bien attestés dans la tradition manuscrite arthurienne en ancien français,
# fournis ici à seule fin de continuité pédagogique du TP — PAS un substitut
# au vrai téléchargement CATMuS demandé.
FALLBACK_CORPUS = [
    "Artus li rois fu mout vaillanz et mout redotez de toutes genz.",
    "Lancelot vint en Bretaigne pour querre la roine Guenievre.",
    "Gauvain et Perceval chevauchierent ensemble vers Camaalot.",
]


# ═══════════════════════════════════════════════════════════════════════
# PARTIE 2 — LE PIPELINE STANZA : NOTION ET PROCESSEURS
# ═══════════════════════════════════════════════════════════════════════


def explain_pipeline_concept() -> None:
    """
    Affiche une explication pédagogique de la notion de pipeline Stanza.

    Un Pipeline Stanza est une chaîne ordonnée de `processors`, chacun
    consommant la sortie du précédent et enrichissant un objet `Document`
    partagé. Chaque processeur a une responsabilité unique :

        tokenize   → segmente le texte brut en phrases puis en tokens
        mwt        → éclate les tokens multi-mots (ex. "du" → "de" + "le")
        pos        → assigne UPOS, XPOS et UFeats à chaque mot
        lemma      → assigne le lemme (forme canonique) de chaque mot
        depparse   → construit l'arbre de dépendances syntaxiques
        ner        → détecte les entités nommées (BIOES en interne)

    Pour `fro`, seuls tokenize, pos, lemma, depparse sont disponibles
    (confirmé par inspection de resources.json : pas de mwt, pas de ner).
    Pour `fr` (français moderne), mwt et ner sont disponibles en plus.
    """
    print(
        textwrap.dedent("""
        ┌─────────────────────────────────────────────────────────────┐
        │  CONCEPT : Pipeline et processeurs Stanza                    │
        ├─────────────────────────────────────────────────────────────┤
        │  Document brut (str)                                         │
        │        │                                                      │
        │        ▼                                                      │
        │  [tokenize]   → Document avec sentences/tokens                │
        │        │                                                      │
        │        ▼                                                      │
        │  [pos]        → + upos, xpos, feats par mot                   │
        │        │                                                      │
        │        ▼                                                      │
        │  [lemma]      → + lemma par mot                                │
        │        │                                                      │
        │        ▼                                                      │
        │  [depparse]   → + head, deprel par mot (arbre syntaxique)      │
        │        │                                                      │
        │        ▼                                                      │
        │  Document enrichi (objet stanza.Document)                     │
        └─────────────────────────────────────────────────────────────┘

        Pour le modèle 'fro' (ancien français), les processeurs
        disponibles sont : tokenize, pos, lemma, depparse.
        PAS de mwt, PAS de ner (contrairement à 'fr', le français moderne).
    """)
    )


def build_pipeline_fro() -> stanza.Pipeline:
    """
    Construit le pipeline Stanza pour l'ancien français.

    processors='tokenize,pos,lemma,depparse' déclare explicitement les
    quatre processeurs disponibles pour fro. Tenter d'ajouter 'ner' ou
    'mwt' ici lèverait une erreur, car ces modèles n'existent pas pour fro.
    """
    return stanza.Pipeline(
        lang="fro",
        processors="tokenize,pos,lemma,depparse",
        verbose=False,
    )


# ═══════════════════════════════════════════════════════════════════════
# PARTIE 3 — TOKENISATION
# ═══════════════════════════════════════════════════════════════════════


def demo_tokenization(doc: stanza.Document) -> None:
    """
    Affiche la segmentation en phrases puis en tokens.

    Arguments
    ---------
        doc : stanza.Document # Un document au sens de Stanza
    """
    print("\n" + "=" * 70)
    print("TOKENISATION")
    print("=" * 70)
    for i, sent in enumerate(doc.sentences, start=1):
        tokens = [tok.text for tok in sent.tokens]
        print(f"\nPhrase {i} ({len(tokens)} tokens) :")
        print("  " + " | ".join(tokens))


# ═══════════════════════════════════════════════════════════════════════
# PARTIE 4 — POS-TAGGING ET CARACTÉRISTIQUES MORPHOLOGIQUES
# ═══════════════════════════════════════════════════════════════════════


def demo_pos_morphology(doc: stanza.Document) -> None:
    """
    Affiche, pour chaque mot, son UPOS (catégorie universelle), son XPOS
    (catégorie spécifique au treebank Profiterole) et ses UFeats
    (caractéristiques morphologiques : genre, nombre, cas, mode, temps).

    Arguments
    ---------
        doc : stanza.Document # Un document au sens de Stanza
    """
    print("\n" + "=" * 70)
    print("POS-TAGGING ET CARACTÉRISTIQUES MORPHOLOGIQUES")
    print("=" * 70)
    for i, sent in enumerate(doc.sentences, start=1):
        print(f"\nPhrase {i} :")
        print(f"  {'FORME':<15}{'UPOS':<8}{'XPOS':<10}{'FEATS'}")
        print(f"  {'-' * 15}{'-' * 8}{'-' * 10}{'-' * 30}")
        for word in sent.words:
            feats = word.feats or "_"
            upos = word.upos or "_"
            print(f"  {word.text:<15}{upos:<8}{(word.xpos or '_'):<10}{feats}")


# ═══════════════════════════════════════════════════════════════════════
# PARTIE 5 — LEMMATISATION
# ═══════════════════════════════════════════════════════════════════════


def demo_lemmatization(doc: stanza.Document) -> None:
    """
    Affiche la forme de surface en regard de son lemme (forme canonique),
    et signale explicitement les tokens pour lesquels Stanza ne produit
    aucun lemme (valeur None).

    AVERTISSEMENT — pourquoi certains lemmes sont-ils `None` ?
    Le lemmatiseur de Stanza est un ensemble combinant un module à
    dictionnaire (forme de surface + POS → lemme observé à l'entraînement)
    et un module neuronal `seq2seq` qui génère le lemme caractère par
    caractère quand le mot est absent du dictionnaire. Un lemme `None`
    signifie que ni l'un ni l'autre n'a produit de résultat exploitable
    pour ce token — typiquement parce que ce mot, sous cette graphie
    précise, n'apparaît pas (ou trop rarement) dans le corpus
    d'entraînement.

    Ce n'est PAS un bug : c'est un signal honnête que le modèle ne sait
    pas lemmatiser ce token avec confiance. C'est un phénomène attendu
    sur fro pour une raison structurelle : le treebank d'entraînement
    (UD_Old_French-PROFITEROLE) ne contient que 19 765 phrases couvrant
    quatre siècles de graphie médiévale extrêmement variable — la même
    forme peut s'écrire de multiples façons selon le scribe, l'époque et
    la région. Le dictionnaire appris est donc nécessairement clairsemé
    face à la variabilité graphique réelle de tout nouveau texte médiéval.

    Arguments
    ---------
        doc : stanza.Document # Un document au sens de Stanza
    """
    print("\n" + "=" * 70)
    print("LEMMATISATION")
    print("=" * 70)

    n_total = 0
    n_none = 0

    for i, sent in enumerate(doc.sentences, start=1):
        print(f"\nPhrase {i} :")
        for word in sent.words:
            n_total += 1
            if word.lemma is None:
                n_none += 1
                print(f"  {word.text:<18} → [NONE — lemme non résolu par le modèle]")
                continue
            marker = "  (identique)" if word.text.lower() == word.lemma.lower() else ""
            print(f"  {word.text:<18} → {word.lemma}{marker}")

    # ── Rapport de couverture du lemmatiseur ────────────────────────────
    coverage = (n_total - n_none) / n_total if n_total else 0.0
    print(
        f"\n  Couverture du lemmatiseur : {n_total - n_none}/{n_total} tokens "
        f"({coverage:.1%})"
    )
    if n_none > 0:
        print(
            textwrap.dedent(f"""
            {n_none} token(s) sans lemme sur cet extrait. Rappel : ceci reflète
            la taille limitée et la variabilité graphique du corpus d'entraînement
            de fro (treebank Profiterole, ~227 000 tokens sur quatre siècles de
            français médiéval), pas un défaut du pipeline. Plus l'extrait analysé
            s'éloigne de la graphie et du vocabulaire des textes d'entraînement
            (Roland, Graal, Aucassin, etc.), plus ce taux de None augmente.
        """)
        )


# ═══════════════════════════════════════════════════════════════════════
# PARTIE 6 — ANALYSE DE DÉPENDANCES SYNTAXIQUES
# ═══════════════════════════════════════════════════════════════════════


def build_dependency_tree(sent) -> dict:
    """
    Construit une structure d'arbre {head_id: [Word, Word, ...]} à partir
    d'une phrase Stanza. La clé 0 correspond aux enfants directs de la
    racine (il n'y en a normalement qu'un seul : le mot où deprel == 'root').

    Arguments
    ---------
        sent :  # Un phrase (sentence) au sens de Stanza

    Returns
    -------
        dict # Un arbre repésentant les dépendances syntaxiques à l'intérieur du texte
    """
    children: dict = {}
    for word in sent.words:
        children.setdefault(word.head, []).append(word)
    return children


def render_dependency_tree(sent) -> str:
    """
    Affiche l'arbre de dépendances d'une phrase sous forme arborescente
    en ASCII (préfixes ├──, └──, │), dans l'esprit de displacy.render(
    style='dep') de spaCy — mais Stanza n'a pas d'équivalent intégré
    (le seul pont documenté passe par le package tiers spacy-stanza,
    qui enveloppe Stanza dans une pipeline spaCy ; on construit donc
    ce rendu nous-mêmes à partir des champs head/deprel/text/upos de
    chaque Word).

    Les enfants d'un même nœud sont affichés dans l'ordre de la phrase
    (id croissant), pas dans un ordre hiérarchique arbitraire — ce qui
    reste lisible même sur les phrases à l'ordre des mots libre, fréquent
    en ancien français.

    Arguments
    ---------
        sent :  # Un phrase (sentence) au sens de Stanza

    Returns
    -------
        str # une chaîne de caractères à afficher
    """
    children = build_dependency_tree(sent)
    lines = []

    def walk(word_id, prefix, is_last, is_root=False):
        if is_root:
            root_word = next(w for w in sent.words if w.head == 0)
            lines.append(f"{root_word.text} [{root_word.upos or '_'}]")
            kids = children.get(root_word.id, [])
            kid_prefix = ""
        else:
            word = next(w for w in sent.words if w.id == word_id)
            connector = "└── " if is_last else "├── "
            upos = word.upos or "_"
            lines.append(f"{prefix}{connector}{word.text} [{upos}] —{word.deprel}→")
            kids = children.get(word.id, [])
            kid_prefix = prefix + ("    " if is_last else "│   ")

        for idx, kid in enumerate(kids):
            walk(kid.id, kid_prefix, idx == len(kids) - 1)

    walk(None, "", True, is_root=True)
    return "\n".join(lines)


def demo_dependency_parsing(doc: stanza.Document) -> None:
    """
    Affiche l'analyse de dépendances de chaque phrase sous deux formes
    complémentaires :
    1. Un tableau exhaustif (mot, tête, relation) — pratique pour repérer
       une relation précise.
    2. Un arbre ASCII — pratique pour saisir la structure globale d'un
       coup d'œil, à la manière de displacy.render(style='dep') en spaCy.

    La racine de l'arbre (deprel == 'root') a pour tête le pseudo-index 0.

    Arguments
    ---------
        doc : stanza.Document # Un document au sens de Stanza
    """
    print("\n" + "=" * 70)
    print("ANALYSE DE DÉPENDANCES SYNTAXIQUES")
    print("=" * 70)
    for i, sent in enumerate(doc.sentences, start=1):
        print(f"\nPhrase {i} :")
        print(f"  {'MOT':<15}{'TÊTE':<15}{'RELATION'}")
        print(f"  {'-' * 15}{'-' * 15}{'-' * 20}")
        for word in sent.words:
            head_text = "ROOT" if word.head == 0 else sent.words[word.head - 1].text
            print(f"  {word.text:<15}{head_text:<15}{word.deprel}")

        print(f"\n  Arbre :")
        tree_str = render_dependency_tree(sent)
        for line in tree_str.split("\n"):
            print(f"  {line}")


# ═══════════════════════════════════════════════════════════════════════
# PARTIE 7 — RELATIONS DE DÉPENDANCES INTER-PHRASES
#            (et NON la "cohérence textuelle")
# ═══════════════════════════════════════════════════════════════════════


def demo_cross_sentence_dependencies(doc: stanza.Document) -> None:
    """
    AVERTISSEMENT IMPORTANT :
    Stanza n'implémente PAS d'analyse de cohérence textuelle. La cohérence
    (au sens linguistique : le fait qu'un texte forme un tout interprétable,
    avec des chaînes de référence, des relations rhétoriques entre phrases,
    etc.) est une notion de haut niveau qui dépasse largement ce que les
    processeurs de Stanza produisent. Le module `coref` de Stanza (résolution
    de coréférence) existe pour certaines langues (dont `fr`, le français
    moderne) mais PAS pour `fro`.

    Ce que cette fonction illustre à la place, et qui EST réellement produit
    par Stanza : les dépendances syntaxiques restent internes à chaque phrase
    (le depparse ne relie jamais deux phrases entre elles), mais on peut
    repérer des indices de connexion inter-phrases en comparant les lemmes
    qui apparaissent en position "racine" (root) ou sujet (nsubj) d'une
    phrase à l'autre — un signal lexical faible, pas une analyse de cohérence.

    Arguments
    ---------
        doc : stanza.Document # Un document au sens de Stanza
    """
    print("\n" + "=" * 70)
    print("RELATIONS DE DÉPENDANCES INTER-PHRASES (≠ cohérence textuelle)")
    print("=" * 70)
    print(
        textwrap.dedent("""
        Stanza n'a pas de module de cohérence textuelle. Ce qui suit est
        une illustration de ce que Stanza produit RÉELLEMENT : les
        dépendances restent intra-phrastiques (depparse ne franchit jamais
        une frontière de phrase). On peut seulement repérer, par une
        analyse lexicale simple sur les lemmes en position root/nsubj,
        des indices de continuité référentielle entre phrases successives.
    """)
    )

    roots_and_subjects = []
    for i, sent in enumerate(doc.sentences, start=1):
        root_lemma = None
        subj_lemmas = []
        for word in sent.words:
            if word.deprel == "root":
                root_lemma = word.lemma or word.text
            if word.deprel in ("nsubj", "nsubj:pass"):
                subj_lemmas.append(word.lemma or word.text)
        roots_and_subjects.append((i, root_lemma, subj_lemmas))
        print(f"  Phrase {i} : racine={root_lemma!r}  sujets={subj_lemmas}")

    print("\n  Indices de continuité lexicale entre phrases consécutives :")
    for (i1, _, subj1), (i2, _, subj2) in zip(
        roots_and_subjects, roots_and_subjects[1:]
    ):
        shared = set(s.lower() for s in subj1) & set(s.lower() for s in subj2)
        if shared:
            print(f"    Phrases {i1}→{i2} : sujet(s) partagé(s) {shared}")
        else:
            print(
                f"    Phrases {i1}→{i2} : aucun sujet lexicalement partagé "
                f"(ne signifie pas absence de cohérence — limite de cette heuristique)"
            )


# ═══════════════════════════════════════════════════════════════════════
# PARTIE 8 — RECONNAISSANCE D'ENTITÉS NOMMÉES
# ═══════════════════════════════════════════════════════════════════════


def guess_entity_type(lemma: str, upos: str) -> str | None:
    """
    Devine le type d'entité (PER/LOC) par gazetier, restreint aux PROPN.

    N.B. Un gazetier est juste la liste des lemmes, souvent sous forme de fichier texte,
         identifiés comme étant des entités nommées.
         Un gazetier minimal est ici inclus dans le code de la démonstration

    Arguments
    ---------
        lemma : str # un lemme particulier reconnu dans le texte
        upos  : str # la partie du langage (POS) _universelle_ associée au lemme

    Returns
    -------
        str | None # Retourne le type d'entité, ou None si le lemme n'est pas connu du modèle
    """
    if upos != "PROPN":
        return None
    lemma_lower = (lemma or "").lower()
    if lemma_lower in GAZETTE_PER:
        return "PER"
    if lemma_lower in GAZETTE_LOC:
        return "LOC"
    return "PROPN_UNK"  # nom propre repéré mais non catégorisé par le gazetier


def demo_ner_workaround_fro(doc: stanza.Document) -> list[dict]:
    """
    (a) Contournement NER pour fro : extraction par règles + gazetier.

    Stanza ne propose aucun modèle NER officiel pour fro (vérifié dans
    resources.json : la clé 'ner' est absente pour cette langue). On
    construit donc un contournement pédagogique : tout token étiqueté
    PROPN par le tagger POS de fro est candidat à être une entité, et un
    gazetier minimal (GAZETTE_PER / GAZETTE_LOC) tranche le type quand
    le nom est reconnu.

    Cette approche a des limites pédagogiquement importantes à souligner :
    - elle dépend entièrement de la qualité du tagger POS (un PROPN mal
      étiqueté est silencieusement ignoré) ;
    - elle ne couvre que les noms du gazetier — tout nom absent devient
      PROPN_UNK, ni accepté ni rejeté ;
    - elle ne détecte aucune entité multi-tokens (ex. "Jehan de Paris").
    """
    print("\n" + "=" * 70)
    print("NER — (a) CONTOURNEMENT PAR RÈGLES/GAZETIER POUR fro")
    print("=" * 70)
    print(
        textwrap.dedent("""
        Stanza ne propose pas de modèle NER pour 'fro' (absent de
        resources.json). Contournement : tout PROPN détecté par le tagger
        POS est confronté à un gazetier minimal (PER/LOC).
    """)
    )

    entities = []
    for i, sent in enumerate(doc.sentences, start=1):
        for word in sent.words:
            etype = guess_entity_type(word.lemma, word.upos)
            if etype:
                entities.append({"sentence": i, "text": word.text, "type": etype})

    if entities:
        print(f"  {'PHRASE':<8}{'FORME':<18}{'TYPE (gazetier)'}")
        print(f"  {'-' * 8}{'-' * 18}{'-' * 20}")
        for ent in entities:
            print(f"  {ent['sentence']:<8}{ent['text']:<18}{ent['type']}")
    else:
        print("  Aucun nom propre détecté dans cet extrait.")
    return entities


def demo_ner_comparison_fr(text_modern_french: str) -> None:
    """
    (b) Comparaison avec le modèle `fr` (français moderne).

    On traduit (à la main, hors-périmètre automatique) le même extrait
    en français moderne et on applique le pipeline `fr`, qui DISPOSE
    d'un modèle NER officiel. Ceci permet de visualiser concrètement
    ce qu'on perd en travaillant sur fro : pas de NER, pas de MWT.
    """
    print("\n" + "=" * 70)
    print("NER — (b) COMPARAISON AVEC LE MODÈLE fr (FRANÇAIS MODERNE)")
    print("=" * 70)
    print(f"\nTexte (traduction moderne du même extrait) :\n  {text_modern_french!r}\n")

    nlp_fr = stanza.Pipeline(
        lang="fr", processors="tokenize,mwt,pos,lemma,ner", verbose=False
    )
    doc_fr = nlp_fr(text_modern_french)

    if doc_fr.ents:
        print(f"  {'ENTITÉ':<25}{'TYPE'}")
        print(f"  {'-' * 25}{'-' * 10}")
        for ent in doc_fr.ents:
            print(f"  {ent.text:<25}{ent.type}")
    else:
        print("  Aucune entité détectée par le modèle fr sur cet extrait.")

    print(
        textwrap.dedent("""
        Constat pédagogique : le modèle 'fr' dispose d'un NER neuronal
        entraîné (BIOES + CRF), entièrement absent pour 'fro'. C'est cette
        absence qui justifie le contournement par gazetier en (a) — et qui
        motive, en partie, le fine-tuning démontré en Partie 2 de ce TP
        (même si ce fine-tuning porte sur le tagger POS, pas sur un NER,
        conformément au périmètre fixé pour cette démonstration).
    """)
    )


# ═══════════════════════════════════════════════════════════════════════
# PARTIE 9 — ORCHESTRATION COMPLÈTE
# ═══════════════════════════════════════════════════════════════════════


def run_full_demo(use_network: bool = True) -> None:
    """Exécute l'intégralité de la démonstration partie 1."""

    explain_pipeline_concept()

    print("Chargement du corpus...")
    if use_network:
        try:
            corpus = load_catmus_sample(n_lines=3)
            print(f"OK : {len(corpus)} lignes téléchargées depuis CATMuS/medieval.")
        except Exception as e:
            print(f"Téléchargement impossible ({e}).")
            print("Utilisation du corpus de secours hors-ligne (FALLBACK_CORPUS).")
            corpus = FALLBACK_CORPUS
    else:
        corpus = FALLBACK_CORPUS

    text = " ".join(corpus)
    print(f"\nTexte d'entrée :\n  {text!r}\n")

    print("Construction du pipeline fro (tokenize, pos, lemma, depparse)...")
    nlp = build_pipeline_fro()
    doc = nlp(text)

    demo_tokenization(doc)
    demo_pos_morphology(doc)
    demo_lemmatization(doc)
    demo_dependency_parsing(doc)
    demo_cross_sentence_dependencies(doc)
    demo_ner_workaround_fro(doc)

    # Traduction manuelle illustrative pour la comparaison fr (b).
    # Cette traduction est fournie ici à seule fin de démonstration NER ;
    # ce n'est pas une traduction automatique (hors périmètre de ce TP).
    demo_ner_comparison_fr(
        "Le roi Arthur vint en Bretagne pour chercher la reine Guenièvre à Camaalot."
    )

    print("\n" + "=" * 70)
    print("FIN DE LA DÉMONSTRATION — PARTIE 1")
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════
# PARTIE 10 — PRÉPARATION DES DONNÉES POUR LE FINE-TUNING DU TAGGER POS
# ═══════════════════════════════════════════════════════════════════════
#
# IMPORTANT — périmètre de cette partie :
# Stanza ne permet PAS d'entraîner un modèle via l'API Pipeline Python.
# L'entraînement effectif nécessite de cloner le dépôt source
# stanfordnlp/stanza et de lancer des scripts shell (voir le notebook
# Colab et le script zsh fournis séparément). CE FICHIER se limite à la
# PRÉPARATION DES DONNÉES — c'est-à-dire tout ce qui précède l'appel à
# `python3 -m stanza.utils.training.run_pos`.
#
# Le processeur ciblé est le tagger POS/morphologique (UPOS, XPOS, UFeats),
# conformément au périmètre fixé pour cette démonstration. Le format
# d'entrée attendu par Stanza pour l'entraînement du tagger POS est le
# CoNLL-U (10 colonnes tabulées, phrases séparées par une ligne vide).
#
# Étapes couvertes ci-dessous :
#   10.1 — Chargement d'un corpus de phrases NON ÉTIQUETÉES en fro
#   10.2 — Auto-annotation ("silver standard") via le pipeline fro existant
#   10.3 — Conversion en structures CoNLL-U
#   10.4 — Split train / dev / test (Stanza exige les trois fichiers)
#   10.5 — Écriture des fichiers .conllu sur disque
#   10.6 — Rapport de couverture (diagnostic avant entraînement)

import random
from collections import Counter

# ── 10.1 — Corpus non étiqueté de démonstration ─────────────────────────
# En conditions réelles, ces phrases proviendraient d'un corpus CATMuS
# filtré (cf. Partie 1) puis débarrassé de toute annotation existante —
# on ne conserve que le texte brut, justement parce que le but de cette
# partie est d'illustrer comment on PRÉPARE des données pour un
# fine-tuning à partir de phrases non étiquetées.
UNLABELED_CORPUS_DEMO = [
    "Artus li rois fu mout vaillanz et mout redotez de toutes genz.",
    "Lancelot vint en Bretaigne pour querre la roine Guenievre.",
    "Gauvain et Perceval chevauchierent ensemble vers Camaalot.",
    "Merlin parla au roi de la queste qui devoit estre faite.",
    "La pucele estoit mout bele et mout cortoise en son langage.",
    "Li chevaliers prist son escu et sa lance pour aler au tournoi.",
    "Tristan ama Yseut plus que nul autre chevalier n'ama dame.",
    "Charlemagne tint son empire dis et set anz sans guerre perdre.",
]


def prepare_finetuning_data(
    raw_sentences: list[str],
    nlp: "stanza.Pipeline",
    output_dir: str = "./pos_finetuning_data",
    train_ratio: float = 0.7,
    dev_ratio: float = 0.15,
    seed: int = 42,
) -> dict:
    """
    Pipeline complet de préparation des données pour le fine-tuning du
    tagger POS de Stanza, à partir de phrases NON ÉTIQUETÉES.

    Étapes détaillées (affichées à l'écran pour pédagogie) :

    1. AUTO-ANNOTATION (silver standard)
       Chaque phrase brute est passée dans le pipeline fro existant
       (tokenize, pos, lemma, depparse). Les annotations produites ne
       sont PAS la vérité terrain — elles reflètent les erreurs du
       modèle actuel. C'est volontaire : le fine-tuning vise justement
       à corriger ces erreurs sur le nouveau domaine/corpus, à condition
       qu'une relecture humaine corrige au moins une partie du silver
       standard avant l'entraînement (étape non automatisable, donc non
       simulée ici — voir l'avertissement dans le code).

    2. CONVERSION CoNLL-U
       Chaque token annoté est sérialisé selon le format à 10 colonnes
       attendu par stanza.utils.training.run_pos :
       ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC.

    3. SPLIT train/dev/test
       Stanza requiert explicitement les trois fichiers séparés pour
       l'entraînement du tagger (train pour l'optimisation, dev pour le
       early-stopping, test pour l'évaluation finale).

    4. ÉCRITURE SUR DISQUE
       Les fichiers sont nommés selon la convention attendue par les
       scripts d'entraînement Stanza : {shorthand}-ud-{split}.conllu
       Le "shorthand" pour un nouveau treebank personnalisé doit suivre
       le format attendu par stanza.utils.datasets.prepare_pos_treebank
       (voir le notebook Colab pour le détail de cette convention).

    5. RAPPORT DE COUVERTURE
       Statistiques de diagnostic : taille du vocabulaire, distribution
       des UPOS, longueur moyenne des phrases. Un corpus trop petit ou
       trop déséquilibré produira un fine-tuning de mauvaise qualité —
       ce rapport permet de le détecter AVANT de lancer l'entraînement
       (potentiellement coûteux en temps GPU).

    Paramètres
    ----------
    raw_sentences : list[str]   phrases brutes, non étiquetées
    nlp           : stanza.Pipeline   pipeline fro déjà construit
    output_dir    : str         dossier de sortie des fichiers .conllu
    train_ratio   : float       proportion du split train
    dev_ratio     : float       proportion du split dev (le reste va au test)
    seed          : int         graine du split aléatoire (reproductibilité)

    Retourne
    --------
    dict   chemins des fichiers générés + rapport de couverture
    """
    import os

    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)

    # ── Étape 1 : auto-annotation (silver standard) ─────────────────────
    print("\n[10.1] Auto-annotation des phrases non étiquetées (silver standard)...")
    print(
        textwrap.dedent("""
        AVERTISSEMENT PÉDAGOGIQUE :
        Les annotations produites ici proviennent du modèle fro existant —
        elles contiennent donc déjà les erreurs de ce modèle. En conditions
        réelles de fine-tuning, une relecture humaine (même partielle, même
        sur un sous-ensemble) est nécessaire pour que le fine-tuning améliore
        effectivement le modèle plutôt que de renforcer ses erreurs actuelles.
        Cette relecture n'est pas automatisable et n'est donc pas simulée ici.
    """)
    )

    annotated_sentences = []
    for raw_text in raw_sentences:
        doc = nlp(raw_text)
        for sent in doc.sentences:
            tokens = []
            for word in sent.words:
                tokens.append(
                    {
                        "id": word.id,
                        "text": word.text,
                        "lemma": word.lemma or "_",
                        "upos": word.upos or "_",
                        "xpos": word.xpos or "_",
                        "feats": word.feats or "_",
                        "head": word.head,
                        "deprel": word.deprel or "_",
                    }
                )
            annotated_sentences.append({"text": sent.text, "tokens": tokens})

    print(f"  {len(annotated_sentences)} phrases auto-annotées.")

    # ── Étape 2+3 : conversion CoNLL-U + split ──────────────────────────
    print("\n[10.2-10.3] Conversion CoNLL-U et split train/dev/test...")
    indices = list(range(len(annotated_sentences)))
    random.shuffle(indices)
    n = len(indices)
    n_train = max(1, int(n * train_ratio))
    n_dev = max(1, int(n * dev_ratio))
    splits = {
        "train": [annotated_sentences[i] for i in indices[:n_train]],
        "dev": [annotated_sentences[i] for i in indices[n_train : n_train + n_dev]],
        "test": [annotated_sentences[i] for i in indices[n_train + n_dev :]]
        or [annotated_sentences[indices[-1]]],  # garantir >= 1 phrase
    }
    for split_name, sents in splits.items():
        print(f"  {split_name:<6} : {len(sents)} phrases")

    # ── Étape 4 : écriture sur disque ───────────────────────────────────
    print("\n[10.4] Écriture des fichiers CoNLL-U...")
    # Convention de nommage Stanza : {shorthand}-ud-{split}.conllu
    # "fro_custom" est le shorthand choisi pour ce corpus de démonstration.
    shorthand = "fro_custom"
    written_paths = {}
    for split_name, sents in splits.items():
        path = os.path.join(output_dir, f"{shorthand}-ud-{split_name}.conllu")
        with open(path, "w", encoding="utf-8") as f:
            for i, sent in enumerate(sents, start=1):
                f.write(f"# sent_id = {split_name}_{i:04d}\n")
                f.write(f"# text = {sent['text']}\n")
                for tok in sent["tokens"]:
                    f.write(
                        "\t".join(
                            [
                                str(tok["id"]),
                                tok["text"],
                                tok["lemma"],
                                tok["upos"],
                                tok["xpos"],
                                tok["feats"],
                                str(tok["head"]),
                                tok["deprel"],
                                "_",
                                "_",
                            ]
                        )
                        + "\n"
                    )
                f.write("\n")
        written_paths[split_name] = path
        print(f"  Écrit : {path}")

    # ── Étape 5 : rapport de couverture ─────────────────────────────────
    print("\n[10.5] Rapport de couverture du corpus...")
    all_tokens = [tok["text"] for sent in annotated_sentences for tok in sent["tokens"]]
    all_upos = [tok["upos"] for sent in annotated_sentences for tok in sent["tokens"]]
    sentence_lengths = [len(sent["tokens"]) for sent in annotated_sentences]

    report = {
        "n_sentences": len(annotated_sentences),
        "n_tokens": len(all_tokens),
        "vocab_size": len(set(t.lower() for t in all_tokens)),
        "avg_sentence_length": sum(sentence_lengths) / max(len(sentence_lengths), 1),
        "upos_distribution": dict(Counter(all_upos)),
    }

    print(f"  Phrases               : {report['n_sentences']}")
    print(f"  Tokens                : {report['n_tokens']}")
    print(f"  Taille du vocabulaire : {report['vocab_size']}")
    print(f"  Longueur moy. phrase  : {report['avg_sentence_length']:.1f} tokens")
    print(f"  Distribution UPOS     : {report['upos_distribution']}")

    if report["n_sentences"] < 50:
        print(
            textwrap.dedent(f"""
            ATTENTION : seulement {report["n_sentences"]} phrases dans ce corpus.
            Un fine-tuning sérieux du tagger POS requiert généralement plusieurs
            centaines à quelques milliers de phrases annotées pour produire un
            gain mesurable. Ce corpus de {report["n_sentences"]} phrases est
            UNIQUEMENT démonstratif — il illustre le PROCESSUS de préparation,
            pas un volume suffisant pour un fine-tuning en conditions réelles.
        """)
        )

    return {"paths": written_paths, "report": report, "shorthand": shorthand}


def explain_training_command(shorthand: str, data_dir: str) -> None:
    """
    Affiche les commandes shell nécessaires pour lancer l'entraînement
    effectif, à exécuter depuis un clone du dépôt stanfordnlp/stanza
    (PAS depuis l'API Pipeline Python — cf. avertissement en tête de
    cette partie). Ces commandes sont détaillées et exécutées dans le
    notebook Colab fourni séparément (GPU T4 requis).
    """
    print(
        textwrap.dedent(f"""
        ┌─────────────────────────────────────────────────────────────┐
        │  COMMANDES D'ENTRAÎNEMENT (à exécuter sur Colab, PAS ici)   │
        ├─────────────────────────────────────────────────────────────┤
        │  Stanza n'entraîne pas via l'API Pipeline Python. Il faut : │
        │                                                             │
        │  1. git clone https://github.com/stanfordnlp/stanza.git     │
        │  2. Placer les fichiers .conllu générés ci-dessus dans      │
        │     le dossier attendu par prepare_pos_treebank             │
        │  3. python3 -m stanza.utils.training.run_pos {shorthand} \\ |
        │       --train_file {data_dir}/{shorthand}-ud-train.conllu \\|
        │       --eval_file   {data_dir}/{shorthand}-ud-dev.conllu \\ |
        │       --max_steps 1000                                      │
        │                                                             │
        │  Voir le notebook Colab joint pour l'exécution complète     │
        │  avec GPU T4 (gratuit), incluant le clonage du dépôt et     │
        │  l'installation des vecteurs de mots pré-entraînés requis.  │
        └─────────────────────────────────────────────────────────────┘
    """)
    )


def run_finetuning_data_prep_demo() -> None:
    """Point d'entrée pour la démonstration de la Partie 2 (préparation
    des données uniquement — pas d'entraînement effectif dans ce script)."""
    print("\n" + "=" * 70)
    print("PARTIE 2 — PRÉPARATION DES DONNÉES POUR LE FINE-TUNING (tagger POS)")
    print("=" * 70)

    print("\nConstruction du pipeline fro pour l'auto-annotation...")
    nlp = build_pipeline_fro()

    result = prepare_finetuning_data(
        raw_sentences=UNLABELED_CORPUS_DEMO,
        nlp=nlp,
        output_dir="./pos_finetuning_data",
    )

    explain_training_command(result["shorthand"], "./pos_finetuning_data")

    print("\n" + "=" * 70)
    print("FIN DE LA DÉMONSTRATION — PARTIE 2 (préparation des données)")
    print("L'entraînement effectif est réservé au notebook Colab (GPU T4).")
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    use_network = "--offline" not in sys.argv

    # Partie 1 : pipeline Stanza complet sur le corpus médiéval
    run_full_demo(use_network=use_network)

    # Partie 2 : préparation des données pour le fine-tuning du tagger POS
    # (l'entraînement effectif se fait sur Colab, voir le notebook fourni)
    run_finetuning_data_prep_demo()
