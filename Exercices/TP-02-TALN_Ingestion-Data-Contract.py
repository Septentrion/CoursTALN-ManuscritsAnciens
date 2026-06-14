"""
TP Guidé — Ingestion du data contract HTR → NLP & Analyse exploratoire
Module NLP · Master Data/IA · MD5 Volet 2 · 2026
──────────────────────────────────────────────────────────────────────
Objectif : prendre en main le JSON livré par le module Computer Vision,
le valider, le disséquer, et en extraire les métriques de baseline
indispensables avant tout fine-tuning.

La chaîne que vous construisez ici :

  JSON data contract HTR → validation schéma → aplatissement DataFrame
  → analyse needs_review → inventaire abréviations → couverture OOV
  → visualisation confiances → split SHA-256 → tableau de baseline

Durée estimée : 3 h 30
Livrables attendus en fin de séance :
  - abbreviations_inventory.json
  - needs_review_report.json
  - split_manifest.json

Instructions générales
──────────────────────
Les cellules marquées  # TODO  contiennent des squelettes à compléter.
Les cellules marquées  # FOURNI  sont à exécuter telles quelles.
Ne modifiez pas les signatures de fonctions : les validations s'appuient dessus.
"""

# %% [markdown]
# # TP — Ingestion du data contract HTR → NLP
#
# Ce TP vous fait parcourir la chaîne complète depuis le JSON brut
# livré par le Volet 1 jusqu'au split train/val/test verrouillé,
# en produisant au passage les métriques de baseline dont vous aurez
# besoin pour interpréter vos résultats de fine-tuning au Jour 2.
#
# **Règle du TP :** avant de fermer votre notebook, vous devez être
# capables de remplir, sans le consulter, le tableau de baseline
# de la Partie 7. Si vous ne connaissez pas ces chiffres par cœur,
# vous n'avez pas terminé.

# %% [markdown]
# ## Partie 0 — Imports et corpus synthétique

# %% [FOURNI]
import json, hashlib, re, math, random, collections, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import jsonschema
from sklearn.model_selection import GroupShuffleSplit

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

print("Environnement prêt.")

# %% [markdown]
# ### Corpus de travail
#
# Vous travaillerez sur un corpus synthétique qui reproduit fidèlement
# la structure du data contract du module. Il contient 16 documents
# répartis en quatre types (charte, registre, roman, liturgique) avec
# des lignes propres et des lignes `needs_review`.
#
# En TP réel, remplacez la cellule ci-dessous par :
# ```python
# corpus_docs = []
# for path in sorted(Path("data/htr_output/").glob("*.json")):
#     with open(path, encoding="utf-8") as f:
#         corpus_docs.append(json.load(f))
# ```

# %% [FOURNI]
# ── Générateur de corpus synthétique ────────────────────────────────────────
_LIGNES = {
    "charte": [
        ("Li roys de france signa l acte en son palays", 0.94, False),
        ("au duc de norm~die et au co~te de champagne",  0.71, True),
        ("la chartre fut seellee du seel royal",          0.89, False),
        ("en l an de grace mil trois cent quarante",      0.92, False),
        ("q~ li chevalier ait restitue les terres",       0.67, True),
        ("don de la terre de gisors au chastelain",       0.88, False),
        ("ledit seigneur porta les lettres patentes",     0.91, False),
        ("pñce de normandie fit don a l eglise",          0.73, True),
    ],
    "registre": [
        ("item le bailli rendit son jugement",            0.85, False),
        ("co~te des deniers receus au tresor",            0.68, True),
        ("messire guillaumes de villehardouin",           0.87, False),
        ("au moys de mars de l an susdit",                0.93, False),
        ("pour le paiement de vingt l~ de rente",         0.66, True),
        ("ledit preudomme fu condamne a l amende",        0.82, False),
        ("no~ du seneschal de champagne",                 0.70, True),
    ],
    "roman": [
        ("li chevaliers arma son destrier",               0.78, True),
        ("et prist sa lance et son escu",                 0.91, False),
        ("la dame du chastel vit le chevalier",           0.83, False),
        ("q~ feist il dist li roys",                      0.62, True),
        ("lors s en ala vers le palays roial",            0.79, True),
        ("messire gauvain respondi au roi",               0.88, False),
        ("la pucele estoit de grant beaute",              0.85, False),
    ],
    "liturgique": [
        ("in nomine patris et filii",                     0.96, False),
        ("dominus vobiscum et cum spiritu tuo",           0.97, False),
        ("gloria in excelsis deo",                        0.95, False),
        ("kyrie eleison christe eleison",                 0.94, False),
        ("per omnia secula seculorum amen",               0.93, False),
    ],
}

def _make_char_confs(text, conf, needs_review):
    confs = []
    for ch in text:
        if ch == ' ':
            confs.append(1.0)
        elif ch in '~ñ':
            confs.append(round(random.uniform(0.50, 0.65), 2))
        elif needs_review:
            confs.append(round(random.uniform(0.55, 0.85), 2))
        else:
            confs.append(round(min(0.99, max(0.70, conf + random.uniform(-0.05, 0.05))), 2))
    return confs

def _make_candidates(text):
    cands = []
    for i, ch in enumerate(text):
        if ch == '~':
            s = round(random.uniform(0.52, 0.65), 2)
            cands.append({"position": i, "alternatives": ["~", "a"], "scores": [s, round(1-s, 2)]})
        elif ch == 'ñ':
            s = round(random.uniform(0.55, 0.70), 2)
            cands.append({"position": i, "alternatives": ["ñ", "n"], "scores": [s, round(1-s, 2)]})
    return cands

random.seed(SEED)
corpus_docs = []
counter = 0
for doc_type, lignes in _LIGNES.items():
    for _ in range(4):
        counter += 1
        doc_id = f"{doc_type}_{counter:04d}"
        sel = random.choices(lignes, k=random.randint(5, 10))
        lines_out = []
        for li, (text, conf, nr) in enumerate(sel):
            conf_j = round(conf + random.uniform(-0.03, 0.03), 3)
            lines_out.append({
                "line_id":          f"l{li+1:03d}",
                "transcription":    text,
                "confidence":       conf_j,
                "needs_review":     nr,
                "char_confidences": _make_char_confs(text, conf_j, nr),
                "candidates":       _make_candidates(text),
                "polygon":          [[100, 40+li*25],[380, 40+li*25],
                                     [380, 63+li*25],[100, 63+li*25]],
            })
        nr_count = sum(1 for l in lines_out if l["needs_review"])
        corpus_docs.append({
            "document_id":   doc_id,
            "document_type": doc_type,
            "metadata": {
                "htr_model":         "cremma-medieval-v2",
                "cer_estimate":      round(random.uniform(0.06, 0.11), 3),
                "total_lines":       len(lines_out),
                "needs_review_count": nr_count,
            },
            "pages": [{"page_id": "p001", "lines": lines_out}],
        })

print(f"Corpus synthétique : {len(corpus_docs)} documents")
for dt in ["charte", "registre", "roman", "liturgique"]:
    n = sum(1 for d in corpus_docs if d["document_type"] == dt)
    print(f"  {dt:12s}: {n} documents")

# %% [markdown]
# ## Partie 1 — Validation du schéma JSON
#
# ### 1.1 Le schéma de référence
#
# Avant toute analyse, nous vérifions que chaque document respecte le contrat.
# Un fichier invalide (champ manquant, type incorrect, valeur hors-bornes)
# peut produire des bugs silencieux qui contaminent toute l'analyse en aval.
#
# Le schéma ci-dessous est fourni. Lisez-le attentivement : il matérialise
# les garanties que le Volet 1 doit respecter.

# %% [FOURNI]
SCHEMA = {
    "type": "object",
    "required": ["document_id", "document_type", "metadata", "pages"],
    "properties": {
        "document_id":   {"type": "string"},
        "document_type": {"enum": ["charte", "registre", "roman", "liturgique"]},
        "metadata": {
            "type": "object",
            "required": ["htr_model", "cer_estimate", "total_lines", "needs_review_count"],
            "properties": {
                "cer_estimate":       {"type": "number", "minimum": 0, "maximum": 1},
                "needs_review_count": {"type": "integer"},
            },
        },
        "pages": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["lines"],
                "properties": {
                    "lines": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["line_id", "transcription", "confidence",
                                         "needs_review", "char_confidences",
                                         "candidates", "polygon"],
                            "properties": {
                                "confidence":       {"type": "number",
                                                     "minimum": 0, "maximum": 1},
                                "needs_review":     {"type": "boolean"},
                                "char_confidences": {"type": "array",
                                                     "items": {"type": "number"}},
                                "polygon":          {"type": "array",
                                                     "minItems": 4, "maxItems": 4},
                            },
                        },
                    }
                },
            },
        },
    },
}

# %% [markdown]
# ### 1.2 Validation et rapport d'erreurs
#
# **À vous de jouer.**
#
# Complétez `validate_corpus` : elle valide chaque document du corpus
# et retourne deux listes — les documents valides et les erreurs détectées.

# %% [TODO]
def validate_corpus(docs: list, schema: dict) -> tuple[list, list]:
    """
    Valide chaque document du corpus contre le schéma JSON.

    Paramètres
    ----------
    docs   : list  liste de dictionnaires (documents JSON)
    schema : dict  schéma jsonschema de référence

    Retourne
    --------
    valid_docs : list  documents conformes au schéma
    errors     : list  liste de tuples (document_id, message_erreur)

    Algorithme
    ----------
    Pour chaque doc dans docs :
      1. Récupérer doc.get("document_id", "INCONNU") pour l'identifier.
      2. Appeler jsonschema.validate(doc, schema).
         - Si aucune exception : ajouter doc à valid_docs.
         - Si jsonschema.ValidationError : ajouter (doc_id, e.message) à errors.
    Retourner (valid_docs, errors).
    """
    valid_docs = []
    errors     = []
    for doc in docs:
        doc_id = doc.get("document_id", "INCONNU")
        # TODO : implémenter la validation
        pass   # ← remplacer cette ligne
    return valid_docs, errors

# %% [markdown]
# **Cellule de validation 1.2**

# %% [FOURNI — validation]
_valid, _errors = validate_corpus(corpus_docs, SCHEMA)
assert len(_valid) == len(corpus_docs), \
    f"Tous les docs valides doivent être dans valid_docs. Obtenus : {len(_valid)}/{len(corpus_docs)}"
assert isinstance(_errors, list), "errors doit être une liste."

# Test avec un document intentionnellement invalide
_bad_doc = {"document_id": "bad_001", "document_type": "INVALIDE",
            "metadata": {}, "pages": []}
_v2, _e2 = validate_corpus([_bad_doc], SCHEMA)
assert len(_e2) == 1, "Un document invalide doit produire une erreur."
print(f"Validation 1.2 : OK — {len(_valid)}/{len(corpus_docs)} documents valides")
print(f"  Test doc invalide : erreur détectée → {_e2[0][1][:60]}...")

# %% [FOURNI]
# Rapport final
valid_docs = _valid
print(f"\nCorpus validé : {len(valid_docs)} documents conformes, {len(_errors)} erreurs")

# %% [markdown]
# ## Partie 2 — Aplatissement en DataFrame
#
# Le JSON est hiérarchique (document → pages → lines). Pour l'analyse,
# nous avons besoin d'un DataFrame plat où chaque ligne du manuscrit
# est une observation, enrichie des métadonnées du document parent.
#
# **À vous de jouer.**
#
# Complétez `flatten_corpus`. Chaque ligne du DataFrame doit contenir :
# tous les champs de la ligne (line_id, transcription, confidence…)
# **plus** document_id, document_type, page_id et cer_estimate.

# %% [TODO]
def flatten_corpus(docs: list) -> pd.DataFrame:
    """
    Aplati le corpus JSON en un DataFrame Pandas.

    Paramètre
    ---------
    docs : list  documents JSON validés

    Retourne
    --------
    pd.DataFrame  une ligne par ligne de transcription, avec colonnes :
        line_id, transcription, confidence, needs_review,
        char_confidences, candidates, polygon,
        document_id, document_type, page_id, cer_estimate

    Algorithme
    ----------
    rows = []
    Pour chaque doc dans docs :
      Pour chaque page dans doc["pages"] :
        Pour chaque line dans page["lines"] :
          Créer un dictionnaire fusionnant :
            - tous les champs de line  (utilisez **line)
            - document_id   = doc["document_id"]
            - document_type = doc["document_type"]
            - page_id       = page["page_id"]
            - cer_estimate  = doc["metadata"]["cer_estimate"]
          Ajouter ce dictionnaire à rows.
    Retourner pd.DataFrame(rows)
    """
    rows = []
    for doc in docs:
        for page in doc["pages"]:
            for line in page["lines"]:
                # TODO : construire le dictionnaire et l'ajouter à rows
                pass   # ← remplacer cette ligne
    return pd.DataFrame(rows)

# %% [markdown]
# **Cellule de validation 2**

# %% [FOURNI — validation]
df = flatten_corpus(valid_docs)
_required_cols = {"line_id", "transcription", "confidence", "needs_review",
                  "char_confidences", "candidates", "polygon",
                  "document_id", "document_type", "page_id", "cer_estimate"}
assert _required_cols.issubset(set(df.columns)), \
    f"Colonnes manquantes : {_required_cols - set(df.columns)}"
assert len(df) > 0, "Le DataFrame ne doit pas être vide."
assert df["needs_review"].dtype == bool or df["needs_review"].dtype == object, \
    "La colonne needs_review doit contenir des booléens."
print(f"Validation 2 : OK — {df.shape[0]} lignes × {df.shape[1]} colonnes")
print(f"  Types de documents : {sorted(df['document_type'].unique())}")
print(f"  Lignes needs_review : {df['needs_review'].sum()} "
      f"({df['needs_review'].mean()*100:.1f} %)")

# %% [markdown]
# ## Partie 3 — Analyse des lignes `needs_review`
#
# ### 3.1 Taux par type de document
#
# **À vous de jouer.** Complétez `compute_needs_review_stats` :
# elle retourne un DataFrame avec le taux de `needs_review` par type de document,
# trié du taux le plus élevé au plus faible.

# %% [TODO]
def compute_needs_review_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les statistiques needs_review par type de document.

    Paramètre
    ---------
    df : DataFrame  corpus aplati

    Retourne
    --------
    pd.DataFrame  avec colonnes n_review, n_total, taux (en %),
                  indexé par document_type,
                  trié par taux décroissant.

    Indice
    ------
    Utilisez df.groupby("document_type")["needs_review"].agg(...).
    La colonne taux doit être en pourcentage, arrondi à 1 décimale.
    """
    # TODO : implémenter le calcul
    pass   # ← remplacer cette ligne

# %% [markdown]
# **Cellule de validation 3.1**

# %% [FOURNI — validation]
_stats = compute_needs_review_stats(df)
assert _stats is not None, "La fonction doit retourner un DataFrame (pas None)."
assert "n_review" in _stats.columns and "n_total" in _stats.columns, \
    "Colonnes n_review et n_total requises."
assert "taux" in _stats.columns, "Colonne taux requise."
assert _stats["taux"].iloc[0] >= _stats["taux"].iloc[-1], \
    "Le DataFrame doit être trié par taux décroissant."
# Les textes liturgiques ont 0 % de needs_review dans notre corpus
assert _stats.loc["liturgique", "taux"] == 0.0, \
    "Les textes liturgiques (propres) doivent avoir un taux de 0 %."
print("Validation 3.1 : OK")
print(_stats.to_string())

# %% [FOURNI]
# Visualisation
fig, ax = plt.subplots(figsize=(8, 4))
colors = {"charte": "#3A7EBF", "registre": "#E87B3E",
          "roman": "#2CA02C", "liturgique": "#9467BD"}
bars = ax.barh(
    _stats.index,
    _stats["taux"],
    color=[colors.get(t, "#888") for t in _stats.index],
)
ax.bar_label(bars, fmt="%.1f %%", padding=4)
ax.axvline(df["needs_review"].mean() * 100, color="red",
           linestyle="--", label=f"Moyenne globale ({df['needs_review'].mean()*100:.1f} %)")
ax.set_xlabel("Taux needs_review (%)")
ax.set_title("Taux de lignes needs_review par type de document")
ax.legend()
plt.tight_layout()
plt.savefig("needs_review_by_type.pdf", dpi=150)
plt.show()

# %% [markdown]
# **Question 3.1** : Le taux `needs_review` varie-t-il de façon significative
# selon le type de document ? Quelle hypothèse formulez-vous pour expliquer
# la différence entre les romans et les textes liturgiques ?
# Quelles conséquences sur la stratégie de split train/val/test ?

# %% [markdown]
# ### 3.2 Diagnostic des causes
#
# **À vous de jouer.** Complétez `diagnose_line` : elle identifie les
# causes probables du flag `needs_review` pour une ligne donnée.
# Une ligne peut avoir plusieurs causes simultanées.

# %% [TODO]
def diagnose_line(row: pd.Series) -> list[str]:
    """
    Identifie les causes probables du flag needs_review d'une ligne.

    Paramètre
    ---------
    row : pd.Series  une ligne du DataFrame corpus

    Retourne
    --------
    list[str]  liste de causes parmi :
        "confidence_basse"          si row["confidence"] < 0.75
        "caractere_tres_incertain"  si min(char_confidences) < 0.5
        "candidats_alternatifs"     si len(candidates) > 0
        "abreviation_residuelle"    si la transcription contient ~ ou ñ
        "ligne_trop_courte"         si la transcription a < 3 mots
        "cause_inconnue"            si aucune autre cause n'est détectée

    Indices
    -------
    - Accédez à row["confidence"], row["char_confidences"], etc.
    - Pour détecter ~ ou ñ : utilisez re.search(r'[~ñ]', row["transcription"])
    - Pour compter les mots : len(row["transcription"].split())
    - Retourner ["cause_inconnue"] si causes est vide en fin de fonction.
    """
    causes = []
    # TODO : implémenter les 5 conditions
    pass   # ← remplacer cette ligne
    return causes if causes else ["cause_inconnue"]

# %% [markdown]
# **Cellule de validation 3.2**

# %% [FOURNI — validation]
# Ligne avec tilde → doit détecter abreviation_residuelle
_row_abbrev = df[df["transcription"].str.contains("~")].iloc[0]
_causes_a   = diagnose_line(_row_abbrev)
assert "abreviation_residuelle" in _causes_a, \
    f"Une ligne avec ~ doit détecter 'abreviation_residuelle'. Obtenu : {_causes_a}"

# Ligne propre (liturgique)
_row_clean = df[(df["document_type"] == "liturgique") &
                (~df["needs_review"])].iloc[0]
_causes_c  = diagnose_line(_row_clean)
assert "cause_inconnue" in _causes_c or len(_causes_c) == 0 or \
       all(c not in ["confidence_basse", "abreviation_residuelle"]
           for c in _causes_c), \
    f"Une ligne liturgique propre ne devrait pas avoir de cause problématique. Obtenu : {_causes_c}"
print("Validation 3.2 : OK")

# %% [FOURNI]
# Application et rapport
review_df = df[df["needs_review"]].copy()
review_df["causes"] = review_df.apply(diagnose_line, axis=1)

cause_counter = collections.Counter(
    c for causes in review_df["causes"] for c in causes
)
print("\nDiagnostic des causes needs_review :")
for cause, n in cause_counter.most_common():
    pct = n / max(len(review_df), 1) * 100
    print(f"  {cause:35s} {n:4d} lignes  ({pct:.0f} %)")

# %% [markdown]
# ### 3.3 Cohérence entre `needs_review` et confiance globale
#
# **À vous de jouer.** Complétez `compute_geometric_mean` : elle calcule
# la moyenne géométrique des `char_confidences` d'une ligne,
# qui est la formule utilisée par le modèle HTR pour agréger les scores
# par caractère en un score global de ligne.

# %% [TODO]
def compute_geometric_mean(char_confs: list) -> float:
    """
    Calcule la moyenne géométrique d'une liste de scores de confiance.

    Formule
    -------
    G = exp( (1/n) * Σ log(conf_i) )

    Paramètre
    ---------
    char_confs : list[float]  scores par caractère (entre 0 et 1)

    Retourne
    --------
    float  moyenne géométrique (0.0 si la liste est vide)

    Indice
    ------
    Utilisez math.log et math.exp.
    Pour éviter log(0), clampez chaque valeur : max(x, 1e-9).
    """
    if not char_confs:
        return 0.0
    # TODO : calculer et retourner la moyenne géométrique
    pass   # ← remplacer cette ligne

# %% [markdown]
# **Cellule de validation 3.3**

# %% [FOURNI — validation]
assert abs(compute_geometric_mean([1.0, 1.0, 1.0]) - 1.0) < 1e-6, \
    "La moyenne géométrique de [1,1,1] doit valoir 1."
assert abs(compute_geometric_mean([0.5, 0.5]) - 0.5) < 1e-6, \
    "La moyenne géométrique de [0.5, 0.5] doit valoir 0.5."
assert compute_geometric_mean([]) == 0.0, \
    "La moyenne géométrique d'une liste vide doit valoir 0."
_test_geo = compute_geometric_mean([0.9, 0.8, 0.7])
assert 0.79 < _test_geo < 0.80, \
    f"compute_geometric_mean([0.9, 0.8, 0.7]) ≈ 0.795. Obtenu : {_test_geo:.4f}"
print(f"Validation 3.3 : OK — géo([0.9, 0.8, 0.7]) = {_test_geo:.4f}")

# %% [FOURNI]
df["conf_geo"] = df["char_confidences"].apply(compute_geometric_mean)
corr = df[["confidence", "conf_geo"]].corr().iloc[0, 1]
print(f"Corrélation confiance_ligne vs géométrique_char : {corr:.3f}")

fig, ax = plt.subplots(figsize=(7, 5))
colors_nr = df["needs_review"].map({True: "#D62728", False: "#3A7EBF"})
ax.scatter(df["conf_geo"], df["confidence"], c=colors_nr, alpha=0.5, s=20)
ax.plot([0, 1], [0, 1], "k--", lw=1, label="y = x")
ax.axvline(0.75, color="orange", linestyle=":", label="Seuil needs_review (0.75)")
legend_elems = [mpatches.Patch(color="#D62728", label="needs_review"),
                mpatches.Patch(color="#3A7EBF", label="propre")]
ax.legend(handles=legend_elems + ax.get_lines())
ax.set_xlabel("Moyenne géométrique des char_confidences")
ax.set_ylabel("Confiance de ligne (champ JSON)")
ax.set_title("Cohérence : confiance ligne vs agrégation caractères")
plt.tight_layout()
plt.savefig("confidence_scatter.pdf", dpi=150)
plt.show()

# %% [markdown]
# **Question 3.3** : La corrélation entre les deux scores est-elle forte ?
# Observez-vous des points qui s'écartent notablement de la diagonale y=x ?
# Que cela implique-t-il sur la fiabilité du champ `confidence` comme
# signal unique pour la stratégie de pondération de la loss au Jour 2 ?

# %% [markdown]
# ## Partie 4 — Distribution des confiances par caractère
#
# ### 4.1 Analyse globale et par type de document
#
# **À vous de jouer.** Complétez `summarize_char_confidences` :
# elle calcule les statistiques descriptives des scores par caractère,
# en distinguant les lignes propres des lignes `needs_review`.

# %% [TODO]
def summarize_char_confidences(df: pd.DataFrame) -> dict:
    """
    Calcule les statistiques sur les char_confidences.

    Retourne
    --------
    dict avec les clés :
        "all_chars"    : list[float]  tous les scores (toutes lignes)
        "clean_chars"  : list[float]  scores des lignes non-needs_review
        "review_chars" : list[float]  scores des lignes needs_review
        "pct_below_05" : float  pourcentage de scores < 0.5 (sur all_chars)
        "pct_below_07" : float  pourcentage de scores < 0.7 (sur all_chars)

    Algorithme
    ----------
    1. Pour chaque ligne, extraire la liste char_confidences.
    2. Séparer selon needs_review.
    3. Calculer les pourcentages sur all_chars.

    Indice
    ------
    Utilisez une compréhension de liste imbriquée :
    [c for confs in df["char_confidences"] for c in confs]
    """
    # TODO : implémenter
    pass   # ← remplacer cette ligne

# %% [markdown]
# **Cellule de validation 4.1**

# %% [FOURNI — validation]
_summary = summarize_char_confidences(df)
assert _summary is not None, "La fonction doit retourner un dict."
assert "all_chars"    in _summary, "Clé 'all_chars' manquante."
assert "pct_below_05" in _summary, "Clé 'pct_below_05' manquante."
assert "pct_below_07" in _summary, "Clé 'pct_below_07' manquante."
assert 0 <= _summary["pct_below_05"] <= 100, "pct_below_05 doit être en %."
assert _summary["pct_below_07"] >= _summary["pct_below_05"], \
    "pct_below_07 doit être >= pct_below_05."
print(f"Validation 4.1 : OK")
print(f"  Chars totaux analysés : {len(_summary['all_chars'])}")
print(f"  Chars confidence < 0.5 : {_summary['pct_below_05']:.1f} %")
print(f"  Chars confidence < 0.7 : {_summary['pct_below_07']:.1f} %")

# %% [FOURNI]
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# Distribution globale
axes[0].hist(_summary["all_chars"], bins=40, color="#3A7EBF", edgecolor="white")
axes[0].axvline(0.5, color="red",    linestyle="--", label="Critique (0.5)")
axes[0].axvline(0.7, color="orange", linestyle="--", label="Attention (0.7)")
axes[0].set_xlabel("Score de confiance par caractère")
axes[0].set_ylabel("Nombre de caractères")
axes[0].set_title("Distribution des char_confidences — corpus complet")
axes[0].legend()

# Propres vs needs_review
axes[1].hist(_summary["clean_chars"],  bins=40, alpha=0.6,
             color="#3A7EBF", edgecolor="white", label="Propres")
axes[1].hist(_summary["review_chars"], bins=40, alpha=0.6,
             color="#D62728", edgecolor="white", label="needs_review")
axes[1].axvline(0.75, color="black", linestyle=":", label="Seuil NR (0.75)")
axes[1].set_xlabel("Score de confiance par caractère")
axes[1].set_title("char_confidences : propres vs needs_review")
axes[1].legend()

plt.tight_layout()
plt.savefig("char_confidence_dist.pdf", dpi=150)
plt.show()

# %% [markdown]
# **Question 4.1** : La distribution est-elle bimodale sur votre corpus ?
# Si oui, que représentent les deux modes ? Quelle est l'implication
# pour le seuil d'arbitrage CamemBERT MLM du Jour 2 (étape 3 du chapitre 4) ?

# %% [markdown]
# ## Partie 5 — Inventaire des abréviations
#
# ### 5.1 Extraction par patron regex
#
# **À vous de jouer.** Complétez `extract_abbreviations` : elle retourne
# un dictionnaire des abréviations trouvées dans un texte, classées par patron.

# %% [FOURNI]
ABBREV_PATTERNS = {
    "tilde_nasale":    r'\w*[~]\w*',      # co~te, norm~die, q~
    "lettre_speciale": r'[ñ]',           # ñ comme marque d'abréviation
    "lettre_p_barre":  r'\bpñ\w*',       # pñce (prince), pñ (prison)
    "q_tilde":         r'(?<!\w)q~(?!\w)', # q~ isolé (que) — \b ne fonctionne pas après ~
    "abreg_monnaie":   r'(?<!\w)[ls]~(?!\w)', # l~ (livres), s~ (sous) isolés
}

# %% [TODO]
def extract_abbreviations(text: str) -> dict[str, list[str]]:
    """
    Détecte les abréviations non résolues dans un texte.

    Paramètre
    ---------
    text : str  une transcription brute

    Retourne
    --------
    dict  {nom_patron: [liste_des_occurrences]}
          Ne contient que les patrons ayant au moins une occurrence.

    Exemple
    -------
    extract_abbreviations("au duc de norm~die et q~ feist")
    → {"tilde_nasale": ["norm~die", "q~"], "q_tilde": ["q~"]}

    Algorithme
    ----------
    Pour chaque (nom, pattern) dans ABBREV_PATTERNS.items() :
      matches = re.findall(pattern, text)
      Si matches : ajouter au dictionnaire résultat.
    """
    found = {}
    # TODO : implémenter
    pass   # ← remplacer cette ligne
    return found

# %% [markdown]
# **Cellule de validation 5.1**

# %% [FOURNI — validation]
_t1 = extract_abbreviations("au duc de norm~die et q~ feist")
assert "tilde_nasale" in _t1, "tilde_nasale non détecté dans 'norm~die'."
assert "norm~die" in _t1["tilde_nasale"], "'norm~die' doit figurer dans tilde_nasale."
assert "q_tilde"   in _t1, "q_tilde non détecté dans 'q~'."

_t2 = extract_abbreviations("gloria in excelsis deo")
assert len(_t2) == 0, "Aucune abréviation ne doit être détectée dans ce texte propre."
print(f"Validation 5.1 : OK")
print(f"  Abréviations dans 'au duc de norm~die et q~ feist' :")
for k, v in _t1.items():
    print(f"    {k}: {v}")

# %% [markdown]
# ### 5.2 Inventaire global et rapport
#
# **À vous de jouer.** Complétez `build_abbreviation_inventory` :
# elle construit le comptage global des abréviations sur tout le corpus,
# ventilé par type de document.

# %% [TODO]
def build_abbreviation_inventory(df: pd.DataFrame) -> tuple[dict, dict]:
    """
    Construit l'inventaire complet des abréviations du corpus.

    Retourne
    --------
    global_counts  : dict  {nom_patron: count_total}
    by_doc_type    : dict  {doc_type: {nom_patron: count}}

    Algorithme
    ----------
    Initialiser global_counts = Counter()
    Initialiser by_doc_type   = {type: Counter() for type in doc_types}

    Pour chaque ligne du DataFrame :
      found = extract_abbreviations(row["transcription"])
      Pour chaque (nom, matches) dans found.items() :
        global_counts[nom]                        += len(matches)
        by_doc_type[row["document_type"]][nom]    += len(matches)

    Retourner (dict(global_counts), {t: dict(c) for t,c in by_doc_type.items()})
    """
    doc_types     = df["document_type"].unique()
    global_counts = collections.Counter()
    by_doc_type   = {t: collections.Counter() for t in doc_types}

    for _, row in df.iterrows():
        # TODO : implémenter
        pass   # ← remplacer cette ligne

    return dict(global_counts), {t: dict(c) for t, c in by_doc_type.items()}

# %% [markdown]
# **Cellule de validation 5.2**

# %% [FOURNI — validation]
_glob, _by_type = build_abbreviation_inventory(df)
assert isinstance(_glob, dict),     "global_counts doit être un dict."
assert isinstance(_by_type, dict),  "by_doc_type doit être un dict."
assert "tilde_nasale" in _glob,     "tilde_nasale doit apparaître dans le corpus."
assert _glob["tilde_nasale"] > 0,   "Le count de tilde_nasale doit être > 0."
assert "liturgique" in _by_type,    "liturgique doit être dans by_doc_type."
# Les textes liturgiques ne contiennent pas d'abréviations médiévales
assert _by_type["liturgique"].get("tilde_nasale", 0) == 0, \
    "Les textes liturgiques ne doivent pas contenir de tilde_nasale."
print("Validation 5.2 : OK")
total = sum(_glob.values())
print(f"\nInventaire global ({total} abréviations) :")
for name, count in sorted(_glob.items(), key=lambda x: -x[1]):
    print(f"  {name:25s}: {count:4d} ({count/max(total,1)*100:.0f} %)")

# %% [FOURNI]
# Sauvegarde de l'inventaire (livrable du TP)
inventory_report = {
    "total_abbreviations": int(total),
    "global_counts":       _glob,
    "by_document_type":    _by_type,
    "lines_with_abbrev":   int(df["transcription"].apply(
        lambda t: bool(extract_abbreviations(t))
    ).sum()),
}
with open("abbreviations_inventory.json", "w", encoding="utf-8") as f:
    json.dump(inventory_report, f, indent=2, ensure_ascii=False)
print("\nLivrable sauvegardé : abbreviations_inventory.json")

# %% [markdown]
# ### 5.3 Couverture de la table statique
#
# **À vous de jouer.** Complétez `measure_table_coverage` : elle mesure
# quelle fraction des abréviations détectées peut être résolue par
# la table de substitution statique fournie.

# %% [FOURNI]
ABBREV_TABLE = {
    "q~":      "que",       "Q~":       "Que",
    "no~":     "nom",       "no~e":     "nome",
    "norm~die":"normandie", "co~te":    "conte",
    "champ~e": "champagne", "l~":       "livres",
    "s~":      "sous",      "pñce":     "prince",
    "pñ":      "prison",
}

# %% [TODO]
def measure_table_coverage(df: pd.DataFrame, table: dict) -> dict:
    """
    Mesure la couverture de la table statique sur le corpus.

    Retourne
    --------
    dict avec :
        "n_total"    : int    nombre total d'occurrences d'abréviations
        "n_resolved" : int    nombre résolvables par la table
        "coverage"   : float  n_resolved / n_total (entre 0 et 1)
        "unresolved_examples" : list[str]  exemples d'abréviations non couvertes
                                           (dédupliqués, au plus 10)

    Algorithme
    ----------
    Pour chaque ligne du DataFrame :
      Pour chaque occurrence dans re.findall(r'\w*[~ñ]\w*', transcription) :
        n_total += 1
        Si occurrence dans table : n_resolved += 1
        Sinon : ajouter à unresolved_set
    """
    n_total, n_resolved = 0, 0
    unresolved_set      = set()

    for _, row in df.iterrows():
        for match in re.findall(r'\w*[~ñ]\w*', row["transcription"]):
            # TODO : incrémenter n_total, vérifier la table, alimenter unresolved_set
            pass   # ← remplacer cette ligne

    return {
        "n_total":    n_total,
        "n_resolved": n_resolved,
        "coverage":   n_resolved / max(n_total, 1),
        "unresolved_examples": sorted(unresolved_set)[:10],
    }

# %% [markdown]
# **Cellule de validation 5.3**

# %% [FOURNI — validation]
_coverage = measure_table_coverage(df, ABBREV_TABLE)
assert "n_total"    in _coverage, "Clé 'n_total' manquante."
assert "n_resolved" in _coverage, "Clé 'n_resolved' manquante."
assert "coverage"   in _coverage, "Clé 'coverage' manquante."
assert 0 <= _coverage["coverage"] <= 1, "coverage doit être entre 0 et 1."
assert _coverage["n_total"] > 0, "n_total doit être > 0 (le corpus contient des abbréviations)."
print(f"Validation 5.3 : OK")
print(f"  Abréviations totales   : {_coverage['n_total']}")
print(f"  Résolues par la table  : {_coverage['n_resolved']} "
      f"({_coverage['coverage']*100:.0f} %)")
print(f"  Exemples non couverts  : {_coverage['unresolved_examples']}")

# %% [markdown]
# **Question 5.3** : Quelle fraction des abréviations votre table couvre-t-elle ?
# Les abréviations non couvertes appartiennent-elles à un patron particulier
# (formule latine, titre, monnaie) ? Cette analyse préfigure la priorisation
# du travail de la semaine — quels patrons faut-il ajouter en priorité ?

# %% [markdown]
# ## Partie 6 — Longueurs et couverture tokenisation
#
# ### 6.1 Distribution des longueurs
#
# **À vous de jouer.** Complétez `compute_length_stats` : elle calcule
# les statistiques de longueur des lignes, en tokens bruts (mots séparés par
# des espaces) et en caractères.

# %% [TODO]
def compute_length_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute au DataFrame les colonnes n_tokens et n_chars,
    et retourne un résumé statistique par type de document.

    Colonnes ajoutées au DataFrame (modification en place) :
        n_tokens : int  nombre de mots (split() sur la transcription)
        n_chars  : int  nombre de caractères (len de la transcription)

    Retourne
    --------
    pd.DataFrame  résumé avec médiane, P95 de n_tokens, par type de document

    Indice
    ------
    Utilisez df.groupby("document_type")["n_tokens"].agg(
        mediane=("median"), p95=lambda x: x.quantile(0.95)
    )
    """
    df["n_tokens"] = df["transcription"].apply(lambda t: len(t.split()))
    df["n_chars"]  = df["transcription"].apply(len)
    # TODO : calculer et retourner le résumé par type de document
    pass   # ← remplacer cette ligne

# %% [markdown]
# **Cellule de validation 6.1**

# %% [FOURNI — validation]
_len_summary = compute_length_stats(df)
assert "n_tokens" in df.columns, "La colonne n_tokens doit être ajoutée au DataFrame."
assert "n_chars"  in df.columns, "La colonne n_chars doit être ajoutée au DataFrame."
assert _len_summary is not None, "compute_length_stats doit retourner un DataFrame."
print("Validation 6.1 : OK")
print(f"\nStatistiques de longueur :")
print(f"  Médiane globale : {df['n_tokens'].median():.0f} tokens")
print(f"  P95 global      : {df['n_tokens'].quantile(0.95):.0f} tokens")
print(f"\nPar type de document :")
print(_len_summary.to_string())

# %% [FOURNI]
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].hist(df["n_tokens"], bins=20, color="#3A7EBF", edgecolor="white")
axes[0].axvline(df["n_tokens"].median(), color="red",
                linestyle="--", label=f"Médiane ({df['n_tokens'].median():.0f})")
axes[0].axvline(df["n_tokens"].quantile(0.95), color="orange",
                linestyle="--", label=f"P95 ({df['n_tokens'].quantile(0.95):.0f})")
axes[0].set_xlabel("Longueur (tokens bruts)")
axes[0].set_title("Distribution des longueurs de lignes")
axes[0].legend()

colors = {"charte":"#3A7EBF","registre":"#E87B3E","roman":"#2CA02C","liturgique":"#9467BD"}
for doc_type, grp in df.groupby("document_type"):
    axes[1].hist(grp["n_tokens"], bins=15, alpha=0.5,
                 color=colors.get(doc_type, "#888"), label=doc_type)
axes[1].set_xlabel("Longueur (tokens bruts)")
axes[1].set_title("Longueurs par type de document")
axes[1].legend()
plt.tight_layout()
plt.savefig("length_distribution.pdf", dpi=150)
plt.show()

# %% [markdown]
# ### 6.2 Simulation du facteur de fragmentation BPE
#
# Sans accès à CamemBERT dans cet environnement, nous simulons le facteur
# de fragmentation : les mots médiévaux connus et les mots contenant des
# abréviations génèrent plus de tokens BPE que les mots courants.
#
# En TP réel avec GPU/réseau, remplacez cette cellule par :
# ```python
# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("almanach/camembert-base")
#
# def real_fragmentation(text):
#     words = text.split()
#     tokens = tokenizer.tokenize(text)
#     return len(tokens) / max(len(words), 1)
# ```
#
# **À vous de jouer.** Complétez `simulate_fragmentation` :
# elle estime le facteur de fragmentation BPE d'une transcription.

# %% [FOURNI]
_MEDIEVAL_HARD = {
    "roys", "palays", "chastelain", "norm~die", "co~te", "pñce",
    "chastel", "roial", "ledit", "susdit", "seellee", "preudomme",
    "vilain", "moys", "chevaliers", "destrier", "pucele",
}

# %% [TODO]
def simulate_fragmentation(text: str,
                            hard_words: set = _MEDIEVAL_HARD) -> float:
    """
    Estime le facteur de fragmentation BPE d'une transcription.

    Le facteur est le ratio :  tokens_BPE_estimés / nombre_de_mots_bruts

    Règles d'estimation (par mot) :
        - Mot dans hard_words ou contenant ~ ou ñ → 2 à 4 tokens (random)
        - Mot de longueur > 8 caractères           → 2 tokens
        - Sinon                                    → 1 token

    Retourne 0.0 si le texte est vide.

    Indice
    ------
    Utilisez random.randint(2, 4) pour les mots problématiques.
    Normalisez par max(len(words), 1) pour éviter la division par zéro.
    """
    words = text.split()
    if not words:
        return 0.0
    tokens_bpe = 0
    for word in words:
        w = word.lower().strip(".,;:")
        # TODO : calculer tokens_bpe pour ce mot selon les règles
        pass   # ← remplacer cette ligne
    return tokens_bpe / len(words)

# %% [markdown]
# **Cellule de validation 6.2**

# %% [FOURNI — validation]
assert simulate_fragmentation("") == 0.0, \
    "simulate_fragmentation('') doit retourner 0.0"
_frag_simple = simulate_fragmentation("le roi de france")
assert 1.0 <= _frag_simple <= 1.5, \
    f"Des mots simples ne devraient pas trop se fragmenter. Obtenu : {_frag_simple:.2f}"
_frag_hard = simulate_fragmentation("li roys de france en son palays")
assert _frag_hard >= _frag_simple, \
    "Un texte avec mots médiévaux doit avoir une fragmentation >= texte simple."
print(f"Validation 6.2 : OK")
print(f"  'le roi de france'                    → {_frag_simple:.2f} tokens/mot")
print(f"  'li roys de france en son palays'      → {_frag_hard:.2f} tokens/mot")

# %% [FOURNI]
df["fragmentation"] = df["transcription"].apply(simulate_fragmentation)
frag_clean  = df[~df["needs_review"]]["fragmentation"].mean()
frag_review = df[df["needs_review"]]["fragmentation"].mean()
print(f"\nFacteur de fragmentation (lignes propres)   : {frag_clean:.2f}")
print(f"Facteur de fragmentation (lignes NR)        : {frag_review:.2f}")
print(f"Surcoût fragmentation needs_review          : +{(frag_review/frag_clean-1)*100:.0f} %")

# %% [markdown]
# **Question 6.2** : Le facteur de fragmentation est-il plus élevé pour
# les lignes `needs_review` ? Après la normalisation orthographique du Jour 2
# (roys→roi, palays→palais…), ce facteur devrait-il diminuer ?
# Comment mesurerez-vous cet impact ?

# %% [markdown]
# ## Partie 7 — Split train/val/test verrouillé
#
# ### 7.1 Split stratifié au niveau document
#
# **À vous de jouer.** Complétez `build_stratified_split` : elle réalise
# un split stratifié par type de document, au niveau du document entier
# (pas de la ligne), et retourne les ensembles d'identifiants.
#
# La stratification au niveau document est obligatoire : si les lignes
# d'un même document se retrouvent dans plusieurs splits, le modèle voit
# la même main de scribe à l'entraînement et en test — la mesure de
# généralisation est alors faussée.

# %% [TODO]
def build_stratified_split(df: pd.DataFrame,
                            train_ratio: float = 0.70,
                            val_ratio:   float = 0.15,
                            test_ratio:  float = 0.15,
                            seed: int    = 42) -> tuple[set, set, set]:
    """
    Construit un split train/val/test stratifié au niveau document.

    Paramètres
    ----------
    df           : DataFrame  corpus aplati
    train_ratio  : float      fraction d'entraînement
    val_ratio    : float      fraction de validation
    test_ratio   : float      fraction de test
    seed         : int        graine aléatoire

    Retourne
    --------
    (train_ids, val_ids, test_ids)  trois ensembles de document_id (set[str])

    Contraintes
    -----------
    - Pas de chevauchement entre les trois ensembles.
    - La stratification est faite par document_type (pas par ligne).
    - Utilisez sklearn.model_selection.GroupShuffleSplit.

    Algorithme
    ----------
    1. Construire doc_df : un DataFrame avec une ligne par document
       (groupby document_id, récupérer document_type).
    2. Premier split : séparer test (test_ratio) du reste (train+val),
       avec groups=doc_df["document_type"].
    3. Deuxième split : séparer val du train+val restant.
       val_size = int(len(doc_trainval) * val_ratio / (train_ratio + val_ratio))
    4. Vérifier l'absence de chevauchement (assert).
    5. Retourner les trois sets.
    """
    doc_df = (df.groupby("document_id")
                .agg(document_type=("document_type", "first"))
                .reset_index())

    # TODO : implémenter les deux splits et retourner les trois ensembles
    pass   # ← remplacer cette ligne

# %% [markdown]
# **Cellule de validation 7.1**

# %% [FOURNI — validation]
_train_ids, _val_ids, _test_ids = build_stratified_split(df)
assert isinstance(_train_ids, set), "train_ids doit être un set."
assert isinstance(_val_ids,   set), "val_ids doit être un set."
assert isinstance(_test_ids,  set), "test_ids doit être un set."
assert not (_train_ids & _val_ids),  "train et val se chevauchent !"
assert not (_train_ids & _test_ids), "train et test se chevauchent !"
assert not (_val_ids   & _test_ids), "val et test se chevauchent !"
all_docs = set(df["document_id"].unique())
assert _train_ids | _val_ids | _test_ids == all_docs, \
    "Tous les documents doivent être assignés à un split."
print(f"Validation 7.1 : OK")
print(f"  Train : {len(_train_ids)} documents")
print(f"  Val   : {len(_val_ids)}   documents")
print(f"  Test  : {len(_test_ids)}  documents")

# %% [FOURNI]
train_ids, val_ids, test_ids = _train_ids, _val_ids, _test_ids
df["split"] = df["document_id"].apply(
    lambda d: "train" if d in train_ids
    else ("val" if d in val_ids else "test")
)
print("\nDistribution du split :")
print(df.groupby(["split", "document_type"]).size().unstack(fill_value=0).to_string())

# %% [markdown]
# ### 7.2 Verrouillage par SHA-256
#
# **À vous de jouer.** Complétez `compute_split_hash` : elle calcule
# un hash SHA-256 déterministe du split courant.
# Ce hash garantit que le split ne change pas entre les expériences.

# %% [TODO]
def compute_split_hash(df: pd.DataFrame) -> str:
    """
    Calcule un hash SHA-256 déterministe du split.

    Le hash est calculé sur la liste **triée** des paires (line_id, split),
    sérialisée en JSON. Le tri garantit que le hash est identique
    quel que soit l'ordre des lignes dans le DataFrame.

    Paramètre
    ---------
    df : DataFrame  doit contenir les colonnes "line_id" et "split"

    Retourne
    --------
    str  hash SHA-256 hexadécimal (64 caractères)

    Algorithme
    ----------
    1. manifest = sorted(zip(df["line_id"], df["split"]))
    2. manifest_str = json.dumps(manifest, ensure_ascii=False, sort_keys=True)
    3. Retourner hashlib.sha256(manifest_str.encode("utf-8")).hexdigest()
    """
    # TODO : implémenter
    pass   # ← remplacer cette ligne

# %% [markdown]
# **Cellule de validation 7.2**

# %% [FOURNI — validation]
_hash1 = compute_split_hash(df)
assert isinstance(_hash1, str) and len(_hash1) == 64, \
    f"Le hash doit être une chaîne hexadécimale de 64 caractères. Obtenu : {_hash1!r}"
# Idempotence : le même DataFrame doit produire le même hash
_hash2 = compute_split_hash(df.sample(frac=1, random_state=99))  # ordre aléatoire
assert _hash1 == _hash2, \
    "Le hash doit être identique quel que soit l'ordre des lignes (utiliser sorted)."
print(f"Validation 7.2 : OK — hash idempotent")
print(f"SHA-256 du split : {_hash1}")

# %% [FOURNI]
# Sauvegarde du manifeste (livrable du TP)
SPLIT_HASH = _hash1
split_manifest = {
    "hash_sha256":   SPLIT_HASH,
    "train_doc_ids": sorted(train_ids),
    "val_doc_ids":   sorted(val_ids),
    "test_doc_ids":  sorted(test_ids),
    "n_train_lines": int((df["split"] == "train").sum()),
    "n_val_lines":   int((df["split"] == "val").sum()),
    "n_test_lines":  int((df["split"] == "test").sum()),
    "ratios":        {"train": 0.70, "val": 0.15, "test": 0.15},
}
with open("split_manifest.json", "w", encoding="utf-8") as f:
    json.dump(split_manifest, f, indent=2, ensure_ascii=False)
print("\nLivrable sauvegardé : split_manifest.json")
print("Ce fichier doit être versionné dans git et ne jamais être modifié.")

# %% [markdown]
# ## Partie 8 — Rapport needs_review et tableau de baseline
#
# ### 8.1 Rapport `needs_review` (livrable du TP)
#
# **À vous de jouer.** Produisez le rapport JSON `needs_review_report.json`,
# en agrégeant les résultats des parties précédentes.

# %% [TODO]
def build_needs_review_report(df: pd.DataFrame,
                               cause_counter: collections.Counter) -> dict:
    """
    Construit le rapport JSON du taux needs_review.

    Retourne
    --------
    dict avec les clés :
        "global_rate"     : float  taux global (entre 0 et 1)
        "by_document_type": dict   {type: {"n_review": int, "n_total": int, "taux": float}}
        "top_causes"      : list   [(cause, count), ...] triées par fréquence décroissante

    Indice
    ------
    - global_rate = df["needs_review"].mean()
    - Pour by_document_type, réutilisez compute_needs_review_stats(df)
      et convertissez le DataFrame en dict.
    - Pour top_causes, utilisez cause_counter.most_common().
    """
    # TODO : construire et retourner le rapport
    pass   # ← remplacer cette ligne

# %% [FOURNI]
nr_report = build_needs_review_report(df, cause_counter)
if nr_report is not None:
    with open("needs_review_report.json", "w", encoding="utf-8") as f:
        json.dump(nr_report, f, indent=2, ensure_ascii=False, default=str)
    print("Livrable sauvegardé : needs_review_report.json")
    print(f"\nTaux global needs_review : {nr_report.get('global_rate', '?')*100:.1f} %")
else:
    print("ATTENTION : build_needs_review_report a retourné None — TODO non complété.")

# %% [markdown]
# ### 8.2 Tableau de baseline
#
# Remplissez ce tableau à partir des résultats du TP.
# Vous devez pouvoir le compléter **sans rouvrir votre notebook**.

# %% [FOURNI]
print("\n" + "="*60)
print("TABLEAU DE BASELINE — À connaître avant le Jour 2")
print("="*60)
metrics = [
    ("Lignes totales",
     str(len(df))),
    ("Lignes needs_review (global)",
     f"{df['needs_review'].mean()*100:.1f} %"),
    ("Taux NR chartes",
     f"{df[df['document_type']=='charte']['needs_review'].mean()*100:.1f} %"),
    ("Taux NR registres",
     f"{df[df['document_type']=='registre']['needs_review'].mean()*100:.1f} %"),
    ("Taux NR romans",
     f"{df[df['document_type']=='roman']['needs_review'].mean()*100:.1f} %"),
    ("Taux NR liturgique",
     f"{df[df['document_type']=='liturgique']['needs_review'].mean()*100:.1f} %"),
    ("Cause principale du flag NR",
     cause_counter.most_common(1)[0][0] if cause_counter else "—"),
    ("Abréviations non résolues (tildes)",
     str(_glob.get("tilde_nasale", 0))),
    ("Couverture table statique",
     f"{_coverage['coverage']*100:.0f} %"),
    ("Fragmentation BPE (lignes propres)",
     f"{df[~df['needs_review']]['fragmentation'].mean():.2f}"),
    ("Fragmentation BPE (needs_review)",
     f"{df[df['needs_review']]['fragmentation'].mean():.2f}"),
    ("Longueur médiane (tokens bruts)",
     f"{df['n_tokens'].median():.0f}"),
    ("P95 longueur (tokens bruts)",
     f"{df['n_tokens'].quantile(0.95):.0f}"),
    ("SHA-256 du split",
     SPLIT_HASH[:16] + "..."),
]
for label, value in metrics:
    print(f"  {label:<42s}: {value}")
print("="*60)

# %% [markdown]
# ## Partie 9 — Pour aller plus loin (optionnel)

# %% [markdown]
# ### 9.1 Analyse des candidats `[a/b]`
#
# Le champ `candidates` liste les positions où le modèle HTR a hésité.
# Analysez la distribution des marges de confiance (score_a - score_b)
# et identifiez les paires de caractères les plus fréquemment confondues.
#
# ```python
# from collections import Counter
# confusion_pairs = Counter()
# for _, row in df.iterrows():
#     for cand in row["candidates"]:
#         if len(cand["alternatives"]) >= 2:
#             pair = tuple(sorted(cand["alternatives"][:2]))
#             confusion_pairs[pair] += 1
#
# print("Paires de caractères les plus confondus :")
# for pair, count in confusion_pairs.most_common(10):
#     print(f"  {pair[0]!r} / {pair[1]!r}  : {count} occurrences")
# ```

# %% [markdown]
# ### 9.2 Validation de la cohérence `needs_review_count` vs lignes flagguées
#
# Le champ `metadata.needs_review_count` devrait correspondre au nombre de
# lignes `needs_review` dans le document. Vérifiez cette cohérence :
#
# ```python
# inconsistencies = []
# for doc in valid_docs:
#     claimed = doc["metadata"]["needs_review_count"]
#     actual  = sum(1 for p in doc["pages"]
#                   for l in p["lines"] if l["needs_review"])
#     if claimed != actual:
#         inconsistencies.append((doc["document_id"], claimed, actual))
#
# if inconsistencies:
#     print(f"INCOHÉRENCES détectées ({len(inconsistencies)}) :")
#     for doc_id, claimed, actual in inconsistencies:
#         print(f"  {doc_id}: metadata={claimed}, réel={actual}")
# else:
#     print("Cohérence needs_review_count : OK")
# ```

# %% [markdown]
# ### 9.3 Couverture OOV réelle avec CamemBERT
#
# En environnement avec GPU et accès réseau, mesurez la fragmentation réelle :
#
# ```python
# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("almanach/camembert-base")
#
# def real_fragmentation(text):
#     words  = text.split()
#     tokens = tokenizer.tokenize(text)
#     return len(tokens) / max(len(words), 1)
#
# df["frag_camembert"] = df["transcription"].apply(real_fragmentation)
# print(f"Fragmentation réelle CamemBERT (propres)   : "
#       f"{df[~df['needs_review']]['frag_camembert'].mean():.2f}")
# print(f"Fragmentation réelle CamemBERT (NR)        : "
#       f"{df[df['needs_review']]['frag_camembert'].mean():.2f}")
# ```

# %% [markdown]
# ---
#
# ## Récapitulatif — Ce que vous avez produit
#
# | Livrable | Fichier | Contenu |
# |---|---|---|
# | Inventaire abréviations | `abbreviations_inventory.json` | Comptages par patron et par type de document |
# | Rapport needs_review | `needs_review_report.json` | Taux par type, causes, top causes |
# | Manifeste de split | `split_manifest.json` | SHA-256, listes de doc_ids par split |
#
# | Fonction implémentée | Usage au Jour 2 |
# |---|---|
# | `validate_corpus` | Réutilisée pour valider les outputs normalisés |
# | `flatten_corpus` | Base de tous les pipelines DataFrame du module |
# | `compute_needs_review_stats` | Mesure de l'impact du nettoyage |
# | `diagnose_line` | Pondération de la loss par type de bruit |
# | `compute_geometric_mean` | Signal pour l'arbitrage CamemBERT MLM |
# | `extract_abbreviations` | Détection des cibles pour le module de règles |
# | `build_abbreviation_inventory` | Priorisation des règles à implémenter |
# | `measure_table_coverage` | Baseline avant le neural (étape 4 Jour 2) |
# | `simulate_fragmentation` | Estimateur du coût BPE pré/post normalisation |
# | `build_stratified_split` | Réutilisé pour tous les fine-tunings |
# | `compute_split_hash` | Verrouillage de reproductibilité |
#
# **Vers le Jour 2 :** vous savez maintenant exactement ce que contient
# votre corpus. Le module de règles que vous implémenterez demain (étape 1)
# doit couvrir en priorité les patrons d'abréviations que vous avez inventoriés
# ici. Le taux de fragmentation BPE que vous avez mesuré sera votre métrique
# de progrès : après normalisation, il doit diminuer.
