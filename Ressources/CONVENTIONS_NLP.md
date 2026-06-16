# Conventions NLP -- Consommation du dataset CV manuscrits

**Version : 1.0 | Date : Juin 2026 | Référence : DATA_CONTRACT_SCHEMA v1.0**
**Document compagnon de : CONVENTIONS_TRANSCRIPTION.md (pipeline CV)**

---

Ce document est destiné à l'équipe du module NLP qui reçoit le dataset
produit par le pipeline Computer Vision. Il décrit comment interpréter les
données livrées, quels traitements sont attendus côté NLP, et quels pièges
éviter lors de la consommation des transcriptions.

Lire **obligatoirement** `CONVENTIONS_TRANSCRIPTION.md` avant ce document :
il définit l'encodage exact de chaque convention dans le champ `transcription`.

---

## 1. Vue d'ensemble : ce que le module NLP reçoit

Le pipeline CV livre un `DatasetDict` HuggingFace avec quatre splits :

| Split | Contenu | Usage NLP recommandé |
|-------|---------|---------------------|
| `train` | Lignes fiables, `confidence >= 0.7` | Entraînement des modèles NLP |
| `test` | Lignes fiables, 20% réservées (seed=42) | Evaluation finale uniquement |
| `validation` | Confiance intermédiaire `[0.5, 0.7[` | Ajustement des hyperparamètres |
| `needs_review` | Lignes flaggées, non vérifiées | Ne jamais utiliser sans vérification humaine |

Chaque exemple contient les champs `page_id`, `line_id`, `text` (la
transcription), `confidence`, `needs_review`, `langue`, `flags`,
`source_humaine`, `pipeline` et `dataset_version`.

Les fichiers JSON individuels (un par page) contiennent en plus les
coordonnées de baseline, les sources brutes par modèle, et les
descriptions d'enluminures.

---

## 2. Interprétation des marqueurs dans `text`

Le champ `text` contient une transcription semi-diplomatique encodée
selon les conventions de `CONVENTIONS_TRANSCRIPTION.md`. Voici comment
le module NLP doit interpréter chaque marqueur.

### 2.1 Abréviations

| Motif dans `text` | Signification | Action NLP attendue |
|-------------------|---------------|---------------------|
| Texte direct (ex. `dominus`) | Abréviation développée de façon certaine par le pipeline CV | Aucune : accepter tel quel |
| `(mot)` | Développement incertain d'une abréviation | Valider ou corriger par modèle de langue ; retirer les parenthèses après validation |
| `[abr]` | Signe abréviatif non identifié par le pipeline CV | Tenter une résolution par contexte ; conserver `[abr]` si échec |
| `⁊` (U+204A) | Note tironienne pour *et* | Normaliser en `et` lors de la normalisation orthographique |
| `ꝑ` (U+A751) | P barré (*per-* ou *par-*) | Normaliser selon le contexte linguistique |

**Piège fréquent :** les parenthèses `(...)` dans le champ `text` ne sont
jamais des parenthèses du texte original. Elles signalent toujours une
incertitude éditoriale du pipeline CV. Le module NLP doit les traiter comme
telles et ne pas les confondre avec de la ponctuation.

### 2.2 Lacunes et zones illisibles

| Motif | Signification | Action NLP |
|-------|---------------|------------|
| `[?]` | Un caractère illisible (nombre connu) | Tenter une prédiction par modèle de langue ; laisser `[?]` si incertain |
| `[...]` | Zone illisible de longueur inconnue | Ne pas tenter de complétion automatique sans contexte fort |
| `[†]` | Lacune physique (parchemin détruit) | Conserver tel quel ; ne pas tenter de reconstruction |
| `[mot?]` | Lecture proposée mais incertaine | Valider par modèle de langue ; retirer le `?` si confirmé |

### 2.3 Eléments structurels

| Motif | Signification | Action NLP |
|-------|---------------|------------|
| `<R>...</R>` | Rubrique (texte à l'encre rouge) | Retirer les balises pour le traitement textuel ; conserver l'information dans les métadonnées si utile pour l'annotation |
| `\|\|` | Fin de colonne | Traiter comme un séparateur de segment, pas comme un caractère du texte |

### 2.4 Caractères médiévaux

Les caractères médiévaux Unicode (MUFI) sont présents tels quels dans le
champ `text`. Le module NLP doit être capable de les lire en UTF-8 et de
décider, selon ses objectifs, s'il les normalise ou les conserve :

| Caractère | Unicode | Normalisation suggérée |
|-----------|---------|----------------------|
| ȝ (yogh) | U+021D | Conserver ou normaliser en `g` / `y` selon le contexte |
| ꝑ (p barré) | U+A751 | Normaliser en `per` ou `par` selon le contexte |
| ⁊ (tironien) | U+204A | Normaliser en `et` |
| æ | U+00E6 | Conserver (ligature standard) |
| œ | U+0153 | Conserver (ligature standard) |

---

## 3. Utilisation des scores de confiance

### 3.1 Interprétation de `confidence`

Le score `confidence` dans le dataset est **calibré** : il a subi une
transformation puissance (x^1.3) qui compresse les scores élevés pour les
rendre plus conservateurs. Il ne correspond pas directement à la
probabilité CTC ou softmax brute des modèles.

Interprétation pratique :

| Plage | Qualité attendue | Stratégie NLP |
|-------|-----------------|---------------|
| `[0.85, 1.0]` | Très fiable | Utiliser directement pour l'entraînement |
| `[0.70, 0.85[` | Fiable | Utiliser pour l'entraînement avec poids normal |
| `[0.50, 0.70[` | Incertain | Utiliser pour la validation, pas pour l'entraînement |
| `[0.0, 0.50[` | Peu fiable | Ne pas utiliser sans vérification humaine |

### 3.2 Pondération par confiance

Pour l'entraînement de modèles NLP (normalisation orthographique,
correction contextuelle), il est recommandé de pondérer les exemples
par leur score de confiance :

```python
# Pseudo-code — pondération par confiance dans la loss
loss = criterion(prediction, target)
weighted_loss = loss * example["confidence"]
```

Cette pondération réduit l'influence des lignes bruitées sans les
éliminer entièrement, ce qui est préférable à un seuil dur.

### 3.3 Comparaison des sources (mode hybride)

Quand `pipeline == "hybride_vote"`, les fichiers JSON contiennent les
transcriptions brutes de chaque modèle dans `lignes[*].sources` :

```json
"sources": {
    "kraken": {"transcription": "...", "confiance": 0.82},
    "trocr":  {"transcription": "...", "confiance": 0.91}
}
```

Le module NLP peut exploiter cette information :
- Si les deux modèles concordent, la transcription est très probablement correcte.
- Si les deux modèles divergent, les différences signalent les zones ambiguës
  que le modèle de langue NLP peut arbitrer.

---

## 4. Gestion des langues mélangées

### 4.1 Le champ `langue`

Le champ `langue` indique la langue **dominante** détectée pour la ligne :
`"fr"` (ancien/moyen français), `"la"` (latin), ou `"inconnu"`.

Ce champ est une indication approximative, pas une annotation linguistique
fine. Une ligne marquée `"fr"` peut contenir des segments latins
(citations, formules liturgiques, gloses) et inversement.

### 4.2 Responsabilité du module NLP

La désambiguïsation linguistique intra-ligne est la responsabilité du
module NLP, pas du pipeline CV. Le pipeline CV fournit la langue dominante
comme point de départ ; le module NLP doit :

1. Segmenter la ligne en spans linguistiques si nécessaire.
2. Appliquer les règles de normalisation propres à chaque langue
   (le vieux français et le latin médiéval ont des conventions
   orthographiques différentes).
3. Enrichir le champ `langue` en annotation plus fine si le format
   de sortie NLP le permet.

### 4.3 Cas particulier : formules mixtes

Certaines constructions sont intrinsèquement bilingues et ne doivent pas
être scindées artificiellement :

- Les incipits latins suivis de texte français sur la même ligne.
- Les citations latines enchâssées dans une phrase française avec `inquit`
  ou équivalent.
- Les rubriques en latin sur des pages en français (et inversement).

---

## 5. Traitement des flags et des lignes `needs_review`

### 5.1 Signification des flags

| Flag | Cause | Impact NLP |
|------|-------|------------|
| `CONFIANCE_BASSE` | Score calibré < 0.5 | La transcription est probablement erronée ; ne pas utiliser pour l'entraînement |
| `DISCORDANCE_MODELES` | Kraken et TrOCR divergent fortement (edit distance > 50%) | La zone est ambiguë ; le modèle de langue NLP peut tenter un arbitrage |
| `LIGNE_COURTE` | Moins de 3 caractères | Peut être un artefact de segmentation ; vérifier si la ligne est réelle |
| `IMAGE_VIDE` | Transcription vide dans une zone saturée (enluminure) | Pas de texte : ignorer pour le traitement linguistique |

### 5.2 Stratégie recommandée pour `needs_review`

Le split `needs_review` contient des lignes dont la qualité n'est pas
garantie. Trois stratégies possibles :

1. **Ignorer entièrement** (par défaut) -- n'utiliser que `train`, `test`
   et `validation` pour les tâches NLP.
2. **Correction automatique** -- passer les lignes `needs_review` dans un
   pipeline de correction par modèle de langue entraîné sur `train`, puis
   réintégrer les lignes corrigées avec une confiance ajustée.
3. **Correction humaine** -- exporter les lignes `needs_review` vers un
   outil d'annotation (ex. INCEpTION, Prodigy) pour validation manuelle.

---

## 6. Tokenisation des manuscrits médiévaux

### 6.1 Particularités orthographiques

Les transcriptions semi-diplomatiques conservent l'orthographe du scribe.
Le module NLP doit s'attendre à :

- Absence de normalisation orthographique (`liure` pour `livre`, `escrist`
  pour `écrit`, `esperit` pour `esprit`).
- Soudure ou séparation variable des mots (`aucuns` / `au cuns`,
  `cest` / `c'est`).
- Absence quasi-totale d'accents (pas d'accentuation systématique avant
  le XVIe siècle).
- Ponctuation minimaliste ou absente (le *punctus* médiéval n'a pas la
  même valeur que le point moderne).

### 6.2 Recommandations de tokenisation

Pour un tokenizer BPE ou WordPiece entraîné sur du français moderne,
les formes médiévales seront systématiquement hors-vocabulaire. Deux
approches possibles :

1. **Entraîner un tokenizer spécifique** sur le corpus médiéval du split
   `train`. C'est l'approche recommandée si le corpus est suffisamment
   grand (> 50 000 lignes).
2. **Utiliser un tokenizer caractère par caractère** pour les tâches de
   normalisation orthographique (le modèle apprend la transformation
   caractère par caractère de l'ancien français vers le français moderne
   ou vers une forme normalisée).

### 6.3 Gestion des marqueurs dans la tokenisation

Les marqueurs spéciaux (`<R>`, `</R>`, `||`, `[...]`, `[?]`, `[†]`,
`(mot)`) doivent être traités comme des tokens atomiques et ne pas être
scindés par le tokenizer. Ajouter ces motifs comme tokens spéciaux :

```python
special_tokens = [
    "<R>", "</R>",      # Rubriques
    "||",               # Fin de colonne
    "[...]",            # Lacune inconnue
    "[?]",              # Caractère illisible
    "[†]",              # Lacune physique
    "[abr]",            # Abréviation non résolue
]
tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
```

---

## 7. Utilisation des descriptions d'enluminures (LLaVA)

Quand le pipeline modulaire ou hybride est utilisé, les fichiers JSON
contiennent un champ optionnel `illustrations` :

```json
"illustrations": [
    {
        "region_id": "r0002",
        "region_type": "illumination",
        "description": "A half-page miniature depicting Saint John...",
        "bbox": [95, 48, 520, 330]
    }
]
```

Ces descriptions sont générées par LLaVA en anglais et décrivent le
contenu iconographique des enluminures. Le module NLP peut les exploiter
pour :

- Contextualiser le texte environnant (une miniature de l'Annonciation
  précède souvent un texte marial).
- Enrichir les métadonnées pour la recherche plein texte.
- Construire un index iconographique du manuscrit.

Les descriptions sont de qualité variable ; elles n'ont pas été validées
par un historien de l'art. Les traiter comme des suggestions, pas comme
des annotations de référence.

---

## 8. Ce que le module NLP ne doit PAS faire

Pour éviter les doublons de traitement et les incohérences :

| Action | Responsable | Ne pas refaire côté NLP |
|--------|------------|------------------------|
| Binarisation, deskewing, nettoyage d'image | Pipeline CV (section 4.1) | Ne pas retraiter les images |
| Segmentation de layout (SAM) | Pipeline CV (section 4.2) | Ne pas redécouper les pages |
| Détection de baselines (BLLA) | Pipeline CV (section 4.2) | Ne pas resegmenter les lignes |
| Transcription brute (Kraken/TrOCR) | Pipeline CV (section 4.2) | Ne pas relancer la HTR |
| Calibration des scores de confiance | Pipeline CV (section 4.3) | Utiliser `confidence` tel quel |
| Alignement NW et vote pondéré | Pipeline CV (section 4.3) | Ne pas réagréger les sources |

Le rôle du module NLP commence **après** la transcription :
normalisation, correction contextuelle, annotation linguistique.

---

## 9. Format de sortie attendu du module NLP

Pour assurer la traçabilité entre le pipeline CV et le module NLP, il est
recommandé que le module NLP produise un format de sortie qui référence
les identifiants du pipeline CV :

```json
{
    "page_id": "bnf_fr768_f042r",
    "line_id": "l0004",
    "transcription_cv": "du saint esperit ⁊ par le (com)mandement",
    "transcription_normalisee": "du saint esprit et par le commandement",
    "corrections_appliquees": [
        {"position": 15, "avant": "⁊", "apres": "et", "type": "normalisation_tironien"},
        {"position": 31, "avant": "(com)", "apres": "com", "type": "validation_abreviation"},
        {"position": 10, "avant": "esperit", "apres": "esprit", "type": "normalisation_orthographique"}
    ],
    "langue_detectee": ["fr"],
    "annotations_morpho": "..."
}
```

Cette structure permet de remonter de la sortie NLP à la transcription
CV originale, et de mesurer précisément l'impact de chaque étape de
normalisation.

---

## 10. Points de vigilance

1. **Ne jamais mélanger les versions du dataset.** Vérifier le champ
   `dataset_version` de chaque exemple. Un modèle entraîné sur la v1.0
   ne doit pas être évalué sur la v2.0 sans recalibration.

2. **Le champ `text` est la transcription finale du pipeline CV, pas la
   transcription brute d'un modèle unique.** En mode hybride, c'est le
   résultat du vote pondéré. En mode kraken, c'est la sortie Kraken
   directe. Le champ `pipeline` du fichier JSON indique la source.

3. **Le split `test` est sacré.** Ne jamais l'utiliser pendant
   l'entraînement ou l'ajustement d'hyperparamètres. Il est fixé par
   seed=42 pour la reproductibilité.

4. **Les coordonnées de baseline ne sont pas des coordonnées de mots.**
   Elles décrivent la position de la ligne entière dans l'image, pas
   la position individuelle de chaque mot ou caractère. Le pipeline CV
   ne fournit pas de segmentation intra-ligne.

5. **Les descriptions LLaVA sont en anglais.** Elles ne sont pas dans
   la même langue que les transcriptions (ancien français / latin).
   C'est intentionnel : LLaVA est plus fiable en anglais pour les
   descriptions iconographiques.

---

*Document rédigé pour le module NLP -- Promotion MD5 -- Juin 2026*
*Référence croisée : CONVENTIONS_TRANSCRIPTION.md (pipeline CV), DATA_CONTRACT_SCHEMA v1.0*
