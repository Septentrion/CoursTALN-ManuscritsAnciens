# Démonstration Stanza sur manuscrits médiévaux français (corpus CATMuS)

Ce dossier contient une démonstration complète des fonctionnalités de
Stanza appliquées à l'ancien français (modèle `fro`), à partir d'un
échantillon du corpus CATMuS/medieval (HuggingFace datasets). Vous trouverez deux sous-dossiers :
1. `taln`, qui libntre comment réaliser les différentes étapes du traitement linguistique d'un texte
2. `fin-tuning`, qui contient une démonstration de fine-tuning du **tagger POS/morphologique**.

## Fichiers fournis

| Fichier | Rôle |
|---|---|
| `taln/stanza_medieval_demo.py` | Script Python complet : pipeline Stanza (Partie 1) + préparation des données de fine-tuning (Partie 2, sans entraînement effectif) |
| `taln/morph_features.py` | Liste des caractéristiques morphologiques reconnues par Stanza |
| `fine-tuning/stanza_medieval_colab.ipynb` | Notebook Colab : **entraînement effectif** du tagger POS (GPU requis) |
| `setup_stanza_medieval.zsh` | Script d'installation locale (macOS/Linux) avec `uv` |
| `pyproject.toml` | Le fichier des dépendances du projet |
| `README.md` | Ce document |

## Prérequis d'installation locale (macOS / Linux)

### 1. Gestionnaire de paquets `uv`

Si `uv` n'est pas déjà installé :

```zsh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

ou, sur macOS avec Homebrew :

```zsh
brew install uv
```

Vérifier l'installation :

```zsh
uv --version
```

### 2. Python

`uv` peut gérer sa propre version de Python, mais le script
`setup_stanza_medieval.zsh` fourni privilégie le Python système s'il est
disponible (≥ 3.9 recommandé, testé avec 3.11 et 3.12) pour éviter un
téléchargement réseau supplémentaire. Vérifiez votre version :

```zsh
python3 --version
```

### 3. Espace disque

Prévoir environ **1,5 à 2 Go** d'espace disque libre :
- PyTorch (CPU-only) : ~200 Mo
- Modèle Stanza `fro` : quelques dizaines de Mo
- Modèle Stanza `fr` (pour la comparaison NER) : ~500 Mo à 1 Go (inclut le NER neuronal et les CharLM)

### 4. Accès réseau

Deux domaines doivent être accessibles :
- `huggingface.co` — téléchargement du corpus CATMuS et des modèles Stanza les plus récents
- `raw.githubusercontent.com` — fichier de résolution des ressources Stanza (`resources.json`)

Si vous êtes derrière un proxy d'entreprise, vérifiez que ces domaines
sont autorisés avant de lancer le script.

## Installation automatisée

```zsh
chmod +x setup_stanza_medieval.zsh
./setup_stanza_medieval.zsh
```

Ce script :
1. Vérifie les prérequis (uv, OS, espace disque)
2. Teste l'accès réseau aux domaines requis
3. Crée un environnement virtuel isolé (`.venv` dans `stanza_medieval_demo/`)
4. Installe PyTorch en version **CPU-only** (la démonstration locale n'a
   pas besoin de GPU ; le fine-tuning effectif se fait sur Colab)
5. Installe `stanza`, `datasets`, `huggingface_hub`
6. Télécharge les modèles Stanza `fro` et `fr`
7. Lance un test de vérification du pipeline

## Installation manuelle (si vous préférez ne pas utiliser le script)

```zsh
# Créer l'environnement virtuel
uv venv stanza_medieval_demo/.venv --python "$(command -v python3)"

# Installer PyTorch CPU-only (évite de télécharger les dépendances CUDA,
# inutiles pour cette démonstration en local)
uv pip install --python stanza_medieval_demo/.venv/bin/python \
    torch --index-url https://download.pytorch.org/whl/cpu

# Installer les autres dépendances
uv pip install --python stanza_medieval_demo/.venv/bin/python \
    stanza datasets huggingface_hub

# Télécharger les modèles Stanza
stanza_medieval_demo/.venv/bin/python -c "
import stanza
stanza.download('fro')
stanza.download('fr')
"
```

### Si vous utilisez `uv add` plutôt que `uv pip install`

Le script et les instructions ci-dessus utilisent `uv pip install`, qui
ne crée pas de `pyproject.toml`. Si vous préférez gérer ce projet comme
un projet `uv` à part entière (`uv init` + `uv add`), assurez-vous que
le nom du projet dans `pyproject.toml` n'est **pas** `stanza` :

```
error: Requirement name `stanza` matches project name `stanza`, but
self-dependencies are not permitted...
```

`uv` interprète alors `uv add stanza` comme une tentative du projet de
se dépendre lui-même. Renommez le projet (champ `name` dans
`pyproject.toml`, ou `uv init --name stanza-medieval-demo` dès le
départ) avant d'ajouter la dépendance `stanza`.

Par ailleurs, `uv` ne prend pas en compte l'option `--index-url`. Il est donc nécessaire de modifier manuellement le fichier `pyproject.toml`.

Pour ces raisons, `pip`est peut-être une solution plus robuste. `uv` tend néanmoins à devenir une alternative de plus en plus utilisée. Le fichier `pyproject.toml` fourni contient déjà toutes les dépendances dont vous aurez besoin.

## Exécution de la démonstration

### Partie 1 et 2 (préparation des données) — en local

```zsh
# version pip
source stanza_medieval_demo/.venv/bin/activate
python stanza_medieval_demo.py

# version uv
# --> utilise l'exécutable Python del'environnement virtuel
uv run python stanza_medieval_demo.py
```

Pour exécuter sans dépendre du réseau (utilise un corpus de secours
hors-ligne au lieu de télécharger CATMuS) :

```zsh
python stanza_medieval_demo.py --offline
```

### Partie 2 (fine-tuning effectif) — sur Google Colab

L'entraînement effectif du tagger POS **nécessite un GPU** et le
clonage du dépôt source `stanfordnlp/stanza` (Stanza n'expose aucune
API d'entraînement via `stanza.Pipeline`). Cette étape est donc réservée
au notebook `stanza_medieval_colab_*.ipynb` :

1. Ouvrir [Google Colab](https://colab.research.google.com/)
2. Importer `stanza_medieval_colab_*.ipynb` (*Fichier* → *Importer un notebook*)
3. *Exécution* → *Modifier le type d'exécution* → sélectionner **GPU (T4)**
4. Exécuter les cellules dans l'ordre

## Limitations connues de Stanza pour l'ancien français (`fro`)

Ces limitations sont structurelles à Stanza et expliquent certains choix
de cette démonstration :

- **Pas de NER officiel pour `fro`** — seules certaines langues (dont
  l'anglais et le français moderne) disposent d'un modèle NER pré-entraîné.
  La démonstration contourne cette limite par un gazetier minimal
  (PER/LOC) appliqué aux tokens étiquetés `PROPN`, et propose une
  comparaison avec le modèle `fr` qui dispose d'un NER complet.
- **Pas de MWT (Multi-Word Token) pour `fro`** — contrairement au
  français moderne, où des contractions comme *du* sont éclatées en
  *de* + *le*.
- **Pas de module de cohérence textuelle** dans Stanza, quelle que soit
  la langue — cette notion linguistique de haut niveau dépasse le
  périmètre de la bibliothèque. La démonstration illustre à la place les
  relations de dépendances inter-phrases réellement produites par Stanza
  (analyse lexicale des racines/sujets), en précisant explicitement la
  différence avec la cohérence.
- **Pas d'entraînement via l'API Python** — `stanza.Pipeline` ne permet
  que l'inférence. Tout entraînement (tokenizer, MWT, tagger POS,
  lemmatiseur, parser de dépendances, NER) nécessite de cloner le dépôt
  source et d'invoquer les scripts `stanza.utils.training.run_*` en
  ligne de commande, comme démontré dans le notebook Colab.

## Note sur le corpus CATMuS et la colonne `language`

La documentation publique du dataset CATMuS/medieval ne liste pas
exhaustivement les valeurs exactes de la colonne `language`. Le filtre
utilisé dans cette démonstration (`is_french_medieval()`) teste plusieurs
motifs plausibles de façon insensible à la casse (`"old french"`,
`"middle french"`, `"french"`, etc.). Si le filtre ne retourne aucune
ligne lors de votre exécution, une cellule de diagnostic est fournie
dans le notebook Colab (section 1.1) pour lister les valeurs réellement
présentes dans le dataset et ajuster le filtre en conséquence.

>  **N.B.** Vous trouverez deux versions du fine-tuning:
> 1. sur un jeu de données jouet qui illustre le processus
> 2. sur un téléchargement du jeu de données CATMuS, qui donne une vision plus réaliste de ce qui est possible.

## Crédits et licences

- **Stanza** : Apache License 2.0 — [stanfordnlp/stanza](https://github.com/stanfordnlp/stanza)
- **CATMuS/medieval** : licence CC-BY-4.0 — [huggingface.co/datasets/CATMuS/medieval](https://huggingface.co/datasets/CATMuS/medieval), résultat de la collaboration des projets CREMMA, GalliCorpora, HTRomance et DEEDS
- **uv** : Apache License 2.0 / MIT — [astral-sh/uv](https://github.com/astral-sh/uv)
