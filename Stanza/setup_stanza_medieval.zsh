#!/usr/bin/env zsh
# =============================================================================
# setup_stanza_medieval.zsh
#
# Script d'installation et de préparation pour la démonstration Stanza sur
# manuscrits médiévaux français (corpus CATMuS). Compatible macOS et Linux.
# Utilise `uv` comme gestionnaire de paquets Python.
#
# Ce script NE FAIT QUE :
#   1. Vérifier les prérequis système (uv, Python, espace disque)
#   2. Créer un environnement virtuel et installer les dépendances Python
#   3. Télécharger le modèle Stanza `fro` (et `fr` pour la comparaison NER)
#   4. Vérifier l'accès réseau à huggingface.co (requis pour CATMuS)
#
# Ce script NE FAIT PAS :
#   - L'entraînement du tagger POS (réservé au notebook Colab, GPU requis)
#   - Le clonage du dépôt stanfordnlp/stanza (uniquement nécessaire sur Colab
#     au moment de l'entraînement effectif)
#
# Usage :
#   chmod +x setup_stanza_medieval.zsh
#   ./setup_stanza_medieval.zsh
# =============================================================================

set -e  # arrêt immédiat en cas d'erreur
set -u  # erreur si variable non définie utilisée

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------
PROJECT_DIR="${PWD}/stanza_medieval_demo"
VENV_DIR="${PROJECT_DIR}/.venv"
PYTHON_VERSION="3.11"

# Couleurs pour la lisibilité du terminal (désactivées si non-tty)
if [[ -t 1 ]]; then
    C_RESET=$'\e[0m'
    C_GREEN=$'\e[32m'
    C_YELLOW=$'\e[33m'
    C_RED=$'\e[31m'
    C_BOLD=$'\e[1m'
else
    C_RESET="" ; C_GREEN="" ; C_YELLOW="" ; C_RED="" ; C_BOLD=""
fi

log_info()  { print -P "%F{green}[INFO]%f $1" }
log_warn()  { print -P "%F{yellow}[ATTENTION]%f $1" }
log_error() { print -P "%F{red}[ERREUR]%f $1" }
log_step()  { print -P "\n%B%F{cyan}=== $1 ===%f%b" }


# -----------------------------------------------------------------------
# Étape 1 — Vérification des prérequis système
# -----------------------------------------------------------------------
log_step "Étape 1 — Vérification des prérequis"

if ! command -v uv &> /dev/null; then
    log_error "uv n'est pas installé."
    print "Installez-le avec :"
    print "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    print "ou (macOS avec Homebrew) :"
    print "  brew install uv"
    exit 1
fi
log_info "uv trouvé : $(uv --version)"

# Détection de l'OS pour les avertissements spécifiques
OS_NAME="$(uname -s)"
case "${OS_NAME}" in
    Darwin)
        log_info "Système détecté : macOS"
        ;;
    Linux)
        log_info "Système détecté : Linux"
        ;;
    *)
        log_warn "Système non testé : ${OS_NAME}. Le script peut échouer."
        ;;
esac

# Vérification de l'espace disque disponible (modèles Stanza ~ 500 Mo à 1 Go)
if command -v df &> /dev/null; then
    AVAILABLE_KB=$(df -k "${PWD}" | awk 'NR==2 {print $4}')
    AVAILABLE_GB=$((AVAILABLE_KB / 1024 / 1024))
    if (( AVAILABLE_GB < 2 )); then
        log_warn "Espace disque disponible faible (~${AVAILABLE_GB} Go)."
        log_warn "Les modèles Stanza (fro + fr) nécessitent environ 1 à 1.5 Go."
    else
        log_info "Espace disque disponible : ~${AVAILABLE_GB} Go (suffisant)"
    fi
fi


# -----------------------------------------------------------------------
# Étape 2 — Vérification de l'accès réseau à huggingface.co
# -----------------------------------------------------------------------
log_step "Étape 2 — Vérification de l'accès réseau"

log_info "Test de connectivité vers huggingface.co (requis pour CATMuS)..."
if curl -sI --max-time 10 https://huggingface.co > /dev/null 2>&1; then
    log_info "huggingface.co accessible."
else
    log_warn "huggingface.co semble inaccessible depuis ce réseau."
    log_warn "Le téléchargement du corpus CATMuS échouera ; le script Python"
    log_warn "principal basculera automatiquement sur son corpus de secours"
    log_warn "hors-ligne (FALLBACK_CORPUS), mais ce n'est qu'un palliatif —"
    log_warn "vérifiez votre proxy ou pare-feu si vous voulez le vrai corpus."
fi

log_info "Test de connectivité vers raw.githubusercontent.com (modèles Stanza)..."
if curl -sI --max-time 10 https://raw.githubusercontent.com > /dev/null 2>&1; then
    log_info "raw.githubusercontent.com accessible."
else
    log_error "raw.githubusercontent.com inaccessible : Stanza ne pourra pas"
    log_error "télécharger la liste des modèles disponibles. Vérifiez votre réseau."
fi


# -----------------------------------------------------------------------
# Étape 3 — Création de l'environnement virtuel et installation
# -----------------------------------------------------------------------
log_step "Étape 3 — Environnement virtuel et dépendances"

mkdir -p "${PROJECT_DIR}"
cd "${PROJECT_DIR}"

log_info "Création de l'environnement virtuel..."
# On utilise le Python système disponible plutôt que d'en télécharger un
# nouveau via uv (plus rapide, évite une dépendance réseau supplémentaire).
# uv détecte automatiquement une version compatible (>= 3.9 recommandé).
if command -v python3 &> /dev/null; then
    log_info "Python système détecté : $(python3 --version)"
    uv venv "${VENV_DIR}" --python "$(command -v python3)"
else
    log_warn "Aucun python3 système détecté, uv va en télécharger un (requiert un accès réseau à GitHub)."
    uv venv --python "${PYTHON_VERSION}" "${VENV_DIR}"
fi

log_info "Installation de PyTorch (version CPU)..."
log_info "  Cette démonstration n'a besoin que du CPU localement (le fine-tuning"
log_info "  effectif se fait sur Colab avec GPU). Installer la version CPU-only"
log_info "  évite de télécharger plusieurs Go de dépendances CUDA inutiles."
uv pip install --python "${VENV_DIR}/bin/python" \
    torch --index-url https://download.pytorch.org/whl/cpu

log_info "Installation des dépendances avec uv pip..."
# Note : --break-system-packages n'est pas nécessaire avec uv (environnement
# isolé par construction), contrairement à un pip système.
uv pip install --python "${VENV_DIR}/bin/python" \
    stanza \
    datasets \
    huggingface_hub

log_info "Dépendances installées :"
uv pip list --python "${VENV_DIR}/bin/python" | grep -E "stanza|datasets|huggingface"


# -----------------------------------------------------------------------
# Étape 4 — Téléchargement des modèles Stanza
# -----------------------------------------------------------------------
log_step "Étape 4 — Téléchargement des modèles Stanza (fro et fr)"

log_info "Téléchargement du modèle 'fro' (ancien français)..."
log_info "  Processeurs disponibles pour fro : tokenize, pos, lemma, depparse"
log_info "  (PAS de mwt, PAS de ner — limitation officielle de ce modèle)"
"${VENV_DIR}/bin/python" -c "
import stanza
stanza.download('fro', verbose=True)
print('Modèle fro téléchargé avec succès.')
"

log_info "Téléchargement du modèle 'fr' (français moderne, pour comparaison NER)..."
"${VENV_DIR}/bin/python" -c "
import stanza
stanza.download('fr', verbose=True)
print('Modèle fr téléchargé avec succès.')
"


# -----------------------------------------------------------------------
# Étape 5 — Vérification finale
# -----------------------------------------------------------------------
log_step "Étape 5 — Vérification de l'installation"

"${VENV_DIR}/bin/python" -c "
import stanza
print('Test du pipeline fro (tokenize, pos, lemma, depparse)...')
nlp = stanza.Pipeline(lang='fro', processors='tokenize,pos,lemma,depparse', verbose=False)
doc = nlp('Artus li rois fu mout vaillanz.')
for sent in doc.sentences:
    for word in sent.words:
        print(f'  {word.text:<12} upos={word.upos:<6} lemma={word.lemma}')
print('Pipeline fro opérationnel.')
"

log_step "Installation terminée"
print ""
log_info "Pour activer l'environnement virtuel manuellement :"
print "  source ${VENV_DIR}/bin/activate"
print ""
log_info "Pour lancer la démonstration complète :"
print "  ${VENV_DIR}/bin/python stanza_medieval_demo.py"
print ""
log_info "Pour lancer la démonstration sans réseau (corpus de secours) :"
print "  ${VENV_DIR}/bin/python stanza_medieval_demo.py --offline"
print ""
log_warn "Rappel : l'entraînement effectif du fine-tuning (Partie 2) doit être"
log_warn "exécuté sur le notebook Colab fourni séparément (GPU T4 requis)."
