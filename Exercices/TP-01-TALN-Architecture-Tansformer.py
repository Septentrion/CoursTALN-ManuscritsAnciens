"""
TP Guidé — Transformer miniature pour le Masked Language Modeling
Module NLP · Master Data/IA · MD5 Volet 2 · 2026
──────────────────────────────────────────────────────────────────
VERSION GPU

Différences avec la version CPU :
  • Modèle significativement plus grand (D_MODEL=128, N_HEADS=8,
    D_FF=512, N_LAYERS=4, SEQ_LEN=64) pour saturer le GPU.
  • Mixed precision float16 activée (gain mémoire et vitesse ×1.5–2).
  • Compilation XLA via jit_compile=True (gain ×1.2–1.5 après warmup).
  • Corpus étendu : CORPUS * 200 au lieu de * 20.
  • Batch size 64, 40 epochs avec early stopping sur val_loss.
  • tf.data.Dataset avec prefetch et cache pour un pipeline sans goulot.
  • Vérification explicite du GPU au démarrage.

Objectif : construire de A à Z, sur un corpus de moyen français,
la chaîne complète :

  Texte brut → BPE de poche → Encodage positionnel sinusoïdal
  → Transformer Encoder (Keras) → tête MLM → prédiction du token masqué

Durée estimée : 2 h 30
Matériel requis : GPU (Colab T4/A100 ou cluster).
Backend : TensorFlow 2.x / Keras 3

Instructions générales
──────────────────────
Les cellules marquées  # TODO  contiennent des squelettes à compléter.
Ne modifiez pas les signatures de fonctions ni les noms de variables :
les cellules de validation s'appuient dessus.
Les cellules marquées  # FOURNI  sont à exécuter telles quelles.
"""

# %% [markdown]
# # TP — Transformer miniature & Masked Language Modeling (version GPU)
#
# Ce TP vous fait implémenter les quatre briques fondamentales du cours :
#
# 1. **Tokenisation BPE** : construire un vocabulaire par fusion de paires
# 2. **Encodage positionnel sinusoïdal** : injecter l'information de position
# 3. **Bloc Transformer Encoder** : self-attention + FFN + résidus + LayerNorm
# 4. **Masked Language Modeling** : pré-entraînement style BERT sur corpus médiéval
#
# Cette version GPU diffère de la version CPU sur quatre points techniques,
# détaillés dans la Partie 0 : mixed precision, compilation XLA, pipeline
# `tf.data`, et hyperparamètres agrandis pour saturer le GPU.
# Le code pédagogique (TODO) reste strictement identique.
#
# À la fin, vous visualiserez les poids d'attention de votre modèle.

# %% [markdown]
# ## Partie 0 — Imports et configuration GPU
#
# Quatre adaptations GPU introduites ici, dans l'ordre d'impact décroissant :
#
# **1. Mixed precision float16**
# Par défaut, les tenseurs flottent en float32 (32 bits par valeur).
# En activant la mixed precision, les activations et les gradients
# transitent en float16 (moitié moins de mémoire, opérations matricielles
# 2× plus rapides sur les Tensor Cores NVIDIA A100/V100/T4),
# tandis que l'optimiseur conserve ses *master weights* en float32 pour
# la stabilité numérique. Keras gère cette distinction automatiquement.
#
# **2. Compilation XLA (jit_compile)**
# XLA (*Accelerated Linear Algebra*) est le compilateur de graphes de TensorFlow.
# Passer `jit_compile=True` à `model.compile` déclenche une fusion de kernels
# GPU : plusieurs opérations élémentaires (multiplication, addition, activation)
# sont compilées en un seul kernel GPU, réduisant les allers-retours mémoire.
# Le premier batch déclenche une compilation (~30 s) ; les suivants sont
# accélérés d'un facteur 1.2 à 1.5.
#
# **3. Pipeline tf.data**
# Un simple `model.fit(X, Y)` charge tout le dataset en mémoire GPU à chaque
# epoch. `tf.data.Dataset` avec `.cache()` et `.prefetch()` permet au CPU de
# préparer le batch suivant pendant que le GPU traite le batch courant,
# éliminant le goulot d'alimentation (*data starvation*).
#
# **4. Hyperparamètres agrandis**
# Un modèle de 32 dimensions sur 300 exemples ne sature pas un GPU :
# les transferts mémoire CPU→GPU dominent le temps de calcul.
# On passe à D_MODEL=128, N_HEADS=8, D_FF=512, N_LAYERS=4, SEQ_LEN=64,
# batch_size=64 — soit un modèle ~60× plus grand (~1.5 M paramètres)
# qui justifie l'utilisation du matériel.

# %% [FOURNI]
import os, math, random, collections, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"     # silence les logs TF
os.environ["KERAS_BACKEND"]        = "tensorflow"
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
import keras
from keras import layers, ops
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Vérification GPU ──────────────────────────────────────────────────────
gpus = tf.config.list_physical_devices("GPU")
if not gpus:
    raise RuntimeError(
        "Aucun GPU détecté. Vérifiez votre environnement (Colab : Exécution → "
        "Modifier le type d'exécution → GPU) ou utilisez la version CPU du TP."
    )
print(f"{len(gpus)} GPU(s) disponible(s) :")
for g in gpus:
    print(f"  {g.name}")

# ── Mixed precision float16 ───────────────────────────────────────────────
keras.mixed_precision.set_global_policy("mixed_float16")
print(f"Politique de précision : {keras.mixed_precision.global_policy().name}")
# Les couches Dense et MultiHeadAttention utilisent désormais float16
# pour leurs calculs, mais stockent les poids en float32.

# ── Reproductibilité ──────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
keras.utils.set_random_seed(SEED)

print(f"Keras {keras.__version__} — backend : {keras.backend.backend()}")

# %% [markdown]
# ## Partie 1 — Corpus et Byte Pair Encoding (BPE)
#
# ### 1.1 Le corpus
#
# Nous travaillons sur un corpus jouet de 15 phrases en moyen français.
# En TP réel (Jour 1), vous le remplacerez par votre sous-corpus CREMMA.

# %% [FOURNI]
CORPUS = [
    "li roys de france signa l acte en son palays",
    "le conte de champagne et le duc de bourgogne firent accord",
    "en l an de grace mil trois cent la reine trespassa",
    "li chevaliers arma son destrier et prist la lance",
    "le bailli du roi rendit son jugement au peuple",
    "la chartre fut seellee du seel du duc de normandie",
    "messire guillaumes de villehardouin escrivit la cronique",
    "au moys de mars li roys tint sa court pleniere",
    "la dame du chastel fit don de terres a l eglise",
    "le seneschal porta les lettres au chastelain de gisors",
    "en icele annee fu grant famine en tout le roiaume",
    "li preudomme de la vile assemblerent en la place commune",
    "le tresor du roi estoit garde en la tour du louvre",
    "li evesque de paris precha devant le roi et sa court",
    "la guerre entre france et angleterre dura trente ans",
]

print(f"Corpus : {len(CORPUS)} phrases")
print("Exemple :", CORPUS[0])

# %% [markdown]
# ### 1.2 BPE : initialisation
#
# **Principe :**
# BPE part de la représentation caractère par caractère de chaque mot,
# puis fusionne itérativement la paire de symboles adjacents la plus fréquente.
# Après `num_merges` fusions, les mots fréquents forment des tokens entiers ;
# les mots rares restent décomposés en sous-mots.
#
# Chaque mot est représenté comme une séquence de caractères séparés par des
# espaces, avec le marqueur `</w>` en fin de mot (convention standard).
#
# **Exemple :**
# ```
# "roys"  →  "r o y s </w>"
# ```

# %% [FOURNI]
def get_initial_vocab(corpus):
    """
    Construit le vocabulaire initial : chaque mot du corpus
    est représenté comme une suite de caractères + marqueur </w>.

    Retourne un Counter : {représentation_caractères: fréquence}
    """
    vocab = collections.Counter()
    for line in corpus:
        for word in line.split():
            # "roys" → "r o y s </w>"
            char_repr = " ".join(list(word)) + " </w>"
            vocab[char_repr] += 1
    return vocab

vocab_init = get_initial_vocab(CORPUS)
print("Exemples de mots dans le vocabulaire initial :")
for w, freq in list(vocab_init.most_common(5)):
    print(f"  {w!r:35s}  (freq={freq})")

# %% [markdown]
# ### 1.3 BPE : comptage des paires et fusion
#
# **À vous de jouer.**
#
# Complétez les deux fonctions ci-dessous :
# - `get_pairs` : compte la fréquence de toutes les paires de symboles adjacents
# - `merge_pair` : applique une fusion (remplace la paire par le symbole fusionné)

# %% [TODO]
def get_pairs(vocab):
    """
    Compte les paires de symboles adjacents dans le vocabulaire.

    Paramètre
    ---------
    vocab : Counter  {représentation_en_symboles: fréquence}

    Retourne
    --------
    Counter  {(symbole_a, symbole_b): fréquence_totale}

    Exemple
    -------
    Si vocab = {"r o y s </w>": 3, "r o i </w>": 2}
    alors la paire ("r", "o") a une fréquence de 3+2 = 5.
    """
    pairs = collections.Counter()
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            # TODO : incrémenter le compteur pour la paire (symbols[i], symbols[i+1])
            # par la fréquence du mot
            pass  # ← remplacer cette ligne
    return pairs


def merge_pair(vocab, pair):
    """
    Applique une fusion BPE : remplace toutes les occurrences de `pair`
    par le symbole fusionné dans le vocabulaire.

    Paramètre
    ---------
    vocab : Counter  vocabulaire courant
    pair  : tuple (str, str)  la paire à fusionner, ex. ("r", "o")

    Retourne
    --------
    Counter  nouveau vocabulaire avec la fusion appliquée

    Indice
    ------
    La représentation de la paire dans un mot est `pair[0] + " " + pair[1]`.
    Son remplacement est `pair[0] + pair[1]` (les deux symboles collés).
    Utilisez str.replace() sur la chaîne représentant le mot.
    """
    new_vocab = collections.Counter()
    bigram      = pair[0] + " " + pair[1]
    replacement = pair[0] + pair[1]
    for word, freq in vocab.items():
        # TODO : appliquer le remplacement et stocker dans new_vocab
        pass  # ← remplacer cette ligne
    return new_vocab

# %% [markdown]
# **Cellule de validation 1.3** — exécutez sans modifier.

# %% [FOURNI — validation]
_vocab_test = collections.Counter({"r o y s </w>": 3, "r o i </w>": 2})
_pairs = get_pairs(_vocab_test)
assert ("r", "o") in _pairs and _pairs[("r", "o")] == 5, \
    "get_pairs : la paire ('r','o') devrait avoir fréquence 5."
_vocab_merged = merge_pair(_vocab_test, ("r", "o"))
assert "ro y s </w>" in _vocab_merged, \
    "merge_pair : 'ro y s </w>' devrait apparaître après la fusion de ('r','o')."
print("Validation 1.3 : OK")

# %% [markdown]
# ### 1.4 BPE : construction complète du vocabulaire
#
# Maintenant que les deux fonctions de base sont prêtes, nous pouvons
# dérouler l'algorithme complet.

# %% [FOURNI]
def build_bpe_vocab(corpus, num_merges=80):
    """
    Construit le vocabulaire BPE complet et la liste des fusions appliquées.

    Retourne
    --------
    token2id : dict {token: id}
    id2token : dict {id: token}
    merges   : list [(symbole_a, symbole_b)]  dans l'ordre d'application
    """
    vocab = get_initial_vocab(corpus)
    merges = []

    for step in range(num_merges):
        pairs = get_pairs(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        merges.append(best)
        vocab = merge_pair(vocab, best)

    # Construction du dictionnaire token → id
    all_tokens = set()
    for word in vocab:
        all_tokens.update(word.split())

    # Tokens spéciaux réservés
    special = ["[PAD]", "[MASK]", "[UNK]"]
    token2id = {t: i for i, t in enumerate(special)}
    for tok in sorted(all_tokens):
        if tok not in token2id:
            token2id[tok] = len(token2id)
    id2token = {v: k for k, v in token2id.items()}
    return token2id, id2token, merges


NUM_MERGES = 80
token2id, id2token, merges = build_bpe_vocab(CORPUS, num_merges=NUM_MERGES)
VOCAB_SIZE = len(token2id)

PAD_ID  = token2id["[PAD]"]
MASK_ID = token2id["[MASK]"]
UNK_ID  = token2id["[UNK]"]

print(f"Taille du vocabulaire BPE : {VOCAB_SIZE} tokens")
print(f"  dont [PAD]={PAD_ID}, [MASK]={MASK_ID}, [UNK]={UNK_ID}")
print(f"\nPremières fusions apprises :")
for i, (a, b) in enumerate(merges[:10], 1):
    print(f"  {i:2d}. ({a!r}, {b!r})  →  {a+b!r}")

# %% [markdown]
# ### 1.5 Encoder un texte avec BPE
#
# **À vous de jouer.**
#
# Complétez `bpe_tokenize_word` : elle applique séquentiellement les fusions
# BPE apprises à un seul mot.

# %% [TODO]
def bpe_tokenize_word(word, merges):
    """
    Segmente un mot en sous-mots BPE en appliquant les fusions dans l'ordre.

    Paramètre
    ---------
    word   : str   un seul mot (ex. "roys")
    merges : list  liste ordonnée des fusions [(a, b), ...]

    Retourne
    --------
    list[str]  sous-mots BPE (ex. ["roys</w>"] si "roys" est fréquent,
               ou ["ro", "y", "s</w>"] sinon)

    Algorithme
    ----------
    1. Initialiser symbols = liste des caractères du mot + ["</w>"]
    2. Pour chaque fusion (a, b) dans merges :
       a. Parcourir symbols avec un index i
       b. Si symbols[i] == a et symbols[i+1] == b :
              remplacer les deux par a+b, avancer i de 2
          Sinon : conserver symbols[i], avancer i de 1
       c. Mettre à jour symbols
    3. Retourner symbols
    """
    symbols = list(word) + ["</w>"]
    for a, b in merges:
        i = 0
        new_syms = []
        while i < len(symbols):
            # TODO : implémenter la logique de fusion décrite ci-dessus
            # Penser au cas limite : i == len(symbols) - 1 (dernier élément)
            pass  # ← remplacer cette ligne
        symbols = new_syms
    return symbols


def encode(sentence, token2id, merges, unk_id=None):
    """Encode une phrase entière en liste d'ids."""
    if unk_id is None:
        unk_id = token2id["[UNK]"]
    ids = []
    for word in sentence.split():
        for tok in bpe_tokenize_word(word, merges):
            ids.append(token2id.get(tok, unk_id))
    return ids

# %% [markdown]
# **Cellule de validation 1.5**

# %% [FOURNI — validation]
_toks = bpe_tokenize_word("de", merges)
assert isinstance(_toks, list) and len(_toks) >= 1, \
    "bpe_tokenize_word doit retourner une liste non vide."
_ids = encode("li roys de france", token2id, merges)
assert len(_ids) >= 4, \
    "encode doit retourner au moins un id par mot."
assert all(isinstance(i, int) for i in _ids), \
    "encode doit retourner des entiers."
print("Validation 1.5 : OK")
print("Tokenisation de 'li roys de france' :")
for tid in _ids:
    print(f"  {id2token[tid]!r:15s} (id={tid})")

# %% [markdown]
# **Observation :** comparez la tokenisation de *"roys"* avec celle de *"roi"*.
# Le premier, moins fréquent dans notre corpus miniature, sera plus fragmenté.
# C'est précisément le problème du moyen français avec CamemBERT à grande échelle.

# %% [markdown]
# ## Partie 2 — Encodage positionnel sinusoïdal
#
# ### 2.1 Rappel de la formule
#
# $$PE_{t,\,2i} = \sin\!\left(\frac{t}{10000^{2i/d_{\text{model}}}}\right)$$
# $$PE_{t,\,2i+1} = \cos\!\left(\frac{t}{10000^{2i/d_{\text{model}}}}\right)$$
#
# - $t$ : position du token dans la séquence
# - $i$ : indice de dimension (de 0 à $d_{\text{model}}/2 - 1$)
#
# **À vous de jouer.** Complétez la fonction ci-dessous.

# %% [TODO]
def sinusoidal_encoding(seq_len, d_model):
    """
    Calcule la matrice d'encodage positionnel sinusoïdal.

    Paramètres
    ----------
    seq_len : int   longueur maximale de séquence
    d_model : int   dimension du modèle (doit être pair)

    Retourne
    --------
    np.ndarray  de forme (seq_len, d_model), dtype float32

    Algorithme
    ----------
    Pour chaque position t in range(seq_len) :
      Pour chaque indice i in range(0, d_model, 2) :
        PE[t, i]   = sin(t / 10000^(i / d_model))
        PE[t, i+1] = cos(t / 10000^(i / d_model))   [si i+1 < d_model]

    Conseil : utilisez math.sin, math.cos.
    """
    PE = np.zeros((seq_len, d_model), dtype=np.float32)
    for t in range(seq_len):
        for i in range(0, d_model, 2):
            # TODO : calculer l'angle et remplir PE[t, i] et PE[t, i+1]
            pass  # ← remplacer cette ligne
    return PE

# %% [markdown]
# **Cellule de validation 2.1**

# %% [FOURNI — validation]
_PE = sinusoidal_encoding(50, 16)
assert _PE.shape == (50, 16), "La forme doit être (seq_len, d_model)."
assert abs(_PE[0, 0]) < 1e-6, "PE[0, 0] = sin(0) doit valoir 0."
assert abs(_PE[0, 1] - 1.0) < 1e-5, "PE[0, 1] = cos(0) doit valoir 1."
assert abs(_PE[1, 0] - math.sin(1.0)) < 1e-5, \
    f"PE[1, 0] devrait valoir sin(1) ≈ {math.sin(1):.4f}."
print("Validation 2.1 : OK")

# %% [markdown]
# **Visualisation de la matrice d'encodage positionnel**

# %% [FOURNI]
PE_viz = sinusoidal_encoding(50, 64)
fig, ax = plt.subplots(figsize=(12, 4))
im = ax.imshow(PE_viz.T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xlabel("Position dans la séquence (t)", fontsize=11)
ax.set_ylabel("Dimension de l'embedding (i)", fontsize=11)
ax.set_title("Encodage positionnel sinusoïdal — PE(t, i)", fontsize=13)
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig("positional_encoding.pdf", dpi=150)
plt.show()
print("Figure sauvegardée : positional_encoding.pdf")

# %% [markdown]
# **Question 2.1** : Que représentent les bandes verticales dans cette image ?
# Pourquoi les basses dimensions (en haut) oscillent-elles plus lentement que
# les hautes dimensions (en bas) ?
#
# *(Réponse attendue dans votre compte-rendu)*

# %% [markdown]
# ## Partie 3 — Bloc Transformer Encoder avec Keras
#
# ### 3.1 La couche MultiHeadAttention de Keras
#
# Keras fournit `layers.MultiHeadAttention`. Sa signature est :
# ```python
# attn_layer = layers.MultiHeadAttention(num_heads=h, key_dim=d_k)
# output = attn_layer(query, value, key=None, training=False)
# # Si key=None, Keras utilise value comme key (self-attention).
# ```
# Lorsque `query == value`, c'est de la **self-attention**.
# Le paramètre `key_dim` correspond à $d_k = d_{\text{model}} / h$.
#
# ### 3.2 Implémentation du bloc Transformer
#
# **À vous de jouer.** Complétez la méthode `call` du bloc Transformer.

# %% [TODO]
class TransformerEncoderBlock(layers.Layer):
    """
    Un bloc Transformer Encoder complet :

        x → MultiHeadSelfAttention → Add & LayerNorm → FFN → Add & LayerNorm

    Paramètres
    ----------
    d_model : int   dimension du modèle
    n_heads : int   nombre de têtes d'attention
    d_ff    : int   dimension interne du FFN (généralement 4 × d_model)
    dropout : float taux de dropout
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, **kwargs):
        super().__init__(**kwargs)

        # Self-attention multi-têtes
        self.attn  = layers.MultiHeadAttention(
            num_heads=n_heads,
            key_dim=d_model // n_heads,
        )
        # Feed-Forward Network : deux couches denses
        self.ffn = keras.Sequential([
            layers.Dense(d_ff, activation="gelu"),   # expansion
            layers.Dense(d_model),                   # projection
        ])
        # Normalisation de couche (après chaque sous-bloc)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        # Dropout (appliqué sur la sortie de l'attention et du FFN)
        self.drop1 = layers.Dropout(dropout)
        self.drop2 = layers.Dropout(dropout)

    def call(self, x, training=False):
        """
        Paramètre
        ---------
        x : Tensor  (batch, seq_len, d_model)

        Retourne
        --------
        Tensor  (batch, seq_len, d_model)

        Séquence d'opérations
        ---------------------
        1. Self-attention : attn_out = self.attn(x, x, training=training)
        2. Dropout sur attn_out
        3. Connexion résiduelle + LayerNorm : x = self.norm1(x + dropout(attn_out))
        4. FFN : ffn_out = self.ffn(x)
        5. Dropout sur ffn_out
        6. Connexion résiduelle + LayerNorm : x = self.norm2(x + dropout(ffn_out))
        7. Retourner x
        """
        # TODO : implémenter les 7 étapes décrites ci-dessus
        pass  # ← remplacer cette ligne

# %% [markdown]
# **Cellule de validation 3.2**

# %% [FOURNI — validation]
_block = TransformerEncoderBlock(d_model=32, n_heads=4, d_ff=64)
_x_test = np.random.randn(2, 10, 32).astype(np.float32)
_out = _block(_x_test, training=False)
assert _out.shape == (2, 10, 32), \
    f"La sortie du bloc doit avoir la même forme que l'entrée (batch, seq, d_model). Obtenu : {_out.shape}"
print("Validation 3.2 : OK — forme de sortie :", _out.shape)

# %% [markdown]
# **Question 3.2** : La connexion résiduelle `x + attn_out` est ajoutée
# *avant* la LayerNorm (Post-LN, style original Vaswani et al.).
# Quelle est l'alternative (Pre-LN) et quels sont ses avantages ?
# *(Réponse attendue dans votre compte-rendu)*

# %% [markdown]
# ## Partie 4 — Modèle complet et données MLM
#
# ### 4.1 Hyperparamètres du modèle — version GPU
#
# Les valeurs CPU sont indiquées en commentaire pour faciliter la comparaison.
# Le facteur d'agrandissement est choisi pour produire un modèle de ~1.5 M
# paramètres — suffisamment grand pour saturer un T4 Colab sur ce corpus,
# suffisamment petit pour converger en quelques minutes.

# %% [FOURNI]
SEQ_LEN  = 64    # CPU : 20  — séquences plus longues, plus de contexte médiéval
D_MODEL  = 128   # CPU : 32  — représentations 4× plus riches
N_HEADS  = 8     # CPU : 4   — d_k = 128/8 = 16 par tête
D_FF     = 512   # CPU : 64  — FFN 4× D_MODEL (ratio standard BERT)
N_LAYERS = 4     # CPU : 2   — profondeur doublée
DROPOUT  = 0.1   # inchangé

print("Hyperparamètres du modèle (GPU) :")
print(f"  SEQ_LEN={SEQ_LEN}, D_MODEL={D_MODEL}, N_HEADS={N_HEADS}")
print(f"  D_FF={D_FF}, N_LAYERS={N_LAYERS}, VOCAB_SIZE={VOCAB_SIZE}")
print(f"  → d_k par tête = {D_MODEL // N_HEADS}")
n_params_est = (VOCAB_SIZE*D_MODEL
                + N_LAYERS*(4*D_MODEL**2 + 2*D_MODEL*D_FF)
                + D_MODEL*VOCAB_SIZE)
print(f"  → paramètres estimés : ~{n_params_est:,}")

# %% [markdown]
# ### 4.2 Préparation des données MLM
#
# **Protocole de masquage (simplifié d'après BERT) :**
# - Chaque token non-padding est masqué avec probabilité `MASK_PROB`.
# - Les tokens masqués sont remplacés par `[MASK]` dans l'entrée.
# - Le label est l'id du token original pour les positions masquées,
#   et `-100` pour les positions non masquées (ignorées dans la loss).
#
# **À vous de jouer.** Complétez `make_mlm_sample`.

# %% [TODO]
MASK_PROB = 0.15

def make_mlm_sample(token_ids, seq_len, mask_prob, mask_id, pad_id):
    """
    Prépare un exemple MLM à partir d'une liste d'ids de tokens.

    Paramètres
    ----------
    token_ids : list[int]   ids des tokens (sans padding)
    seq_len   : int         longueur cible (troncature + padding)
    mask_prob : float       probabilité de masquage (ex. 0.15)
    mask_id   : int         id du token [MASK]
    pad_id    : int         id du token [PAD]

    Retourne
    --------
    masked_ids : list[int]  séquence d'entrée (certains tokens → [MASK])
    labels     : list[int]  -100 si non masqué, id original si masqué

    Algorithme
    ----------
    1. Tronquer token_ids à seq_len.
    2. Compléter avec pad_id jusqu'à seq_len (padding à droite).
    3. Initialiser masked_ids = copie de ids, labels = [-100] * seq_len.
    4. Pour chaque position i :
         - Si ids[i] == pad_id : laisser labels[i] = -100, ne pas masquer.
         - Sinon, tirer u ~ Uniform(0,1) :
             * Si u < mask_prob : masked_ids[i] = mask_id,
                                  labels[i]     = ids[i]  (token original)
             * Sinon            : rien (token conservé, label = -100)
    """
    # Étape 1 : troncature
    ids = token_ids[:seq_len]
    # Étape 2 : padding
    ids = ids + [pad_id] * (seq_len - len(ids))

    labels     = [-100] * seq_len
    masked_ids = ids[:]

    for i in range(seq_len):
        # TODO : implémenter les étapes 3 et 4
        pass  # ← remplacer cette ligne

    return masked_ids, labels

# %% [markdown]
# **Cellule de validation 4.2**

# %% [FOURNI — validation]
random.seed(0)
_sample_ids = list(range(5, 15))   # ids fictifs 5..14
_mx, _ly = make_mlm_sample(_sample_ids, seq_len=12, mask_prob=0.9,
                             mask_id=MASK_ID, pad_id=PAD_ID)
assert len(_mx) == 12 and len(_ly) == 12, "Les listes doivent avoir longueur seq_len."
assert _mx[10] == PAD_ID and _ly[10] == -100, \
    "Les positions de padding doivent rester PAD_ID avec label -100."
# Avec mask_prob=0.9, au moins un token doit être masqué parmi les 10 non-padding
_masked_count = sum(1 for v in _mx[:10] if v == MASK_ID)
assert _masked_count >= 1, "Avec mask_prob=0.9, au moins un token doit être masqué."
print(f"Validation 4.2 : OK — {_masked_count}/10 tokens masqués (prob=0.9)")
random.seed(SEED)

# %% [markdown]
# ### 4.3 Construction du dataset — pipeline tf.data

# %% [FOURNI]
# ── Génération des exemples ───────────────────────────────────────────────
# On répète 200× (vs 20× CPU) pour produire 3 000 exemples.
# Avec SEQ_LEN=64 et batch_size=64, chaque epoch = ~47 steps GPU.
random.seed(SEED)
all_X, all_Y = [], []
for line in CORPUS * 200:   # CPU : * 20
    ids = encode(line, token2id, merges)
    x, y = make_mlm_sample(ids, SEQ_LEN, MASK_PROB, MASK_ID, PAD_ID)
    all_X.append(x)
    all_Y.append(y)

X = np.array(all_X, dtype=np.int32)   # (n_samples, SEQ_LEN)
Y = np.array(all_Y, dtype=np.int32)   # (n_samples, SEQ_LEN)

print(f"Dataset : {X.shape[0]} exemples × {X.shape[1]} tokens")

# ── Pipeline tf.data ──────────────────────────────────────────────────────
# Différence clé avec la version CPU (model.fit sur numpy array) :
#
#   numpy array → chaque epoch recopie les données CPU→GPU = goulot d'I/O.
#   tf.data     → .cache() garde le dataset en mémoire GPU après la 1ère epoch ;
#                 .prefetch(AUTOTUNE) prépare le batch N+1 pendant que le GPU
#                 calcule sur le batch N, éliminant le temps d'attente CPU.
#
BATCH_SIZE = 64      # CPU : 8  — batch plus grand pour amortir les transferts GPU
VAL_SPLIT  = 0.1

n_val    = int(len(X) * VAL_SPLIT)
n_train  = len(X) - n_val

X_train, Y_train = X[:n_train], Y[:n_train]
X_val,   Y_val   = X[n_train:], Y[n_train:]

AUTOTUNE = tf.data.AUTOTUNE

train_ds = (
    tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    .shuffle(buffer_size=n_train, seed=SEED)
    .batch(BATCH_SIZE)
    .cache()                       # mis en cache GPU après la 1ère epoch
    .prefetch(AUTOTUNE)            # préchargement asynchrone
)
val_ds = (
    tf.data.Dataset.from_tensor_slices((X_val, Y_val))
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTOTUNE)
)

print(f"Train : {n_train} exemples ({len(list(train_ds))} batches de {BATCH_SIZE})")
print(f"Val   : {n_val} exemples ({len(list(val_ds))} batches de {BATCH_SIZE})")
print(f"Inspection du premier exemple (10 premières positions) :")
ex_x, ex_y = X[0], Y[0]
for pos in range(10):
    xi, yi = ex_x[pos], ex_y[pos]
    marker = " ← MASQUÉ" if xi == MASK_ID else ""
    label_str = f"(label={id2token.get(yi,'?')!r})" if yi != -100 else ""
    print(f"  pos {pos:2d}  id={xi:3d}  {id2token.get(xi,'?')!r:15s} {label_str}{marker}")

# %% [markdown]
# ## Partie 5 — Construction du modèle Keras

# %% [FOURNI]
PE_matrix = sinusoidal_encoding(SEQ_LEN, D_MODEL)   # (SEQ_LEN, D_MODEL)

inp = keras.Input(shape=(SEQ_LEN,), dtype="int32", name="input_ids")

# ── Embedding des tokens ──────────────────────────────────────────────────
x = layers.Embedding(VOCAB_SIZE, D_MODEL, name="token_embedding")(inp)
# x : (batch, SEQ_LEN, D_MODEL)

# ── Encodage positionnel : ajout du PE sinusoïdal ────────────────────────
# On convertit PE_matrix en tensor et on l'ajoute (broadcast sur le batch)
pe_tensor = ops.convert_to_tensor(PE_matrix[np.newaxis, :, :])  # (1, SEQ_LEN, D_MODEL)
x = x + pe_tensor

x = layers.Dropout(DROPOUT, name="emb_dropout")(x)

# ── Empilage des blocs Transformer ───────────────────────────────────────
for layer_idx in range(N_LAYERS):
    x = TransformerEncoderBlock(
        d_model=D_MODEL,
        n_heads=N_HEADS,
        d_ff=D_FF,
        dropout=DROPOUT,
        name=f"transformer_block_{layer_idx}"
    )(x)

# ── Tête MLM : projection sur le vocabulaire ────────────────────────────
logits = layers.Dense(VOCAB_SIZE, name="mlm_head")(x)
# logits : (batch, SEQ_LEN, VOCAB_SIZE)

model = keras.Model(inputs=inp, outputs=logits, name="MiniTransformerMLM")
model.summary()

# %% [markdown]
# **Question 5** : Le modèle GPU a ~1.5 M paramètres (vs ~25 000 en CPU).
# Identifiez dans le `summary` :
# - Combien de paramètres pour l'embedding ?
# - Combien pour un bloc Transformer (attention + FFN) ?
# - Combien pour la tête MLM ?
#
# Calculez le ratio GPU/CPU pour chaque composant.
# Lequel a le plus bénéficié de l'agrandissement ?
# *(Indice : l'attention croît en $O(d_{\text{model}}^2)$, le FFN également.)*

# %% [markdown]
# ## Partie 6 — Loss MLM et entraînement
#
# ### 6.1 La loss MLM
#
# La loss MLM est une entropie croisée *sparse* (labels = indices de classes,
# non one-hot), appliquée **uniquement** sur les positions masquées.
# Les positions avec label `-100` sont ignorées grâce à un masque binaire.

# %% [FOURNI]
def mlm_loss(y_true, y_pred):
    """
    Loss MLM : entropie croisée sur les tokens masqués uniquement.

    Paramètres
    ----------
    y_true : Tensor (batch, seq_len)  labels (-100 = ignorer)
    y_pred : Tensor (batch, seq_len, vocab_size)  logits

    Retourne
    --------
    Scalar  loss moyenne sur les positions masquées
    """
    # Masque : 1 là où le label n'est pas -100
    mask = ops.cast(y_true != -100, dtype="float32")

    # Entropie croisée token par token (from_logits=True car pas de softmax en sortie)
    # On remplace les -100 par 0 pour éviter les indices hors-vocabulaire
    y_true_safe = ops.where(y_true == -100, ops.zeros_like(y_true), y_true)
    per_token_loss = keras.losses.sparse_categorical_crossentropy(
        y_true_safe, y_pred, from_logits=True
    )   # (batch, seq_len)

    # Appliquer le masque et moyenner sur les positions masquées
    masked_loss = per_token_loss * mask
    return ops.sum(masked_loss) / (ops.sum(mask) + 1e-8)

# %% [markdown]
# ### 6.2 Compilation et entraînement
#
# **À vous de jouer.** Complétez l'appel à `model.compile` en choisissant
# l'optimiseur (Adam, lr=3e-4) et la loss `mlm_loss` définie ci-dessus.

# %% [markdown]
# ### 6.2 Compilation et entraînement — version GPU
#
# **À vous de jouer.** Complétez l'appel à `model.compile`.
# La seule différence avec la version CPU est l'ajout de `jit_compile=True`
# qui active la compilation XLA décrite en Partie 0.

# %% [TODO]
# TODO : compléter model.compile avec :
#   - optimizer  : Adam, learning_rate=3e-4
#   - loss       : mlm_loss
#   - jit_compile: True   ← nouveauté GPU (compilation XLA)

# model.compile(...)   ← décommenter et compléter

# %% [FOURNI]
# ── Callbacks ────────────────────────────────────────────────────────────
# EarlyStopping : arrête l'entraînement si val_loss ne progresse plus
# depuis `patience` epochs — évite le sur-apprentissage et économise du temps.
# restore_best_weights recharge automatiquement les poids de la meilleure epoch.
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1,
)

# ── Entraînement ─────────────────────────────────────────────────────────
# Note : on passe train_ds et val_ds (tf.data.Dataset) au lieu de
# (X, Y, validation_split=...) — cela active le pipeline prefetch/cache.
# La 1ère epoch est plus lente (compilation XLA + mise en cache du dataset).
print("Démarrage de l'entraînement (1ère epoch plus lente : warmup XLA)...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=40,              # CPU : 60 — réduit car early stopping prend le relais
    callbacks=[early_stop],
    verbose=1,
)

# %% [markdown]
# ### 6.3 Courbe d'apprentissage

# %% [FOURNI]
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(history.history["loss"],     label="Train loss", linewidth=2)
ax.plot(history.history["val_loss"], label="Val loss",   linewidth=2, linestyle="--")
ax.set_xlabel("Epoch")
ax.set_ylabel("MLM Loss")
ax.set_title("Courbe d'apprentissage — MiniTransformer MLM")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("training_curve.pdf", dpi=150)
plt.show()

# %% [markdown]
# **Question 6.3** : À quelle epoch l'early stopping a-t-il déclenché l'arrêt ?
# Comparez visuellement la courbe train/val avec celle de la version CPU (60 epochs
# sans early stopping). Qu'apporte l'early stopping en termes de généralisation ?
# Auriez-vous pu obtenir un résultat équivalent en fixant simplement `epochs=N`
# pour un N bien choisi ? Discutez.

# %% [markdown]
# ## Partie 7 — Inférence : démasquage
#
# ### 7.1 Fonction de prédiction
#
# **À vous de jouer.** Complétez `predict_masked` : elle encode une phrase
# contenant le token spécial `"[MASK]"`, l'envoie dans le modèle,
# et retourne les top-k prédictions pour chaque position masquée.

# %% [TODO]
def predict_masked(sentence, model, token2id, id2token, merges,
                   seq_len=SEQ_LEN, mask_id=MASK_ID, pad_id=PAD_ID, top_k=5):
    """
    Prédit les tokens masqués dans une phrase.

    Paramètre
    ---------
    sentence : str  phrase avec le mot "[MASK]" à la place du token à prédire
                    (ex. "li [MASK] de france signa l acte en son palays")

    Retourne
    --------
    list of (position, [(token_str, logit_float), ...])
        Une entrée par position masquée, avec les top_k candidats.

    Algorithme
    ----------
    1. Tokeniser mot par mot :
         - Si le mot est "[MASK]" : ajouter mask_id à la liste d'ids.
         - Sinon : appliquer bpe_tokenize_word et convertir en ids.
    2. Tronquer/padder à seq_len.
    3. Créer inp_arr = np.array([ids], dtype=np.int32)  — batch de taille 1.
    4. logits = model.predict(inp_arr, verbose=0)  → (1, seq_len, vocab_size)
    5. Pour chaque position où ids[pos] == mask_id :
         a. Extraire logits[0, pos, :]
         b. Trier par ordre décroissant (np.argsort [::-1])
         c. Garder les top_k premiers ids et logits correspondants
    6. Retourner la liste de résultats.
    """
    ids = []
    for word in sentence.split():
        if word == "[MASK]":
            # TODO : ajouter mask_id
            pass
        else:
            # TODO : tokeniser le mot et ajouter les ids
            pass

    # TODO : tronquer/padder à seq_len

    results = []
    # TODO : inférence et extraction des top_k pour chaque position masquée

    return results

# %% [markdown]
# ### 7.2 Test de démasquage

# %% [FOURNI]
test_sentences = [
    "li [MASK] de france signa l acte en son palays",
    "le conte de champagne et le [MASK] de bourgogne firent accord",
    "en l an de grace mil trois cent la [MASK] trespassa",
]

for sent in test_sentences:
    print(f"\nPhrase  : {sent}")
    results = predict_masked(sent, model, token2id, id2token, merges)
    for pos, candidates in results:
        print(f"  [MASK] à pos {pos} — top 5 candidats :")
        for rank, (tok, score) in enumerate(candidates, 1):
            print(f"    {rank}. {tok!r:18s} (logit={score:+.2f})")

# %% [markdown]
# **Question 7.2** : Les prédictions sont-elles linguistiquement sensées ?
# Si non, quels facteurs limitants du modèle (taille du corpus, taille du
# modèle, nombre d'epochs) expliquent les erreurs ?

# %% [markdown]
# ## Partie 8 — Visualisation des poids d'attention
#
# ### 8.1 Extraction des poids d'attention
#
# Keras `MultiHeadAttention` expose les poids d'attention via le paramètre
# `return_attention_scores=True`. Nous devons construire un modèle auxiliaire
# qui retourne ces poids.

# %% [FOURNI]
def build_attention_extractor(model, layer_idx=0):
    """
    Construit un sous-modèle qui retourne les poids d'attention
    du bloc Transformer `layer_idx`.

    La couche MultiHeadAttention de Keras retourne (output, scores)
    lorsqu'on appelle attn(q, v, return_attention_scores=True).
    """
    block = model.get_layer(f"transformer_block_{layer_idx}")
    inp   = model.input

    # On réconstruit le forward pass jusqu'au bloc visé
    # en récupérant la sortie intermédiaire après l'embedding + PE
    emb_out = model.get_layer("emb_dropout")(
        ops.convert_to_tensor(PE_matrix[np.newaxis, :, :]) +
        model.get_layer("token_embedding")(inp)
    )
    # Passer dans les blocs précédents si layer_idx > 0
    x = emb_out
    for i in range(layer_idx):
        x = model.get_layer(f"transformer_block_{i}")(x)

    # Appel explicite à la couche d'attention avec retour des scores
    attn_out, attn_scores = block.attn(
        x, x, return_attention_scores=True
    )
    return keras.Model(inputs=inp, outputs=attn_scores,
                       name=f"attn_extractor_block{layer_idx}")


def visualize_attention_heads(sentence, model, token2id, id2token, merges,
                               seq_len=SEQ_LEN, layer_idx=0):
    """
    Affiche les poids d'attention de toutes les têtes pour une phrase.
    """
    ids = encode(sentence, token2id, merges)
    ids = ids[:seq_len] + [PAD_ID] * max(0, seq_len - len(ids))
    tokens_str = [id2token.get(i, "?") for i in ids]

    inp_arr = np.array([ids], dtype=np.int32)
    extractor = build_attention_extractor(model, layer_idx=layer_idx)
    attn = extractor.predict(inp_arr, verbose=0)   # (1, n_heads, seq, seq)
    attn = attn[0]                                  # (n_heads, seq, seq)

    n_heads = attn.shape[0]
    fig, axes = plt.subplots(1, n_heads, figsize=(4 * n_heads, 5))
    if n_heads == 1:
        axes = [axes]

    for h, ax in enumerate(axes):
        im = ax.imshow(attn[h], cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(seq_len))
        ax.set_yticks(range(seq_len))
        ax.set_xticklabels(tokens_str, rotation=90, fontsize=7)
        ax.set_yticklabels(tokens_str, fontsize=7)
        ax.set_title(f"Tête {h+1}", fontsize=10)

    fig.suptitle(f"Poids d'attention — Bloc {layer_idx+1}\n\"{sentence}\"",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(f"attention_block{layer_idx}.pdf", dpi=150)
    plt.show()
    print(f"Figure sauvegardée : attention_block{layer_idx}.pdf")

# %% [FOURNI]
phrase_attn = "li roys de france signa l acte en son palays"
visualize_attention_heads(phrase_attn, model, token2id, id2token, merges,
                           layer_idx=0)
visualize_attention_heads(phrase_attn, model, token2id, id2token, merges,
                           layer_idx=1)

# %% [markdown]
# **Question 8** : Comparez les matrices d'attention du bloc 1 et du bloc 2.
# - Bloc 1 : l'attention est-elle concentrée sur les tokens proches (locale)
#   ou distribuée (globale) ?
# - Bloc 2 : observe-t-on des têtes qui "ignorent" certains tokens ?
#   Lesquels (tokens de padding notamment) ?
# - Quel lien faites-vous avec les résultats de Clark et al. (2019) évoqués
#   en cours ?

# %% [markdown]
# ## Partie 9 — Pour aller plus loin (optionnel)
#
# Les exercices ci-dessous ne sont pas notés mais préparent directement le Jour 2.

# %% [markdown]
# ### 9.1 Taux OOV sur votre corpus réel
#
# Remplacez `CORPUS` par vos transcriptions HTR et mesurez le taux de tokens
# `[UNK]` produits par notre tokeniseur BPE jouet vs. CamemBERT.
#
# ```python
# from transformers import AutoTokenizer
# tok_camembert = AutoTokenizer.from_pretrained("almanach/camembert-base")
#
# def oov_rate(corpus, tokenizer):
#     unk_id = tokenizer.unk_token_id
#     total, oov = 0, 0
#     for line in corpus:
#         ids = tokenizer.encode(line, add_special_tokens=False)
#         total += len(ids)
#         oov   += sum(1 for i in ids if i == unk_id)
#     return oov / total if total else 0
#
# print(f"OOV CamemBERT : {oov_rate(CORPUS, tok_camembert):.2%}")
# ```

# %% [markdown]
# ### 9.2 RoPE à la place du PE sinusoïdal
#
# Remplacez l'encodage positionnel additif par RoPE :
# au lieu d'additionner PE à l'embedding, appliquez une rotation aux vecteurs
# Q et K avant le calcul d'attention, en vous appuyant sur la formule vue en cours.
# Comparez la courbe de loss avec la version sinusoïdale.

# %% [markdown]
# ### 9.3 Visualisation de l'entropie des têtes
#
# Une tête à entropie élevée distribue son attention uniformément
# (elle "ne sait pas où regarder"). Calculez, pour chaque tête,
# l'entropie moyenne des distributions d'attention et affichez-la.
#
# $$H(\text{tête}_h) = -\frac{1}{n} \sum_{i=1}^{n} \sum_{j=1}^{n} \alpha_{ij}^{(h)} \log \alpha_{ij}^{(h)}$$
#
# Les têtes à haute entropie sont candidates à l'élagage (*pruning*) —
# c'est l'une des intuitions derrière LoRA qui n'adapte que certaines projections.

# %% [markdown]
# ### 9.4 (GPU uniquement) Profiler le pipeline d'entraînement
#
# TensorFlow embarque un profileur GPU accessible via TensorBoard.
# Il permet de visualiser l'utilisation réelle du GPU step par step,
# d'identifier les goulots CPU (chargement de données, preprocessing),
# et de mesurer l'effet du `.prefetch()` et du `.cache()`.
#
# ```python
# import tensorflow as tf
#
# # Créer un callback TensorBoard avec profiling activé
# logs_dir = "logs/profiler"
# tb_callback = tf.keras.callbacks.TensorBoard(
#     log_dir=logs_dir,
#     profile_batch=(2, 5),   # profiler les batches 2 à 5 de la 1ère epoch
# )
#
# # Relancer un entraînement court avec le callback
# model.fit(train_ds, epochs=3, callbacks=[tb_callback])
#
# # Lancer TensorBoard dans Colab :
# # %load_ext tensorboard
# # %tensorboard --logdir logs/profiler
# ```
#
# Dans l'onglet "Profile", observez la colonne "GPU Compute" vs "Input Pipeline".
# Si "Input Pipeline" dépasse 10-15 % du temps total, votre pipeline de données
# est un goulot : augmentez le `buffer_size` du shuffle ou ajoutez un `.cache()`
# plus tôt dans la chaîne.

# %% [markdown]
# ### 9.5 (GPU uniquement) Comparer mixed precision vs float32
#
# Désactivez temporairement la mixed precision et comparez :
# - Le temps par epoch (via `time.time()` autour de `model.fit`).
# - L'utilisation mémoire GPU (`nvidia-smi` ou `tf.config.experimental.get_memory_info`).
# - La loss finale (elle doit être identique à ±1 % — la precision float16
#   ne dégrade pas la convergence si l'optimiseur conserve ses weights en float32).
#
# ```python
# import time
#
# # Version float32 (désactiver mixed precision)
# keras.mixed_precision.set_global_policy("float32")
# # ... reconstruire et compiler le modèle ...
#
# t0 = time.time()
# history_f32 = model.fit(train_ds, epochs=5)
# print(f"float32 : {time.time()-t0:.1f} s — loss={history_f32.history['loss'][-1]:.4f}")
#
# # Version float16 (réactiver)
# keras.mixed_precision.set_global_policy("mixed_float16")
# # ... reconstruire et compiler le modèle ...
#
# t0 = time.time()
# history_f16 = model.fit(train_ds, epochs=5)
# print(f"float16 : {time.time()-t0:.1f} s — loss={history_f16.history['loss'][-1]:.4f}")
# ```

# %% [markdown]
# ---
#
# ## Récapitulatif — Ce que vous avez implémenté
#
# | Brique | Fonction/Classe | Lien avec le cours |
# |---|---|---|
# | Tokenisation BPE | `get_pairs`, `merge_pair`, `bpe_tokenize_word` | Fragmentation du moyen français, taux OOV |
# | Encodage positionnel | `sinusoidal_encoding` | Formule $\sin/\cos$, invariance par permutation |
# | Self-attention | `TransformerEncoderBlock.call` | $\text{softmax}(QK^\top/\sqrt{d_k})V$ |
# | MLM masquage | `make_mlm_sample` | Protocole BERT, label $-100$ |
# | Loss MLM | `mlm_loss` | Entropie croisée masquée |
# | Inférence | `predict_masked` | Top-k décodage |
# | Visualisation | `visualize_attention_heads` | Spécialisation des têtes |
#
# **Adaptations GPU spécifiques à cette version :**
#
# | Technique | Paramètre | Gain |
# |---|---|---|
# | Mixed precision float16 | `set_global_policy("mixed_float16")` | −50 % mémoire, ×1.5–2 vitesse |
# | Compilation XLA | `jit_compile=True` | ×1.2–1.5 après warmup |
# | Pipeline tf.data | `.cache().prefetch(AUTOTUNE)` | élimine la famine GPU |
# | Early stopping | `patience=5, restore_best_weights` | généralisation + temps économisé |
# | Modèle agrandi | D_MODEL=128, N_LAYERS=4 | ~1.5 M params, saturation GPU |
#
# **Vers le Jour 2 :** Vous avez construit un Transformer "from scratch".
# Demain, vous adapterez un Transformer pré-entraîné (CamemBERT, mT5)
# à votre corpus médiéval avec LoRA — une mise à jour de rang faible
# de précisément les matrices $W^Q$ et $W^V$ que vous venez d'implémenter.
# Les techniques GPU de ce TP (mixed precision, tf.data) se retrouvent
# identiques dans les scripts de fine-tuning HuggingFace Trainer.
