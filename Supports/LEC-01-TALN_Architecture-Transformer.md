# Chapitre 1 — Architecture Transformer de A à Z

**Module NLP · Master Data/IA · MD5 Volet 2 · 2026**  
Cours magistral — 3 heures

---

## Avant-propos : pourquoi comprendre l'architecture

Ce cours a une promesse simple : à la fin de ces trois heures, les décisions que vous prendrez dans les TPs suivants ne seront plus des décisions par défaut. Pourquoi r=8 pour LoRA et pas r=64 ? Parce que la matrice de poids que vous allez décomposer n'a pas un rang intrinsèquement élevé — et parce que plus r est grand, plus vous vous rapprochez d'un full fine-tuning en mémoire et en risque de sur-apprentissage. Pourquoi adapter les projections Q et V mais pas K dans certaines configurations LoRA ? Parce que Q et V portent respectivement la requête de sémantique et le contenu récupéré, alors que K joue un rôle de comparaison plus stable. Pourquoi un modèle encoder-only pour la NER et un encoder-decoder pour la normalisation orthographique ? Parce que la NER est une tâche de classification par token qui n'exige pas de génération, alors que la normalisation est une transduction de séquence — c'est une différence architecturale profonde, pas un choix de convenance.

Ces réponses émergent naturellement de la compréhension du mécanisme d'attention. C'est ce que ce chapitre construit.

---

## 1. Les limites du paradigme séquentiel : RNN et LSTM

### 1.1 Le problème de la mémoire séquentielle

Avant les Transformers, le traitement du langage naturel reposait sur des architectures récurrentes : les RNN (*Recurrent Neural Networks*) et leurs variantes à portes, les LSTM (*Long Short-Term Memory*) et GRU (*Gated Recurrent Units*). L'idée de base est intuitive : pour traiter une séquence de tokens $(x_1, x_2, \ldots, x_T)$, on maintient un état caché $h_t$ qui agrège l'information des tokens précédents :

$$h_t = f(W_h \cdot h_{t-1} + W_x \cdot x_t + b)$$

L'état caché $h_t$ est censé encoder "tout ce qui s'est passé avant le token $t$". En pratique, cet état est un vecteur de dimension fixe — souvent 512 ou 1024 — qui doit compresser l'ensemble du contexte passé. C'est une contrainte architecturale fondamentale, pas un défaut d'implémentation.

### 1.2 Le vanishing gradient

Le problème du gradient évanescent (*vanishing gradient*) est le premier obstacle structurel des RNN. Lors de la rétropropagation, le gradient de la perte par rapport aux paramètres des couches précoces traverse le réseau en se multipliant à chaque pas de temps. Si la dérivée de la fonction d'activation $f$ est systématiquement inférieure à 1 — ce qui arrive facilement avec la sigmoïde ou la tangente hyperbolique — le gradient décroît exponentiellement avec la longueur de la séquence.

Pour une séquence de longueur $T$, la norme du gradient en $h_1$ est approximativement :

$$\left\|\frac{\partial \mathcal{L}}{\partial h_1}\right\| \approx \left\|\frac{\partial \mathcal{L}}{\partial h_T}\right\| \cdot \prod_{t=2}^{T} \left\|\frac{\partial h_t}{\partial h_{t-1}}\right\|$$

Si chaque facteur vaut 0.9, après 50 pas de temps le gradient a diminué d'un facteur $0.9^{50} \approx 0.005$. La dépendance longue distance — par exemple, relier le sujet d'une phrase à son verbe séparés par une proposition relative longue — devient pratiquement inaccessible à l'apprentissage.

Les LSTM atténuent ce problème grâce aux portes oubli (*forget gate*), entrée (*input gate*) et sortie (*output gate*) qui régulent le flux d'information dans une cellule mémoire $c_t$ distincte de l'état caché. Le gradient peut traverser les pas de temps en passant par cette cellule sans être multiplié systématiquement. Mais ce n'est qu'une atténuation partielle : pour des textes très longs ou des dépendances à très longue portée, les LSTM restent limités.

### 1.3 La séquentialité comme goulot d'étranglement

Le deuxième problème est d'ordre computationnel : les RNN sont fondamentalement séquentiels. Le calcul de $h_t$ dépend de $h_{t-1}$, qui dépend de $h_{t-2}$, etc. Il est impossible de paralléliser le traitement d'une séquence sur GPU. Cela a deux conséquences pratiques :

- L'entraînement sur de grands corpus est lent — le temps de calcul croît linéairement avec la longueur des séquences.
- L'inférence sur des textes longs souffre de la même contrainte.

Les Transformers, publiés en 2017 par Vaswani et al. dans l'article fondateur "*Attention Is All You Need*", résolvent ces deux problèmes simultanément. Ils abandonnent la récurrence au profit d'un mécanisme d'attention qui calcule directement les dépendances entre tous les tokens d'une séquence, en une seule opération matricielle entièrement parallélisable.

---

## 2. Le mécanisme d'attention : queries, keys, values

### 2.1 Intuition

Avant d'écrire la formule, construisons l'intuition. Imaginez que vous lisez la phrase : *"Le roi signa l'acte en son palais royal."* Pour comprendre à quoi se réfère *"son"*, vous devez établir une connexion entre le pronom et *"roi"*. Ce processus de référencement est précisément ce que l'attention modélise.

Dans le mécanisme d'attention, chaque token de la séquence pose une **question** (query) à tous les autres tokens. Chaque token expose une **clé** (key) — une signature permettant d'évaluer à quel point il répond à cette question — et une **valeur** (value) — le contenu informatif qu'il est prêt à partager si sa clé est pertinente.

La sortie pour un token donné est une combinaison pondérée des valeurs de tous les autres tokens, où les poids sont déterminés par la compatibilité entre sa query et leurs keys.

### 2.2 Scaled Dot-Product Attention : la formule

Formellement, pour une séquence de $n$ tokens représentés par des vecteurs de dimension $d_{model}$, on projette chaque token dans trois espaces distincts à l'aide de matrices de paramètres apprenables :

$$Q = X W^Q, \quad K = X W^K, \quad V = X W^V$$

où $X \in \mathbb{R}^{n \times d_{model}}$, et $W^Q, W^K \in \mathbb{R}^{d_{model} \times d_k}$, $W^V \in \mathbb{R}^{d_{model} \times d_v}$.

La matrice d'attention est alors calculée comme :

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

Décomposons chaque terme :

**Le produit scalaire $QK^\top$** produit une matrice $n \times n$ où l'élément $(i, j)$ mesure la similarité entre la query du token $i$ et la key du token $j$. Un produit scalaire élevé signifie que le token $i$ "accorde de l'attention" au token $j$.

**La division par $\sqrt{d_k}$** est un facteur d'échelle crucial. Sans ce facteur, quand $d_k$ est grand (typiquement 64 ou 128), les produits scalaires peuvent atteindre des valeurs très élevées, poussant la softmax vers des régions à gradient quasi nul. Le facteur $\sqrt{d_k}$ maintient les valeurs dans une plage raisonnable. C'est l'origine du terme "scaled" dans le nom de l'opération.

**La softmax** convertit les scores en poids d'attention qui somment à 1 :

$$\alpha_{ij} = \frac{\exp(q_i \cdot k_j / \sqrt{d_k})}{\sum_{l=1}^{n} \exp(q_i \cdot k_l / \sqrt{d_k})}$$

**La multiplication par $V$** produit finalement, pour chaque token $i$, une somme pondérée des valeurs de tous les tokens de la séquence :

$$\text{out}_i = \sum_{j=1}^{n} \alpha_{ij} \, v_j$$

La complexité de cette opération est $O(n^2 d_k)$ en temps et $O(n^2)$ en mémoire — le coût quadratique en longueur de séquence est la principale limitation des Transformers pour les très longs documents.

### 2.3 Illustration : la matrice d'attention

```
Phrase : "Le roi signa l'acte en son palais royal."
Tokens  :  Le  roi  signa  l'  acte  en  son  palais  royal

         Le   roi  signa   l'  acte   en   son  palais  royal
Le      [0.1  0.1   0.1   0.1  0.1   0.1  0.1   0.1    0.1 ]
roi     [0.0  0.8   0.0   0.0  0.1   0.0  0.0   0.0    0.1 ]
signa   [0.0  0.4   0.1   0.0  0.3   0.0  0.0   0.0    0.0 ]  ← attention sur "roi" et "acte"
...
son     [0.0  0.7   0.0   0.0  0.0   0.0  0.1   0.1    0.0 ]  ← résolution de coréférence
palais  [0.0  0.0   0.0   0.0  0.0   0.1  0.0   0.2    0.4 ]
```

La ligne "son" montre que le modèle apprend à connecter le pronom possessif à "roi" — c'est la résolution de coréférence émergente par attention.

### 2.4 Exemple de code

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q, K : (batch, n_heads, seq_len, d_k)
    V    : (batch, n_heads, seq_len, d_v)
    """
    d_k = Q.size(-1)
    # (batch, n_heads, seq_len, seq_len)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attn_weights = F.softmax(scores, dim=-1)   # somme à 1 sur l'axe des tokens
    output = torch.matmul(attn_weights, V)     # (batch, n_heads, seq_len, d_v)
    return output, attn_weights
```

---

## 3. Multi-Head Attention : pourquoi plusieurs têtes

### 3.1 Le problème d'une seule tête

Une seule tête d'attention calcule une relation moyenne sur l'ensemble de l'espace de représentation. Or, une phrase entretient simultanément plusieurs types de relations : syntaxiques (accord sujet-verbe), sémantiques (coréférence), discursives (cohésion thématique). Un seul ensemble de paramètres $(W^Q, W^K, W^V)$ ne peut capturer ces relations hétérogènes simultanément.

### 3.2 Multi-Head Attention

La solution de Vaswani et al. est d'exécuter $h$ opérations d'attention en parallèle, chacune dans un sous-espace de dimension réduite :

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \, W^O$$

$$\text{head}_i = \text{Attention}(Q W_i^Q,\; K W_i^K,\; V W_i^V)$$

où $W_i^Q \in \mathbb{R}^{d_{model} \times d_k}$, $W_i^K \in \mathbb{R}^{d_{model} \times d_k}$, $W_i^V \in \mathbb{R}^{d_{model} \times d_v}$, $W^O \in \mathbb{R}^{h d_v \times d_{model}}$.

Dans BERT-base : $d_{model} = 768$, $h = 12$, $d_k = d_v = 64$. La dimension totale reste $h \times d_v = 768$.

### 3.3 Spécialisation des têtes

Les études de probing (Clark et al., 2019 ; Voita et al., 2019) ont montré empiriquement que différentes têtes se spécialisent :

- Certaines têtes capturent des relations syntaxiques (dépendance objet-verbe).
- D'autres résolvent les coréférences pronominales.
- D'autres encore capturent la position relative (tokens adjacents).
- Des têtes dans les couches profondes capturent des relations sémantiques abstraites.

Cette spécialisation n'est pas imposée — elle émerge de l'entraînement. C'est ce qui rend les Transformers si puissants et si difficiles à interpréter.

### 3.4 Lien direct avec LoRA

Ce point est crucial pour la suite du module. Lorsque vous appliquerez LoRA, vous choisirez quelles matrices de projection adapter. Les projections $W^Q$ et $W^V$ sont les cibles préférentielles parce que :

- $W^Q$ contrôle ce que chaque token cherche dans le contexte — adapter $W^Q$ permet de réorienter l'attention vers des patterns pertinents pour votre tâche.
- $W^V$ contrôle le contenu récupéré — adapter $W^V$ change ce qui est propagé aux couches suivantes.
- $W^K$ joue un rôle plus passif de comparaison — les gains à l'adapter sont souvent marginaux par rapport au coût en paramètres.

Ne pas adapter $W^K$ n'est donc pas un choix arbitraire : c'est un choix architectural justifié par la compréhension des rôles respectifs des projections dans le mécanisme d'attention.

---

## 4. L'encodage positionnel : comment un modèle sans récurrence connaît l'ordre

### 4.1 Un problème fondamental

L'attention est une opération invariante par permutation : si vous mélangez les tokens d'une séquence, la matrice $QK^\top$ sera différente, mais le mécanisme d'attention lui-même ne "sait" pas dans quel ordre les tokens se trouvaient. Il faut injecter explicitement l'information de position.

### 4.2 Encodage sinusoïdal (Vaswani et al., 2017)

La solution originale de Vaswani et al. est d'additionner à chaque embedding de token un vecteur de position $PE_t$ dont les composantes sont des sinusoïdes de fréquences différentes :

$$PE_{t, 2i} = \sin\!\left(\frac{t}{10000^{2i/d_{model}}}\right)$$

$$PE_{t, 2i+1} = \cos\!\left(\frac{t}{10000^{2i/d_{model}}}\right)$$

où $t$ est la position du token et $i$ est l'indice de la dimension. Les hautes fréquences encodent la position fine (parité), les basses fréquences encodent la position globale.

Propriété clé : pour n'importe quel décalage $\delta$ fixé, $PE_{t+\delta}$ peut s'exprimer comme une transformation linéaire de $PE_t$. Cela permet au modèle de raisonner sur les distances relatives entre tokens.

### 4.3 RoPE — Rotary Position Embedding

RoPE (Su et al., 2021), utilisé dans LLaMA, Mistral et de nombreux modèles récents, est une approche plus élégante. Plutôt qu'd'additionner un vecteur de position aux embeddings, RoPE fait **pivoter** les vecteurs de query et de key d'un angle proportionnel à leur position avant le calcul d'attention :

$$q_t' = R_t \, q_t, \quad k_s' = R_s \, k_s$$

où $R_t$ est une matrice de rotation d'angle $t\theta$ pour chaque paire de dimensions. Le produit scalaire $q_t' \cdot k_s'$ dépend alors uniquement de la différence de position $t - s$ :

$$q_t' \cdot k_s' = q_t^\top R_{t-s} \, k_s$$

RoPE a deux avantages pratiques majeurs : il est compatible avec une extrapolation à des longueurs de séquence supérieures à celles vues à l'entraînement (avec des variantes comme *positional interpolation*), et il préserve naturellement la notion de distance relative — clé pour comprendre l'ordre syntaxique.

### 4.4 ALiBi — Attention with Linear Biases

ALiBi (Press et al., 2022) adopte une approche encore différente : plutôt que d'encoder la position dans les embeddings, il ajoute un biais négatif linéaire à la matrice de scores d'attention, proportionnel à la distance entre les tokens :

$$\text{score}_{ij} = q_i \cdot k_j / \sqrt{d_k} - m \cdot |i - j|$$

où $m$ est une pente spécifique à chaque tête d'attention. Les tokens éloignés reçoivent un biais négatif croissant, biaisant naturellement l'attention vers les tokens proches. ALiBi généralise mieux à des séquences plus longues que celles vues à l'entraînement.

**Pour votre corpus :** ces distinctions auront de l'importance au Jour 2. Les modèles utilisant RoPE ou ALiBi gèrent mieux l'extrapolation de longueur — pertinent pour des textes médiévaux dont les unités textuelles (chartes, registres) ont des longueurs très variables.

---

## 5. Architecture complète du Transformer

### 5.1 Le bloc Transformer

L'architecture Transformer empile $N$ blocs identiques. Chaque bloc contient :

1. **Multi-Head Self-Attention** avec connexion résiduelle et normalisation de couche (*Layer Norm*).
2. **Feed-Forward Network** (FFN) avec connexion résiduelle et Layer Norm.

$$\text{out}_1 = \text{LayerNorm}(x + \text{MultiHead}(x, x, x))$$
$$\text{out}_2 = \text{LayerNorm}(\text{out}_1 + \text{FFN}(\text{out}_1))$$

Le FFN est une simple réseau à deux couches avec activation GELU ou ReLU :

$$\text{FFN}(x) = W_2 \cdot \text{GELU}(W_1 x + b_1) + b_2$$

avec $d_{ff}$ généralement 4 fois plus grand que $d_{model}$ (3072 pour BERT-base).

La connexion résiduelle (He et al., 2016) est essentielle : elle crée un chemin de gradient direct entre les couches, évitant le vanishing gradient que l'on cherchait justement à fuir. La Layer Norm stabilise les activations.

### 5.2 Visualisation de l'architecture complète

```
 ENCODER BLOCK (×N)                    DECODER BLOCK (×N)
┌─────────────────────────┐           ┌─────────────────────────────┐
│  Input Embeddings       │           │  Output Embeddings (décalés)│
│  + Positional Encoding  │           │  + Positional Encoding      │
└──────────┬──────────────┘           └──────────┬──────────────────┘
           │                                     │
┌──────────▼──────────────┐           ┌──────────▼──────────────────┐
│  Multi-Head             │           │  Masked Multi-Head          │
│  Self-Attention         │           │  Self-Attention             │
│  + Add & Norm           │           │  + Add & Norm               │
└──────────┬──────────────┘           └──────────┬──────────────────┘
           │                                     │ ← (K, V depuis encoder)
┌──────────▼──────────────┐           ┌──────────▼──────────────────┐
│  Feed-Forward Network   │           │  Cross-Attention            │
│  + Add & Norm           │           │  (Q depuis decoder,         │
└──────────┬──────────────┘           │   K, V depuis encoder)      │
           │                          │  + Add & Norm               │
    (vers couche N+1)                 └──────────┬──────────────────┘
                                                 │
                                      ┌──────────▼──────────────────┐
                                      │  Feed-Forward Network       │
                                      │  + Add & Norm               │
                                      └──────────┬──────────────────┘
                                          (vers couche N+1)
```

---

## 6. Trois familles architecturales et leurs usages

La compréhension des différentes familles de Transformers est directement opérationnelle pour vos choix dans ce module. Ce n'est pas une taxonomie abstraite — c'est la raison pour laquelle vous utiliserez CamemBERT pour la NER et T5/mT5 pour la normalisation orthographique.

### 6.1 Encoder-only : BERT et ses descendants

**Principe :** L'encodeur transforme une séquence d'entrée en représentations contextualisées. Chaque token "voit" tous les autres tokens (attention bidirectionnelle). Il n'y a pas de décodeur, pas de génération.

**Pré-entraînement :** BERT est pré-entraîné sur deux objectifs :
- *Masked Language Modeling* (MLM) : 15 % des tokens sont masqués, le modèle prédit les tokens manquants à partir du contexte bidirectionnel.
- *Next Sentence Prediction* (NSP) : le modèle prédit si deux phrases sont consécutives.

**Usages idéaux :** Classification de tokens (NER, POS-tagging), classification de séquences (sentiment, topic), extraction de features, question-answering extractif.

**Pour votre module :** CamemBERT (Martin et al., 2020) est un BERT entraîné sur 138 Go de texte français (OSCAR corpus). Vous l'utiliserez pour la NER au Jour 3. L'attention bidirectionnelle est précieuse pour la reconnaissance d'entités : pour décider si *"Guillaume"* est un PER ou un TITLE, le contexte gauche ET droit est nécessaire.

**Modèles notables :** BERT, RoBERTa, DeBERTa, CamemBERT, XLM-RoBERTa.

### 6.2 Decoder-only : GPT et la famille des modèles génératifs

**Principe :** Le décodeur génère une séquence token par token, de gauche à droite. L'attention est causale (*causal* ou *autoregressive*) : le token $t$ ne peut "voir" que les tokens $1, \ldots, t-1$.

Le masque causal est appliqué sur la matrice d'attention :

$$M_{ij} = \begin{cases} 0 & \text{si } j \leq i \\ -\infty & \text{si } j > i \end{cases}$$

Ce masque garantit que la softmax attribue un poids nul aux tokens futurs.

**Pré-entraînement :** *Next Token Prediction* — prédire le prochain token étant donné tous les précédents. Objectif simple, dataset massif, comportements émergents surprenants.

**Usages idéaux :** Génération de texte, complétion, dialogue, few-shot prompting.

**Modèles notables :** GPT-2/3/4, LLaMA, Mistral, Falcon.

### 6.3 Encoder-decoder : T5, BART, mT5

**Principe :** L'encodeur traite l'entrée de façon bidirectionnelle et produit des représentations. Le décodeur génère la sortie token par token en combinant sa propre attention causale (self-attention masquée) et une **attention croisée** (*cross-attention*) sur les représentations de l'encodeur.

Dans la cross-attention, les queries viennent du décodeur et les keys/values viennent de l'encodeur :

$$\text{CrossAttention}(Q_{dec}, K_{enc}, V_{enc})$$

Cela permet au décodeur, à chaque étape de génération, de "consulter" n'importe quelle partie de la séquence d'entrée.

**Usages idéaux :** Toute tâche de transduction : traduction, résumé, normalisation orthographique, correction d'erreurs, question-answering génératif.

**Pour votre module :** mT5 (Xue et al., 2021) sera votre modèle de base pour la normalisation orthographique du moyen français au Jour 2. La tâche est naturellement formulée comme une transduction : token brut → forme normalisée. L'encodeur lit la forme brute, le décodeur génère la forme normalisée.

**Modèles notables :** T5, mT5, BART, mBART, PEGASUS.

### 6.4 Tableau de synthèse

| Famille | Attention | Tâches typiques | Modèles | Usage dans ce module |
|---|---|---|---|---|
| Encoder-only | Bidirectionnelle | NER, classification, extraction | BERT, CamemBERT | NER médiévale (Jour 3) |
| Decoder-only | Causale | Génération, complétion | GPT, LLaMA | Prompting (Jour 4) |
| Encoder-decoder | Bidir. + Cross | Traduction, résumé, normalisation | T5, mT5, BART | Normalisation (Jour 2) |

---

## 7. Tokenisation : BPE, SentencePiece et les enjeux pour le moyen français

### 7.1 Pourquoi tokeniser ?

Les Transformers opèrent sur des tokens discrets, pas sur des caractères bruts. La tokenisation est le pont entre texte brut et vocabulaire fini. Elle a des conséquences directes sur la qualité de représentation : un token bien formé (un mot complet ou un sous-mot fréquent) sera mieux représenté qu'un token résiduel fragmenté en sous-sous-mots isolés.

### 7.2 Byte Pair Encoding (BPE)

BPE (Sennrich et al., 2016) est l'algorithme de tokenisation le plus répandu. L'idée est simple :

1. Initialiser le vocabulaire avec tous les caractères du corpus.
2. Compter les paires de symboles adjacents les plus fréquentes.
3. Fusionner la paire la plus fréquente en un nouveau symbole.
4. Répéter jusqu'à atteindre la taille de vocabulaire cible.

Résultat : les mots fréquents ont tendance à être représentés par des tokens entiers, les mots rares sont décomposés en sous-mots plus fréquents. Tout token inconnu peut être encodé (à l'exception des caractères absents du vocabulaire), mais un mot inconnu peut nécessiter de nombreux tokens.

**Exemple simple :**
```
Corpus initial : "low lower lowest new newer newest"
Paires initiales : l-o, o-w, w-e, e-r, e-s, s-t, n-e, ...
Fusion 1 : (e, r) → er  [paire la plus fréquente]
Fusion 2 : (lo, w) → low
...
Résultat : "low", "er", "est", "new" sont des tokens entiers
```

### 7.3 SentencePiece et Unigram LM

SentencePiece (Kudo & Richardson, 2018) est un framework qui peut implémenter BPE mais aussi le modèle Unigram, utilisé par CamemBERT et T5. Le modèle Unigram part d'un grand vocabulaire initial et supprime itérativement les tokens qui minimisent la perte de log-vraisemblance sur le corpus.

Différence clé avec BPE : SentencePiece traite l'espace comme un caractère ordinaire (représenté par `▁`), permettant une tokenisation indépendante des espaces — utile pour les langues sans espaces ou pour les textes avec espacement irrégulier.

### 7.4 Le problème du moyen français

Voici le défi central de votre module sur le plan de la tokenisation. Les tokeniseurs de CamemBERT et RoBERTa ont été entraînés sur du français moderne (OSCAR, Wikipédia, presse). Le moyen français présente des caractéristiques qui le rendent profondément étranger à ces vocabulaires :

**Graphies non standardisées :** Une même forme peut apparaître sous de nombreuses variantes graphiques dans un même document. *roi* s'écrit aussi *roy*, *roÿ*, *roys*, *roy*, *rei*. *que* s'écrit *que*, *q~*, *qe*, *qu*. Ces variantes sont systématiquement absentes du vocabulaire BPE du français moderne.

**Abréviations manuscrites :** Les scribes médiévaux utilisent massivement des abréviations : tilde de nasalité (`q~` pour *que*, `pñ` pour *prison*), signes suprascripts, lettres spéciales. Ces caractères sont souvent absents du vocabulaire Unicode standard des tokeniseurs modernes.

**Agglutination et contractions :** Des formes comme *dou* (= *du*), *au* (= *à le*), *ès* (= *en les*) sont parfois tokenisées comme un token entier en français moderne mais correspondent à des structures différentes en moyen français.

**Conséquence mesurable — le taux OOV :**

Le taux de tokens hors-vocabulaire (Out-of-Vocabulary, OOV) sur un corpus de moyen français avec un tokeniseur CamemBERT est typiquement 2 à 5 fois supérieur à celui observé sur du français moderne. Sur le corpus CREMMA Medieval, des études pilotes rapportent des taux OOV de l'ordre de 15 à 30% selon les genres de documents (chartes vs romans).

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("almanach/camembert-base")

# Texte en français moderne vs moyen français
texte_moderne = "Le roi signa l'acte dans son palais."
texte_medieval = "Li roys signa l'acte en son paloys roial."

tokens_m = tokenizer.tokenize(texte_moderne)
tokens_med = tokenizer.tokenize(texte_medieval)

print("Français moderne  :", tokens_m)
# ['▁Le', '▁roi', '▁signa', '▁l', "'", 'acte', '▁dans', '▁son', '▁palais', '.']

print("Moyen français    :", tokens_med)
# ['▁Li', '▁ro', 'ys', '▁signa', '▁l', "'", 'acte', '▁en', '▁son', '▁pa', 'lo', 'ys', '▁ro', 'ial', '.']
# → "roys" découpé en 2 tokens, "paloys" en 3, "roial" en 2
```

Cette fragmentation a des conséquences directes sur la qualité des représentations :

1. Un token fragmenté (`▁ro`, `ys`) ne bénéficie pas de l'embedding pré-entraîné du mot entier.
2. Le modèle doit "réapprendre" à composer la signification de ces sous-mots au lieu d'exploiter des représentations déjà riches.
3. En NER, les entités fragmentées sont plus difficiles à détecter (un nom propre médiéval peut être découpé en 4-5 sous-tokens).

**Point de vigilance du syllabus :** Avant de choisir votre modèle de base, vous devez quantifier ce taux OOV sur votre corpus spécifique. Ce sera l'une des premières analyses de l'EDA du Jour 1. Ne choisissez pas CamemBERT "par défaut" — choisissez-le en sachant exactement quel est son coût en coverage.

---

## 8. Cas pratique : visualiser les têtes d'attention sur un texte médiéval

### 8.1 Pourquoi visualiser l'attention ?

La visualisation des poids d'attention est une fenêtre partielle — et parfois trompeuse — sur ce que le modèle apprend. Elle reste pédagogiquement utile pour :

- Vérifier que le modèle capte des relations linguistiques sensées.
- Identifier des pathologies (têtes entropiques qui distribuent l'attention uniformément).
- Motiver les choix d'interprétabilité pour un rapport à des humanistes.

Une mise en garde fondamentale (Jain & Wallace, 2019) : les poids d'attention ne sont pas des explications causales. Un poids d'attention élevé entre deux tokens ne signifie pas que l'un *cause* l'interprétation de l'autre — d'autres mécanismes (connexions résiduelles, FFN) contribuent également.

### 8.2 Pipeline de visualisation

```python
import torch
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def visualize_attention(text, model_name="almanach/camembert-base",
                        layer=11, head=0):
    """
    Visualise la matrice d'attention d'une tête donnée
    pour un texte en entrée.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_attentions=True)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    with torch.no_grad():
        outputs = model(**inputs)

    # outputs.attentions : tuple de (batch, n_heads, seq, seq) par couche
    attn = outputs.attentions[layer][0, head].numpy()  # (seq, seq)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(attn, cmap="Blues", vmin=0, vmax=attn.max())
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(tokens, fontsize=9)
    ax.set_title(f"Couche {layer+1}, Tête {head+1}", fontsize=12)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(f"attention_layer{layer}_head{head}.pdf", dpi=150)
    plt.show()

# Application sur un fragment de texte médiéval
text_medieval = "Li roys de France et le conte de Champagne firent accord."
visualize_attention(text_medieval, layer=11, head=3)
```

### 8.3 Ce qu'observer

Lors de votre TP, concentrez-vous sur :

- **Têtes dans les premières couches (1-4) :** souvent, attention aux tokens adjacents et aux ponctuations. Rôle syntaxique local.
- **Têtes dans les couches intermédiaires (5-8) :** attention plus diffuse, captant des relations à moyenne portée.
- **Têtes dans les dernières couches (9-12) :** relations sémantiques abstraites, coréférence, structure argumentale.

Pour un texte médiéval, comparez avec un texte français moderne équivalent : les différences de patterns d'attention reflètent la désorientation du modèle face à une graphie non vue à l'entraînement.

---

## 9. Connexion aux décisions du module

### 9.1 Récapitulatif des liens architecture → choix pratiques

| Question du module | Réponse architecturale |
|---|---|
| Pourquoi CamemBERT pour NER ? | Attention bidirectionnelle nécessaire pour classification de tokens |
| Pourquoi mT5 pour normalisation ? | Cross-attention encoder-decoder pour transduction de séquences |
| Pourquoi LoRA sur Q et V ? | Q contrôle ce que l'on cherche, V contrôle ce que l'on récupère |
| Pourquoi r=8 souvent suffisant ? | Le rang intrinsèque des mises à jour de fine-tuning est faible |
| Pourquoi quantifier l'OOV d'abord ? | Un taux OOV élevé dégrade la qualité des représentations pré-entraînées |
| Pourquoi RoPE extrapolation ? | Distance relative encodée dans l'angle de rotation, pas la position absolue |

### 9.2 Ce que vous verrez au Jour 2

Lorsque vous choisirez r=8 ou r=16 pour votre expérience d'ablation LoRA, vous saurez que r contrôle la dimension du sous-espace de mise à jour que vous autorisez. Un r faible fait l'hypothèse que les ajustements nécessaires pour votre tâche (normalisation du moyen français) vivent dans un espace de faible dimension dans l'espace des paramètres. Cette hypothèse est souvent vérifiée empiriquement — et vous la testerez directement avec vos données.

---

## Bibliographie de référence

### Articles fondateurs

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). **Attention is All You Need**. *Advances in Neural Information Processing Systems*, 30. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**. *Proceedings of NAACL-HLT 2019*. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)

### Encodage positionnel

Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2021). **RoFormer: Enhanced Transformer with Rotary Position Embedding**. [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)

Press, O., Smith, N. A., & Lewis, M. (2022). **Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation**. *ICLR 2022*. [arXiv:2108.12409](https://arxiv.org/abs/2108.12409)

### Architectures encoder-decoder et modèles multilingues

Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., & Liu, P. J. (2020). **Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer** (T5). *JMLR*, 21(140). [arXiv:1910.10683](https://arxiv.org/abs/1910.10683)

Xue, L., Constant, N., Roberts, A., Kale, M., Al-Rfou, R., Siddhant, A., Barua, A., & Raffel, C. (2021). **mT5: A Massively Multilingual Pre-Trained Text-to-Text Transformer**. *Proceedings of NAACL 2021*. [arXiv:2010.11934](https://arxiv.org/abs/2010.11934)

Lewis, M., Liu, Y., Goyal, N., Ghazvininejad, M., Mohamed, A., Levy, O., Stoyanov, V., & Zettlemoyer, L. (2020). **BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension**. *ACL 2020*. [arXiv:1910.13461](https://arxiv.org/abs/1910.13461)

### Tokenisation

Sennrich, R., Haddow, B., & Birch, A. (2016). **Neural Machine Translation of Rare Words with Subword Units** (BPE). *ACL 2016*. [arXiv:1508.07909](https://arxiv.org/abs/1508.07909)

Kudo, T., & Richardson, J. (2018). **SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing**. *EMNLP 2018 System Demonstrations*. [arXiv:1808.06226](https://arxiv.org/abs/1808.06226)

### CamemBERT et NLP pour le français

Martin, L., Muller, B., Suárez, P. J. O., Dupont, Y., Romary, L., de la Clergerie, É. V., Seddah, D., & Sagot, B. (2020). **CamemBERT: a Tasty French Language Model**. *ACL 2020*. [arXiv:1911.03894](https://arxiv.org/abs/1911.03894)

### Interprétabilité des têtes d'attention

Clark, K., Khandelwal, U., Levy, O., & Manning, C. D. (2019). **What Does BERT Look at? An Analysis of BERT's Attention**. *BlackboxNLP Workshop, ACL 2019*. [arXiv:1906.04341](https://arxiv.org/abs/1906.04341)

Voita, E., Talbot, D., Moiseev, F., Sennrich, R., & Titov, I. (2019). **Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned**. *ACL 2019*. [arXiv:1905.09418](https://arxiv.org/abs/1905.09418)

Jain, S., & Wallace, B. C. (2019). **Attention is not Explanation**. *NAACL 2019*. [arXiv:1902.10186](https://arxiv.org/abs/1902.10186)

### Fine-tuning efficace (PEFT/LoRA — anticipation Jour 2)

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). **LoRA: Low-Rank Adaptation of Large Language Models**. *ICLR 2022*. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)

Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). **QLoRA: Efficient Finetuning of Quantized LLMs**. *NeurIPS 2023*. [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)

### NLP pour les langues historiques et le vieux/moyen français

Clerice, T. (2023). **Pie Extended, an package for historical language models, beyond LatinCY**. [Zenodo](https://doi.org/10.5281/zenodo.3883589)

Camps, J.-B., Vinsonneau, A., & Clérice, T. (2021). **Corpus and Models for Lemmatisation and POS-tagging of Old French**. *Journal of Data Mining & Digital Humanities*.

Pinche, A. (2022). **CREMMA Medieval : corpus de manuscrits médiévaux pour HTR**. [GitHub : HTR-United/CREMMA-Medieval](https://github.com/HTR-United/CREMMA-Medieval)

Tittel, S., Bermudez-Sabel, H., & Chiarcos, C. (2020). **Using RDFa to Link Text and Dictionary Data for Medieval French**. *Proceedings of the Linked Data in Linguistics Workshop, LREC 2020*.

### Manuels et références pédagogiques

Jurafsky, D., & Martin, J. H. (2024). *Speech and Language Processing* (3e éd., brouillon). Chapitres 9–11. [Disponible en ligne](https://web.stanford.edu/~jurafsky/slp3/)

Tunstall, L., von Werra, L., & Wolf, T. (2022). *Natural Language Processing with Transformers*. O'Reilly Media.

Rush, A. (2018). **The Annotated Transformer**. *EMNLP 2018 Tutorial*. [Disponible en ligne](http://nlp.seas.harvard.edu/annotated-transformer/)

---

*Support de cours rédigé pour le Master Data/IA · Module NLP · MD5 Volet 2 · 2026. Ce document est destiné à accompagner le cours magistral du Jour 1 et doit être lu en parallèle avec les notebooks de TP.*
