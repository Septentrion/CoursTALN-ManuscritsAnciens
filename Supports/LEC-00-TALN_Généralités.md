# La pile du traitement automatique des langues : de la phonologie à la rhétorique

**Document de référence — Module NLP · Master Data/IA · MD5 Volet 2 · 2026**

---

## Préambule : panorama historique de la linguistique et de la sémiotique

### Des Grecs aux néogrammairiens

La réflexion sur le langage naît en Grèce antique avec une question qui n'a pas encore trouvé de réponse définitive : le lien entre un mot et ce qu'il désigne est-il naturel ou conventionnel ? Platon, dans le *Cratyle* (vers 390 av. J.-C.), oppose la thèse naturaliste — les mots imitent la nature des choses — à la thèse conventionaliste — les mots sont des signes arbitraires établis par convention. Aristote tranche en faveur de la convention dans le *De Interpretatione* (vers 350 av. J.-C.) et pose les premières catégories grammaticales : nom, verbe, proposition.

La grammaire descriptive systématique commence avec Denys de Thrace, dont la *Techne grammatike* (vers 100 av. J.-C.) identifie huit parties du discours et reste la grammaire de référence pendant quinze siècles. En latin, Donat au IVe siècle et Priscien au VIe siècle adaptent ce cadre, produisant les manuels qui formeront les clercs médiévaux pendant mille ans.

Le Moyen Âge ajoute une dimension philosophique à l'analyse grammaticale. Les *modistes* (grammairiens *modi significandi*, XIIIe siècle) cherchent à démontrer que la grammaire est universelle parce que le langage reflète les structures de la réalité. Roger Bacon, dans son *De signis* (vers 1270), développe une théorie du signe qui anticipe certaines questions de la sémiotique moderne. La *Grammaire générale et raisonnée* de Port-Royal (Arnauld et Lancelot, 1660) hérite de cette ambition : montrer que les langues naturelles partagent une logique profonde universelle, thèse que Chomsky citera trois siècles plus tard comme ancêtre direct de son programme générativiste.

La linguistique au sens moderne — c'est-à-dire comme discipline scientifique empirique — naît avec la découverte de la parenté des langues indo-européennes. William Jones, en 1786, observe la ressemblance systématique entre le sanskrit, le grec, le latin et les langues germaniques, et pose l'hypothèse d'une origine commune. Franz Bopp fonde en 1816 la grammaire comparée avec son étude du système de conjugaison du sanskrit, du grec et du latin. La découverte que les langues évoluent selon des lois régulières — notamment la loi de Grimm (1822) sur les mutations consonantiques germaniques — transforme la linguistique en science historique.

Les néogrammairiens (*Junggrammatiker*), autour de Karl Brugmann et August Leskien dans les années 1870, formulent la thèse radicale que les lois phonétiques sont sans exception — que toute irrégularité apparente cache une règle plus générale ou une analogie. Cette affirmation programmatique est à la base de la phonologie moderne : elle implique que le changement linguistique est systématique et donc scientifiquement descriptible.

### Saussure et la révolution structuraliste

Ferdinand de Saussure tient ses cours de linguistique générale à Genève entre 1906 et 1911. Ses étudiants publient leurs notes en 1916, trois ans après sa mort, sous le titre *Cours de linguistique générale*. Ce livre refonde entièrement la discipline.

Saussure pose que la langue est un **système de différences** : un signe linguistique tire sa valeur non de ce qu'il désigne, mais de ce qu'il n'est pas. *Chat* vaut ce qu'il vaut parce qu'il n'est pas *chien*, *rat*, *bête*, ni *chat* en anglais. Cette conception relationnelle et différentielle est l'essence du structuralisme.

Deux distinctions saussuriennes traversent tout le TALN :

La distinction **langue / parole** — la langue est le système abstrait partagé par une communauté, la parole est l'acte individuel de production. Chomsky reprendra cette dichotomie sous les noms de *compétence* et *performance*. En TALN, le corpus n'est que de la parole ; les modèles de langage tentent d'approximer la langue.

La distinction **synchronie / diachronie** — l'étude de l'état d'une langue à un moment donné (synchronie) est indépendante de l'étude de son évolution dans le temps (diachronie). Pour un TALN appliqué à des manuscrits médiévaux, cette distinction est opérationnelle : le pipeline de normalisation est une opération de transformation diachronique (ramener une forme ancienne à une forme moderne), tandis que le modèle NER opère en synchronie sur le texte normalisé.

Saussure introduit aussi la notion **d'arbitraire du signe** : le lien entre le signifiant (image acoustique ou graphique) et le signifié (concept) est arbitraire — il n'y a aucune ressemblance naturelle entre le son [∫a] et l'animal *chat*. L'arbitraire est la condition de la variation interlinguistique : le même concept est désigné par des sons entièrement différents dans des langues différentes.

### Peirce et les fondements de la sémiotique

Parallèlement à Saussure, Charles Sanders Peirce développe aux États-Unis une théorie du signe d'une tout autre ampleur. Là où Saussure s'intéresse à la langue comme système, Peirce s'intéresse à la **sémiosis** — le processus général par lequel quelque chose (un signe) représente quelque chose d'autre (son objet) pour quelqu'un (l'interprétant).

La tripartition peircéenne des signes est l'une des classifications les plus utilisées en linguistique et en TALN :

Un **icône** ressemble à ce qu'il représente : un dessin d'oiseau est une icône d'oiseau, une onomatopée (*cocorico*) est partiellement iconique.

Un **indice** est causalement lié à son objet : la fumée est l'indice du feu, la fièvre est l'indice d'une infection. En TALN, les marqueurs de cohérence (*donc*, *cependant*, *en conséquence*) sont des indices de relations rhétoriques.

Un **symbole** est relié à son objet par convention : le mot *oiseau*, le drapeau français, l'icône de la corbeille informatique. La grande majorité des signes linguistiques sont des symboles au sens peircéen.

Charles Morris (1938) formalisera la sémiotique en trois branches : la **syntaxe** (relations des signes entre eux), la **sémantique** (relations des signes à leurs objets), et la **pragmatique** (relations des signes à leurs utilisateurs). Cette tripartition structure encore aujourd'hui la division du TALN en couches.

### Le XXe siècle : générativisme, sémantique formelle, pragmatique

Noam Chomsky publie *Syntactic Structures* en 1957 et bouleverse la linguistique américaine. Contre le béhaviorisme bloomfieldien qui refusait toute référence aux états mentaux, Chomsky réintroduit la compétence — la connaissance implicite et tacite de la grammaire — comme objet légitime de la linguistique. Sa thèse centrale est que la créativité du langage (la capacité à produire et comprendre des phrases jamais entendues) ne peut s'expliquer que par un système de règles récursives génératives.

La hiérarchie de Chomsky (1956) classifie les grammaires formelles par leur puissance expressive : les **grammaires régulières** (type 3, équivalentes aux automates finis) sont insuffisantes pour le langage naturel ; les **grammaires hors-contexte** (type 2, CFG) capturent la structure syntagmatique mais pas certaines dépendances longue-distance ; les **grammaires contextuelles** (type 1) et les systèmes encore plus puissants sont nécessaires pour des phénomènes comme la copie ou la réduplication dans certaines langues. Cette hiérarchie demeure le cadre de référence pour l'analyse de la complexité des langages en informatique théorique.

Richard Montague apporte en 1970 la rigueur de la logique formelle à la sémantique avec son programme de **grammaire universelle** : toute langue naturelle peut être traitée comme un langage formel, et le sens d'une phrase peut être calculé compositionnellement depuis le sens de ses parties via le λ-calcul. La **sémantique compositionnelle** de Montague reste le fondement des systèmes de compréhension du langage naturel les plus précis.

H.P. Grice apporte en 1975 la notion d'**implicature conversationnelle** : ce qu'un locuteur communique dépasse souvent ce qu'il dit littéralement. Si quelqu'un demande *"Peux-tu passer le sel ?"*, la réponse attendue n'est pas *"Oui"* mais l'action de passer le sel. Grice formalise les **maximes conversationnelles** — quantité, qualité, relation, manière — dont la violation délibérée produit des effets implicatifs. Ces maximes ont inspiré des systèmes de pragmatique computationnelle et l'analyse des dialogues.

J.L. Austin (1962) et John Searle (1969) développent la théorie des **actes de langage** : dire c'est faire. Une promesse, une menace, une question ne sont pas de simples descriptions du monde — elles accomplis des actes sociaux. La taxonomie des actes illocutoires de Searle (assertifs, directifs, commissifs, expressifs, déclaratifs) structure encore aujourd'hui les systèmes de dialogue et l'annotation des corpus de conversations.

### Du TALN classique aux grands modèles de langage

Le traitement automatique des langues naît formellement avec le test de Turing (1950) et la première démonstration de traduction automatique Georgetown-IBM (1954). Mais le rapport ALPAC (1966) dresse un bilan désastreux des deux premières décennies de MT et gèle le financement américain pour une décennie.

Le renouveau vient des approches symboliques rigoureuses des années 1970–1980 : SHRDLU de Winograd (1972), qui comprend le langage en contexte clos, les systèmes experts de compréhension du discours, et surtout les formalismes de représentation des connaissances. Les réseaux de neurones reviennent en linguistique computationnelle avec les travaux PDP de Rumelhart et McClelland (1986) sur la morphologie.

Les années 1990 voient le triomphe des approches statistiques : les modèles IBM de traduction (1990–1993), la modélisation n-gram du langage, et la constitution de grands corpus annotés — le Penn Treebank (1993) pour la syntaxe, PropBank pour les rôles sémantiques, CoNLL pour la NER. Les CRF de Lafferty et al. (2001) unifient l'étiquetage de séquences sur une base probabiliste solide.

L'apprentissage profond s'impose en TALN à partir de 2013 avec Word2Vec (Mikolov et al.), qui montre que les plongements vectoriels de mots capturent des relations sémantiques non triviales. L'architecture Transformer (Vaswani et al. 2017) révolutionne l'ensemble du domaine en éliminant la récurrence au profit du mécanisme d'attention. BERT (Devlin et al. 2018) introduit le pré-entraînement bidirectionnel massif ; GPT-3 (Brown et al. 2020) montre qu'un modèle suffisamment grand peut réaliser des tâches NLP arbitraires avec quelques exemples seulement — ce qui pose de façon aiguë la question de la relation entre performance statistique et compréhension linguistique réelle.

---

## Introduction : la pile NLP comme architecture en couches

Le traitement automatique des langues peut être vu comme une architecture en couches, chaque couche s'appuyant sur les résultats de la couche inférieure et produisant des structures plus abstraites. Cette organisation n'est pas arbitraire : elle reflète la structure du langage lui-même, telle que la linguistique l'a progressivement mise au jour depuis deux siècles.

La métaphore de la pile — *stack* en anglais — est courante en informatique : chaque couche fournit une interface à la couche supérieure et consomme les services de la couche inférieure. En TALN, la pile ressemble à ceci, de bas en haut :

```
┌─────────────────────────────────────────────────────┐
│ Rhétorique       figures, argumentation, persuasion │
├─────────────────────────────────────────────────────┤
│ Stylistique      registre, style, attribution       │
├─────────────────────────────────────────────────────┤
│ Analyse du discours   cohérence, coréférence        │
├─────────────────────────────────────────────────────┤
│ Pragmatique      actes de langage, implicatures     │
├─────────────────────────────────────────────────────┤
│ Sémantique       sens, référence, vérité            │
├─────────────────────────────────────────────────────┤
│ Syntaxe          structure, dépendances             │
├─────────────────────────────────────────────────────┤
│ Lexicologie      mots, lemmes, sens lexical         │
├─────────────────────────────────────────────────────┤
│ Morphologie      morphèmes, flexion, dérivation     │
├─────────────────────────────────────────────────────┤
│ Phonologie       sons, phonèmes, prosodie           │
└─────────────────────────────────────────────────────┘
```

Cette organisation est pédagogiquement utile, mais il faut immédiatement en souligner la limite : elle est en grande partie fictive d'un point de vue computationnel. Un modèle Transformer comme BERT traite simultanément des informations qui appartiennent à plusieurs couches de cette pile. La pragmatique peut résoudre des ambiguïtés syntaxiques avant même que la syntaxe soit complètement analysée. La prosodie peut signaler des frontières syntaxiques que la grammaire laisserait ambiguës.

La pile est donc mieux comprise comme un outil analytique que comme une description du traitement réel — humain ou machine. Elle sert à identifier les phénomènes, à associer les méthodes aux niveaux d'abstraction auxquels elles opèrent, et à comprendre comment les erreurs se propagent d'une couche à l'autre.

---

## Couche 1 — Phonologie : sons, phonèmes, prosodie

### Ce que la linguistique étudie

La phonologie s'intéresse à l'organisation des sons du langage en systèmes abstraits. Elle se distingue de la **phonétique** — qui étudie les propriétés physiques des sons (spectres acoustiques, configurations articulatoires) — en ce qu'elle s'intéresse aux contrastes fonctionnels : les distinctions sonores qui changent le sens.

Un **phonème** est la plus petite unité sonore distinctive d'une langue. En français, /p/ et /b/ sont deux phonèmes distincts parce que *pain* et *bain* sont deux mots de sens différents. La paire minimale — deux mots identiques à un segment près — est l'outil classique pour identifier les phonèmes.

Le **trait distinctif** (Jakobson, Halle et Fant, 1952) descend plus bas encore : chaque phonème peut être décomposé en un ensemble de traits binaires — voisé/non voisé, nasal/oral, occlusive/fricative, etc. /p/ = [–voisé, +occlusive, +labiale] ; /b/ = [+voisé, +occlusive, +labiale]. Cette décomposition rend compte des processus phonologiques réguliers : la nasalisation en français (*bon ami* → [bõ.na.mi]) s'explique par la propagation du trait [+nasal].

La **prosodie** étudie les unités suprasegmentales : l'accent, le ton, l'intonation, le rythme. En français, l'intonation montante transforme une assertion en question : *Il vient* vs *Il vient ?* sont le même segment phonémique avec deux contours intonatifs différents. En mandarin, le ton est lexical : les quatre tons du syllabe [ma] distinguent *mā* (mère), *má* (chanvre), *mǎ* (cheval) et *mà* (gronder).

### Connexion au TALN

En TALN écrit, la phonologie semble absente. Elle intervient pourtant de façon indirecte :

La **normalisation orthographique** du moyen français (Chapitres 3 et 4 du cours) repose sur des règles phonologiques : la règle *u/v* s'explique par le fait que [u] et [v] étaient deux graphies du même phonème en position initiale ou médiale. La règle sur les tildes de nasalité (*norm~die → normandie*) repose sur une règle phonologique : devant /b/ ou /p/, la nasale est [m] par assimilation de lieu d'articulation.

La **conversion graphème-phonème** (G2P) est nécessaire pour la synthèse vocale et la reconnaissance de la parole. Elle est implémentée soit par des règles de réécriture, soit par des automates à états finis.

La **segmentation prosodique** — identifier les frontières de groupes prosodiques dans un texte — aide à la segmentation syntaxique et à la génération de parole naturelle.

**Algorithmes classiques :**

Le modèle standard de G2P utilise des **transducteurs à états finis** (FST). Un FST lit une séquence de caractères (graphèmes) et produit une séquence de phonèmes. Kaplan et Kay (1994) montrent que la composition de règles de réécriture phonologiques peut être compilée en un seul FST, rendant l'application en temps linéaire. OpenFST (Allauzen et al. 2007) est la bibliothèque de référence.

Pour la reconnaissance vocale, les **modèles de Markov cachés** (HMM) ont longtemps dominé (Rabiner 1989) : chaque phonème est modélisé par un HMM dont les émissions sont des vecteurs de coefficients acoustiques MFCC. Depuis 2014, les réseaux récurrents puis les Transformers ont remplacé les HMM dans les systèmes de reconnaissance de la parole de pointe (Whisper, wav2vec 2.0).

**Exemple linguistique :** La liaison en français illustre un phénomène phonologique dépendant du contexte syntaxique. *Les amis* se prononce [le.za.mi] (liaison) mais *les | homards* peut se prononcer sans liaison selon le registre. La règle de liaison n'est pas purement phonologique : elle dépend de la frontière de constituant syntaxique entre le déterminant et le nom. C'est un exemple de la façon dont les couches interagissent : la phonologie a besoin de la syntaxe.

---

## Couche 2 — Morphologie : morphèmes, flexion, dérivation

### Ce que la linguistique étudie

Le **morphème** est la plus petite unité de sens ou de fonction grammaticale. Dans *chantions*, on distingue *chant-* (morphème lexical, porteur du sens verbal), *-i-* (morphème de mode — imparfait), *-ons* (morphème de personne/nombre — 1re personne du pluriel). Cette décomposition en morphèmes est au cœur de la morphologie.

La **morphologie flexionnelle** produit les formes d'un même lexème selon les catégories grammaticales : *chanter, chante, chantons, chanté, chantant*. Ces formes partagent le même lemme (*chanter*) et la même entrée dans le dictionnaire.

La **morphologie dérivationnelle** crée de nouveaux lexèmes par affixation : *chanter → chanteur → chanteuresque*, *possible → impossible → impossibilité*. Contrairement à la flexion, la dérivation change souvent la catégorie grammaticale et crée une nouvelle entrée lexicale.

Les langues varient énormément dans leur morphologie. Le français est une langue **fusionnelle** : un même suffixe exprime plusieurs catégories à la fois (*-ons* = 1re pers. + pluriel + présent ou imparfait selon le contexte). Le turc est **agglutinant** : chaque morphème exprime une seule catégorie et les morphèmes se concatènent de façon transparente. Le mandarin est **isolant** : les mots sont invariables et la grammaire est exprimée par l'ordre des mots plutôt que par la flexion.

### Connexion au TALN

La **lemmatisation** — ramener une forme fléchie à son lemme canonique — est l'opération morphologique centrale du TALN. *Chantaient* → *chanter*, *meilleurs* → *bon* (avec suppression du superlatif), *chevaux* → *cheval*. La lemmatisation est distincte du stemming (qui tronque le mot sans analyse morphologique) : le stemme de *chantaient* par l'algorithme de Porter est *chant*, pas *chanter*.

La **tokenisation** par sous-mots (BPE, WordPiece, Unigram LM) a révolutionné le TALN depuis 2016. Plutôt que de traiter les mots comme des unités atomiques, ces algorithmes découpent les mots en sous-mots fréquents. *Chantaient* peut être découpé en *chant* + *aient*, ou en *ch* + *ant* + *aient* selon la taille du vocabulaire. Ce découpage résout le problème des mots hors-vocabulaire (*out-of-vocabulary*) au prix d'une fragmentation des unités morphologiques.

**Algorithmes classiques :**

L'**algorithme de Porter** (Porter 1980) est le stemmer anglais le plus utilisé. Il applique cinq passes de règles de réécriture dans un ordre précis, supprimant les suffixes communs (*-ing*, *-ed*, *-tion*, etc.). Il est rapide (O(n) en longueur du mot) mais produit des stemmes qui ne sont pas toujours des mots réels.

La **morphologie à deux niveaux** de Koskenniemi (1983) utilise deux bandes parallèles — une pour les morphèmes sous-jacents, une pour les réalisations de surface — reliées par des règles de correspondance phonologiques. Elle est implémentée comme un transducteur à états finis et produit une analyse morphologique complète.

L'algorithme de **Byte Pair Encoding** (Sennrich et al. 2016) apprend un vocabulaire de sous-mots par fusion itérative : on commence avec les caractères individuels, on identifie la paire de symboles consécutifs la plus fréquente, on la fusionne en un symbole unique, et on répète jusqu'à atteindre la taille de vocabulaire cible. Appliqué à *un corpus normalisé de moyen français*, BPE fragmenterait *sénéchal* en *sén* + *échal* plutôt que *séné* + *chal* selon les fréquences observées.

**Exemple linguistique :** La *supplétion* est un phénomène morphologique où les formes d'un paradigme n'ont pas de racine commune : *bon / meilleur* (pas **bonner*), *aller / vais / irai*. Ces formes posent problème aux systèmes morphologiques basés sur des règles : elles doivent être listées explicitement, elles ne peuvent pas être dérivées par règles générales. Dans un système de lemmatisation, *suis* doit être identifié comme une forme de *être* — ce que seul un lexique annoté peut garantir.

---

## Couche 3 — Lexicologie : mots, lemmes, sens lexical

### Ce que la linguistique étudie

La **lexicologie** étudie le lexique — l'ensemble des mots d'une langue — dans ses dimensions formelle, sémantique et référentielle. Elle distingue la **lexicographie** (description et codification dans des dictionnaires) de la **sémantique lexicale** (étude des relations de sens entre les mots).

Les principales relations lexicales sont : la **synonymie** (*commencer* / *débuter*), l'**antonymie** (*chaud* / *froid*), l'**hyponymie** (*pinsон* est un hyponyme de *oiseau*), la **méronymie** (*moteur* est une méronyme de *voiture*). Ces relations constituent la structure d'un réseau lexical.

La **polysémie** — un même mot ayant plusieurs sens liés — est omniprésente : *canal* désigne un cours d'eau artificiel, un conduit anatomique, et une voie de télécommunication. Elle se distingue de l'**homonymie** — deux mots de formes identiques mais d'étymologies distinctes et de sens non liés : *louer* (prendre en location) et *louer* (faire l'éloge), deux lexèmes distincts liés par accident formel.

### Connexion au TALN

La **désambiguïsation lexicale** (Word Sense Disambiguation, WSD) est l'une des tâches fondamentales du TALN : déterminer lequel des sens d'un mot polysémique est actif dans un contexte donné. *Il a déposé la plainte à la banque* — *banque* désigne ici une institution financière, pas un banc ou un talus.

Les **plongements lexicaux** (word embeddings) représentent les mots comme des vecteurs denses dans un espace de faible dimension, de telle façon que les mots sémantiquement proches soient proches dans l'espace vectoriel. L'hypothèse distributionnelle de Harris (1954) sous-tend cette approche : *"A word is characterized by the company it keeps"* — les mots qui apparaissent dans des contextes similaires ont des sens similaires.

**Algorithmes classiques :**

La **TF-IDF** (Term Frequency-Inverse Document Frequency, Sparck Jones 1972) pondère l'importance d'un terme dans un document par rapport à sa fréquence dans l'ensemble du corpus. Un terme fréquent dans un document mais rare dans le corpus a une TF-IDF élevée — il est *caractéristique* de ce document. TF-IDF reste le baseline de référence pour la recherche d'information et la représentation de documents.

$$\text{TF-IDF}(t, d) = \text{tf}(t, d) \times \log\!\frac{N}{|\{d' : t \in d'\}|}$$

La **PMI** (Pointwise Mutual Information, Church & Hanks 1990) mesure l'association statistique entre deux mots :

$$\text{PMI}(w_1, w_2) = \log\frac{P(w_1, w_2)}{P(w_1) \cdot P(w_2)}$$

Une PMI élevée indique que les deux mots cooccurrent plus souvent qu'attendu par le hasard — signal d'une collocation ou d'une relation lexicale. La PMI positifve (PPMI) supprime les valeurs négatives peu fiables sur les corpus petits.

**Word2Vec** (Mikolov et al. 2013) apprend des représentations vectorielles denses par deux architectures : Skip-gram (prédire le contexte depuis le mot central) et CBOW (prédire le mot central depuis le contexte). La propriété la plus célèbre de Word2Vec est la linéarité des relations : *vec(roi) − vec(homme) + vec(femme) ≈ vec(reine)*. Cette arithmétique vectorielle capture des relations analogiques.

**WordNet** (Miller 1995) est une base lexicale hiérarchique qui organise les noms, verbes, adjectifs et adverbes en **synsets** (ensembles de synonymes) reliés par des relations lexicales. WordNet est la ressource lexicale de référence pour la WSD et l'inférence lexicale.

**Exemple linguistique :** La **métaphore conceptuelle** (Lakoff et Johnson 1980) est un phénomène lexical et cognitif : des domaines abstraits sont systématiquement structurés par des domaines concrets. La métaphore conceptuelle *ARGUMENT IS WAR* structure tout un réseau d'expressions : *défendre sa position*, *attaquer un argument*, *démolir une thèse*, *capituler devant les preuves*. Ces expressions ne sont pas des métaphores vives — elles sont lexicalisées et souvent inaperçues. En TALN, la détection de métaphores conceptuelles (Shutova et al. 2016) vise à identifier ces patterns pour améliorer la compréhension sémantique.

---

## Couche 4 — Syntaxe : structures, dépendances, arbres

### Ce que la linguistique étudie

La syntaxe étudie les règles qui régissent la combinaison des mots en phrases grammaticales. Deux grands paradigmes structurent le domaine.

La **grammaire des syntagmes** (phrase structure grammar, Chomsky 1957) décompose la phrase en constituants imbriqués : une phrase (S) se compose d'un syntagme nominal (NP) et d'un syntagme verbal (VP) ; un VP se compose d'un verbe (V) et optionnellement d'un NP, etc. L'arbre syntagmatique représente cette décomposition hiérarchique.

La **grammaire de dépendance** (Tesnière 1959) représente les relations entre les mots directement, sans constituants intermédiaires : chaque mot dépend d'un autre (son *tête* ou *gouverneur*), et la relation est étiquetée (sujet, objet, modificateur, etc.). Le graphe de dépendance est un arbre enraciné dont la racine est le verbe principal. Universal Dependencies (Nivre et al. 2016) a standardisé ces étiquettes sur 70 langues.

```
Phrase : "Le sénéchal porta les lettres au roi."

Arbre de dépendances (Universal Dependencies) :

        porta (VERB, racine)
       /      \        \
sénéchal    lettres    roi
(nsubj)      (obj)    (iobj)
    |           |
   Le          les
  (det)        (det)
                        |
                       au
                      (case)
```

La **valence** d'un verbe (Tesnière) est le nombre et la nature des compléments qu'il requiert : *dormir* est monovalent (un seul actant, le dormeur) ; *donner* est trivalent (un donateur, un donataire, une chose donnée). La valence est une propriété lexicale et syntaxique simultanément.

### Connexion au TALN

Le **parsing syntaxique** — construire l'arbre syntaxique d'une phrase — est une tâche fondamentale qui alimente l'extraction de relations, la sémantique compositionnelle, et la traduction automatique.

**Algorithmes classiques :**

L'**algorithme CYK** (Cocke-Younger-Kasami, Cocke 1969) est le parser bottom-up de référence pour les grammaires hors-contexte probabilistes (PCFG). Sa complexité est $O(n^3 \cdot |G|)$ où $n$ est la longueur de la phrase et $|G|$ le nombre de règles de la grammaire. Il opère sur une table triangulaire (chart) remplie de bas en haut.

L'**algorithme de Earley** (1970) est un parser top-down/bottom-up pour toute CFG (y compris ambiguë et récursive à gauche) avec une complexité $O(n^3)$ en général et $O(n^2)$ pour les grammaires non ambiguës.

Le **parsing de dépendances arc-eager** (Nivre 2003) est un parser à déplacement-réduction (shift-reduce) linéaire en temps ($O(n)$). À chaque étape, il choisit parmi quatre actions : *shift* (empiler le mot suivant), *left-arc* (créer un arc de dépendance du sommet de pile vers le mot courant), *right-arc* (créer un arc en sens inverse), *reduce* (dépiler). Le choix est guidé par un classifieur entraîné sur des corpus annotés.

Le **biaffine parser** (Dozat et Manning 2017) est l'architecture neurale de référence pour le parsing de dépendances. Il encode chaque token par un réseau BiLSTM ou un Transformer, puis calcule le score de chaque arc potentiel $(i, j, r)$ par un produit bilinéaire — d'où son nom. Il atteint les meilleures performances sur la plupart des langues de Universal Dependencies.

**Exemple linguistique :** La **montée du clitique** en français est un phénomène syntaxique où un clitique objet monte d'une proposition infinitive enchâssée vers le verbe principal : *Je veux le voir* (le monte de la proposition *voir le*). Ce phénomène pose un défi aux parsers qui ne modélisent pas les mouvements à longue distance. L'analyse en dépendances le représente par un arc direct du pronom *le* au verbe *veux*, avec la relation `obj`.

---

## Couche 5 — Sémantique : sens, référence, vérité

### Ce que la linguistique étudie

La sémantique s'intéresse au sens — ce que les expressions linguistiques signifient, indépendamment de leur usage en contexte (qui relève de la pragmatique).

La **sémantique référentielle** (Frege 1892) distingue le **sens** (*Sinn*) — le mode de présentation — de la **dénotation** (*Bedeutung*) — l'objet dans le monde. *L'étoile du matin* et *l'étoile du soir* ont des sens différents mais dénotent le même objet (Vénus). Cette distinction est cruciale pour la résolution de coréférence : deux expressions de sens différents peuvent référer au même entité.

La **sémantique compositionnelle** de Montague (1970) exprime le sens d'une phrase comme une fonction de son sens de ses constituants, calculée par le λ-calcul. Le sens de *Le sénéchal dorme* est calculé depuis le sens de *dormir* (une propriété, $\lambda x.\text{dormir}(x)$) appliqué au sens de *le sénéchal* (un individu $\iota x.\text{sénéchal}(x)$), produisant $\text{dormir}(\iota x.\text{sénéchal}(x))$.

La **sémantique des rôles thématiques** (Fillmore 1968, cas grammar) décrit les relations entre le verbe et ses arguments : l'Agent (qui fait l'action), le Patient (qui subit l'action), le Thème (ce qui est déplacé), l'Instrument, la Localisation, etc. *Jean a cassé la vitre avec une pierre* : Jean = Agent, vitre = Patient, pierre = Instrument.

### Connexion au TALN

L'**étiquetage des rôles sémantiques** (Semantic Role Labeling, SRL) assigne les rôles thématiques aux arguments d'un prédicat. La ressource **FrameNet** (Fillmore et al. 2003) organise les prédicats en *frames* (cadres sémantiques) et leurs *frame elements* (rôles). Le verbe *donner* active le cadre GIVING avec les éléments Donor, Recipient, Theme.

La **représentation abstraite du sens** (Abstract Meaning Representation, AMR, Banarescu et al. 2013) encode le sens d'une phrase comme un graphe orienté étiqueté, indépendant de la syntaxe de surface. Elle représente la structure prédicats-arguments, la coréférence, et certains phénomènes sémantiques complexes.

**Algorithme classique :**

L'**algorithme de Lesk** (1986) résout les ambiguïtés lexicales (WSD) en mesurant le recoupement entre la définition du mot dans le dictionnaire et les définitions des mots du contexte. Si *banque* dans *déposer à la banque* a le sens "institution financière", la définition de ce sens contiendra les mots *argent*, *dépôt*, *compte* qui apparaissent probablement dans le contexte. L'algorithme de Yarowsky (1995) améliore Lesk par un apprentissage semi-supervisé (bootstrapping) qui exploite les contraintes *"one sense per discourse"* et *"one sense per collocation"*.

**Exemple linguistique :** La **présupposition** est un contenu sémantique qui reste vrai même sous négation. *Jean a arrêté de fumer* présuppose que Jean fumait — et cette présupposition est maintenue par *Jean n'a pas arrêté de fumer*. Les présuppositions posent un défi aux systèmes d'inférence automatique : ils doivent distinguer ce qui est affirmé (l'assertion principale), ce qui est impliqué par les mots (la présupposition), et ce qui est inféré du contexte (l'implicature).

---

## Couche 6 — Pragmatique : usage en contexte et actes de langage

### Ce que la linguistique étudie

La pragmatique étudie le sens tel qu'il se construit dans l'usage — ce que le locuteur veut dire, pas seulement ce que les mots signifient. La distinction entre **sens littéral** et **sens pragmatique** est au cœur du domaine.

La théorie des **actes de langage** d'Austin et Searle distingue trois dimensions de tout énoncé : l'acte **locutoire** (dire quelque chose avec un sens), l'acte **illocutoire** (faire quelque chose en disant — promettre, affirmer, ordonner), et l'acte **perlocutoire** (produire un effet sur l'interlocuteur — convaincre, intimider). *Il fait froid ici* peut être un constat (illocution assertive) ou une demande implicite de fermer la fenêtre (illocution directive indirecte) selon le contexte.

La taxonomie des actes illocutoires de Searle (1969) comprend cinq types : **assertifs** (affirmer que quelque chose est le cas), **directifs** (chercher à faire faire quelque chose à l'interlocuteur), **commissifs** (s'engager à faire quelque chose), **expressifs** (exprimer un état psychologique), **déclaratifs** (changer un état de fait par le seul fait de le dire — *Je vous déclare mari et femme*).

Les **maximes conversationnelles** de Grice (1975) régissent la coopération dans le discours. La maxime de quantité (*sois aussi informatif que nécessaire, mais pas plus*), de qualité (*ne dis pas ce que tu crois faux*), de relation (*sois pertinent*), et de manière (*sois clair, bref, ordonné*). La violation délibérée d'une maxime produit une **implicature** : si quelqu'un dit *"Il est ponctuel"* d'un collègue dont on lui a demandé l'évaluation, l'absence d'information positive implique une évaluation négative globale.

### Connexion au TALN

La **reconnaissance des actes de dialogue** (Dialogue Act Recognition) classe chaque énoncé dans un système de dialogues en un type d'acte (question, assertion, confirmation, back-channel, etc.). Le schéma DAMSL (Discourse Annotation and Markup System for Linguistic Data, Core et Allen 1997) définit une taxonomie de 42 types d'actes.

La **détection d'implicatures** — inférer ce qu'un locuteur veut dire au-delà de ce qu'il dit — est l'une des tâches les plus difficiles du TALN. Les LLM modernes réussissent relativement bien sur les implicatures conversationnelles simples, mais échouent sur les ironie et l'humour qui nécessitent une modélisation précise des croyances et des intentions du locuteur.

**Exemple linguistique :** La **politesse** (Brown et Levinson 1987) est un phénomène pragmatique qui modifie la forme des actes de langage pour gérer les menaces à la *face* (image sociale) de l'interlocuteur. Une demande directe (*Ferme la fenêtre*) est un acte menaçant pour la face de l'interlocuteur ; une demande indirecte (*Tu n'aurais pas froid ?*) ou en forme de question (*Est-ce que tu peux fermer la fenêtre ?*) atténue cette menace. En TALN, la détection du niveau de politesse et du rapport social entre les locuteurs est utile pour les systèmes de dialogue et l'analyse des discours institutionnels.

---

## Couche 7 — Analyse du discours : cohérence, coréférence, structure

### Ce que la linguistique étudie

L'**analyse du discours** étudie les unités linguistiques qui dépassent la phrase — paragraphes, textes, conversations. Elle s'intéresse à la **cohérence** (un texte forme un tout sensé), à la **cohésion** (les mécanismes formels qui relient les phrases), et à la **structure rhétorique** (l'organisation des arguments et des descriptions).

La **coréférence** est le phénomène par lequel plusieurs expressions dans un texte référent au même individu ou objet : *Jean est arrivé. Il était fatigué. Le voyageur s'assit.* Les trois expressions (*Jean*, *il*, *le voyageur*) coréfèrent. La résolution de coréférence est le mécanisme cognitif et computationnel qui lie ces expressions.

La **Rhetorical Structure Theory** (RST, Mann et Thompson 1988) décrit la structure d'un texte comme un arbre de relations rhétoriques entre segments. Les relations RST incluent l'**élaboration** (un segment développe un autre), la **contraste**, la **condition**, la **cause**, la **concession**, etc. Chaque relation distingue un **noyau** (le segment principal) et un ou plusieurs **satellites** (les segments dépendants).

```
RST exemple :
  [Le roi ordonna que les sénéchaux rendent compte]  ← NOYAU
        |
  [ÉLABORATION]
        |
  [annuellement devant le parlement]  ← satellite
```

### Connexion au TALN

La **résolution de coréférence** est une tâche centrale du NLP documentaire. Les systèmes classiques (Lee et al. 2011, Stanford CoreNLP) utilisent des règles de saillance : un pronom coréfère préférentiellement avec l'antécédent le plus récent et le plus saillant syntaxiquement (sujet > objet > oblique). SpanBERT (Joshi et al. 2020) encode des spans de texte et prédit la coréférence par un scoring de paires.

La **segmentation thématique** (TextTiling, Hearst 1997) identifie les frontières entre thèmes dans un texte en mesurant la cohérence lexicale des fenêtres glissantes : une chute de la similarité cosinus entre deux fenêtres adjacentes indique une transition thématique.

**Algorithme classique :**

Le modèle de **grille d'entités** (Barzilay et Lapata 2008) représente la cohérence d'un texte par une matrice dont les lignes sont les phrases et les colonnes sont les entités mentionnées. Chaque cellule indique le rôle syntaxique de l'entité dans la phrase (sujet, objet, autre, absent). Un texte cohérent montre des transitions régulières entre les rôles — sujet dans une phrase, sujet ou objet dans la suivante. Ce modèle sert à l'évaluation de la cohérence dans les systèmes de génération de texte.

**Exemple linguistique :** Les **marqueurs de discours** (*donc*, *cependant*, *en effet*, *or*, *pourtant*, *ainsi*) sont des indices (au sens peircéen) de relations rhétoriques : *cependant* signale une concession, *donc* une conclusion, *en effet* une justification. En TALN, l'identification des marqueurs de discours et de la relation qu'ils dénotent est une composante de l'analyse RST automatique.

---

## Couche 8 — Stylistique : registre, style, attribution

### Ce que la linguistique étudie

La **stylistique** étudie les choix linguistiques qui caractérisent un texte au-delà de son contenu référentiel : le registre (formel / informel), le style (littéraire / administratif / journalistique), et les propriétés idiosyncrasiques qui permettent d'identifier un auteur.

Le **registre** est l'ensemble des choix lexicaux et syntaxiques adaptés à une situation communicative. Un même contenu peut être exprimé dans un registre soutenu (*Il m'a remis le document*), courant (*Il m'a donné le document*), familier (*Il m'a filé le document*). La variation de registre n'est pas seulement lexicale : elle affecte la syntaxe (inversion du sujet dans les registres formels), la morphologie (subjonctif dans le registre soutenu), et la prosodie.

Douglas Biber (1988) a identifié empiriquement, sur de grands corpus, les dimensions de variation stylistique des textes anglais. Sa méthode statistique (analyse factorielle sur des centaines de traits linguistiques) révèle que la variation stylistique n'est pas unidimensionnelle : les textes varient simultanément sur plusieurs axes — implication interactive vs production informationnelle, dépendance contextuelle vs élaboration explicite, etc.

La **stylométrie** quantifie les propriétés stylistiques pour des applications d'attribution d'auteur. L'hypothèse fondamentale est que chaque auteur a une empreinte stylistique — un profil de fréquences de mots fonctionnels, de longueur de phrases, de richesse lexicale — suffisamment stable et idiosyncrasique pour l'identifier.

### Connexion au TALN

Les métriques de **lisibilité** modélisent la difficulté de lecture d'un texte. La formule de Flesch-Kincaid (Kincaid et al. 1975) utilise la longueur moyenne des mots et des phrases :

$$\text{Flesch} = 206.835 - 1.015 \frac{N_{\text{mots}}}{N_{\text{phrases}}} - 84.6 \frac{N_{\text{syllabes}}}{N_{\text{mots}}}$$

L'**attribution d'auteur** (authorship attribution) est l'une des applications les plus emblématiques de la stylométrie. La mesure Delta de Burrows (2002) compare les profils de fréquences des mots les plus fréquents entre un texte de requête et un ensemble de textes de référence. Elle a été utilisée pour attribuer (ou réfuter l'attribution de) des textes anonymes du XVIIe siècle, des manuscrits médiévaux, et même pour identifier des auteurs de publications anonymes contemporaines.

$$\Delta(T, A) = \frac{1}{n} \sum_{i=1}^{n} \frac{|z_i(T) - z_i(A)|}{\sigma_i}$$

où $z_i(T)$ est le score z de la fréquence du mot $i$ dans le texte $T$ (normalisée par la moyenne et l'écart-type de l'ensemble des textes), $z_i(A)$ est le score z correspondant dans le texte auteur $A$.

**Exemple linguistique :** La **variation diaphasique** — variation selon la situation de communication — se manifeste concrètement dans les corpus de manuscrits médiévaux. Les chartes royales utilisent des formules figées (*"Sachent tous, présents et à venir"*), une syntaxe latine héritée (*"Item que..."*), et un vocabulaire juridique spécialisé. Les chroniques narratives de la même époque emploient des temps verbaux différents (passé simple narratif, présent historique), une syntaxe plus souple, et un lexique plus varié. La stylométrie peut détecter ces différences sans supervision — les documents se regroupent par genre dans l'espace des fréquences de mots fonctionnels.

---

## Couche 9 — Rhétorique : figures, argumentation, persuasion

### Ce que la linguistique étudie

La **rhétorique** est l'art de bien parler pour persuader — discipline fondée par Aristote, théorisée par Cicéron et Quintilien, et réinterprétée par les linguistes modernes comme étude des stratégies discursives de persuasion et d'expression.

La rhétorique classique distingue cinq parties du discours (*inventio*, *dispositio*, *elocutio*, *memoria*, *actio*) et trois modes de preuve : le **logos** (argument rationnel), l'**éthos** (crédibilité de l'orateur), et le **pathos** (appel aux émotions de l'auditoire). Cette trichotomie est opérationnelle en TALN : détecter le type d'appel rhétorique d'un texte (faits / autorité / émotions) est une tâche d'argumentation mining.

Les **figures de style** sont des déviations par rapport au langage ordinaire qui produisent des effets expressifs. On distingue les figures de **substitution** (métaphore, métonymie, synecdoque), de **répétition** (anaphore, épiphore, chiasme), d'**omission** (ellipse, asyndète), et d'**addition** (pléonasme, gradation, hyperbole).

La **métonymie** substitue un terme par un autre entretenant une relation de contiguïté avec lui : *"Lire Proust"* pour *"lire l'œuvre de Proust"*, *"La Maison Blanche a déclaré"* pour *"Le gouvernement américain a déclaré"*. La métonymie est fondamentalement différente de la métaphore (substitution par ressemblance) mais les deux sont souvent confondues dans les systèmes de détection automatique.

### Connexion au TALN

L'**argumentation mining** (Stab et Gurevych 2017) vise à extraire automatiquement la structure argumentative d'un texte : identifier les revendications (*claims*), les preuves (*evidence*), et les relations entre elles (support, réfutation). Cette tâche s'appuie sur la RST (couche 7) et sur la détection des marqueurs discursifs (couche 7) mais requiert aussi une compréhension du contenu sémantique (couche 5) pour distinguer une assertion factuelle d'une évaluation normative.

La **détection de métaphores** (Shutova et al. 2016) identifie les usages métaphoriques des mots — *"cette théorie s'effondre sous les critiques"* (*s'effondre* employé métaphoriquement d'une théorie abstraite). Les systèmes actuels utilisent des ressources ontologiques (WordNet, FrameNet) pour détecter les incompatibilités sémantiques entre un mot et son contexte.

L'**analyse des sentiments** (Pang et Lee 2008) est une forme de rhétorique computationnelle simplifiée : elle détecte la valence émotionnelle d'un texte (positif, négatif, neutre) et éventuellement des émotions plus fines (joie, colère, tristesse, dégoût, surprise, peur). Les approches modernes entraînent des classifieurs sur des corpus annotés — critiques de films, commentaires de produits, messages de réseaux sociaux.

**Exemple linguistique :** Le **chiasme** est une figure rhétorique d'inversion symétrique : *"Il faut manger pour vivre, non vivre pour manger"* (Molière). Sa structure est ABBA : les éléments du premier membre sont repris en ordre inverse dans le second. En TALN, la détection automatique de chiasmes requiert de l'identification des structures syntaxiques (couche 4), de la sémantique lexicale pour vérifier les relations thématiques (couche 5), et une analyse de la structure du discours (couche 7) pour détecter les paires en parallèle.

---

## Les interconnexions entre couches

La présentation en pile, utile pour l'analyse, dissimule les interactions réelles entre les niveaux. Voici les couplages les plus importants :

**Phonologie ↔ Syntaxe :** la prosodie signale les frontières syntaxiques. La phrase *"Les poules du couvent couvent"* est ambiguë à l'écrit (le second *couvent* est-il un nom ou un verbe ?) mais non à l'oral, où l'accent tombe différemment selon l'analyse syntaxique. Les systèmes de parsing peuvent exploiter les informations prosodiques pour lever les ambiguïtés syntaxiques.

**Morphologie ↔ Sémantique :** la structure morphologique d'un mot contraint son sens possible. *Dé-faire* signifie l'inverse de *faire* parce que le préfixe *dé-* a ce sens. Un système de sémantique compositionnelle doit intégrer la morphologie pour calculer le sens des mots dérivés.

**Syntaxe ↔ Pragmatique :** l'ordre des mots en français, malgré sa relative rigidité, encode des informations pragmatiques. La topicalisation (*Ce livre, je l'ai lu*) place le topique en tête de phrase. La focalisation (*C'est Jean qui a signé, pas Paul*) met en relief l'élément focalisé. Ces constructions sont syntaxiques en surface mais pragmatiques en fonction.

**Sémantique ↔ Pragmatique :** la frontière entre ce que dit une phrase (sens sémantique) et ce qu'elle implique (sens pragmatique) est souvent floue et toujours contextuelle. *"Il y a du sel sur la table"* est sémantiquement une assertion sur la localisation du sel, mais pragmatiquement souvent une demande de le passer. Les systèmes de NLU (Natural Language Understanding) doivent gérer cette interface constamment.

**Coréférence ↔ Tous les niveaux :** la résolution de coréférence requiert des informations morphologiques (genre et nombre des pronoms), syntaxiques (saillance des positions sujet/objet), sémantiques (compatibilité des rôles thématiques), pragmatiques (centre d'attention du discours, accessibilité), et discursives (distance dans le texte). C'est la tâche qui illustre le mieux l'inséparabilité des couches.

---

## Les grands modèles de langage et la pile

Les modèles Transformer comme BERT ou GPT traitent le texte comme une séquence de tokens et apprennent des représentations qui intègrent, de façon non explicite, des informations de toutes les couches de la pile.

Des analyses probing (Tenney et al. 2019) montrent que les couches inférieures d'un Transformer pré-entraîné encodent préférentiellement des informations phonologiques et morphologiques ; les couches médianes encodent des informations syntaxiques ; les couches supérieures encodent des informations sémantiques et pragmatiques. Cette correspondance approximative entre profondeur de la couche et niveau de la pile n'est pas programmée — elle émerge du pré-entraînement sur de grands corpus.

Ce résultat est à la fois fascinant et troublant. Il suggère que les LLM reproduisent, sans supervision explicite, l'organisation hiérarchique que les linguistes ont mis deux siècles à formaliser. Mais il ne signifie pas que les LLM comprennent le langage au sens où les linguistes entendent ce terme : la représentation d'une information n'implique pas la compréhension du système qui la produit.

---

## Bibliographie de référence

### Panorama historique

Saussure, F. de (1916). *Cours de linguistique générale*. Payot. — Édition critique Engler (1967).

Chomsky, N. (1957). *Syntactic Structures*. Mouton. — Et (1965). *Aspects of the Theory of Syntax*. MIT Press.

Jakobson, R. (1963). *Essais de linguistique générale*. Minuit. — Traduction française des travaux fondateurs du Cercle de Prague.

Peirce, C. S. (1931–1958). *Collected Papers*. Harvard University Press. 8 volumes.

Morris, C. (1938). *Foundations of the Theory of Signs*. University of Chicago Press.

### Phonologie et morphologie

Koskenniemi, K. (1983). **Two-Level Morphology**. Thèse, Université d'Helsinki.

Sennrich, R., Haddow, B., & Birch, A. (2016). **Neural Machine Translation of Rare Words with Subword Units**. *ACL 2016*. [arXiv:1508.07909](https://arxiv.org/abs/1508.07909)

Porter, M. F. (1980). **An algorithm for suffix stripping**. *Program*, 14(3).

Kaplan, R., & Kay, M. (1994). **Regular models of phonological rule systems**. *Computational Linguistics*, 20(3).

### Lexicologie et sémantique lexicale

Sparck Jones, K. (1972). **A statistical interpretation of term specificity and its application in retrieval**. *Journal of Documentation*, 28(1).

Church, K. W., & Hanks, P. (1990). **Word association norms, mutual information, and lexicography**. *Computational Linguistics*, 16(1).

Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). **Efficient Estimation of Word Representations in Vector Space**. [arXiv:1301.3781](https://arxiv.org/abs/1301.3781)

Miller, G. A. (1995). **WordNet: a lexical database for English**. *Communications of the ACM*, 38(11).

Lakoff, G., & Johnson, M. (1980). *Metaphors We Live By*. University of Chicago Press.

### Syntaxe

Cocke, J. (1969). *Programming Languages and Their Compilers: Preliminary Notes*. NYU.

Nivre, J. (2003). **An Efficient Algorithm for Projective Dependency Parsing**. *IWPT 2003*.

McDonald, R., Pereira, F., Ribarov, K., & Hajič, J. (2005). **Non-Projective Dependency Parsing using Spanning Tree Algorithms**. *HLT/EMNLP 2005*.

Dozat, T., & Manning, C. D. (2017). **Deep Biaffine Attention for Neural Dependency Parsing**. *ICLR 2017*. [arXiv:1611.01734](https://arxiv.org/abs/1611.01734)

Nivre, J., et al. (2020). **Universal Dependencies v2**. *LREC 2020*.

### Sémantique

Frege, G. (1892). **Über Sinn und Bedeutung**. *Zeitschrift für Philosophie und philosophische Kritik*, 100.

Montague, R. (1970). **Universal Grammar**. *Theoria*, 36(3).

Fillmore, C. J. (2003). **FrameNet as a 'Net'**. *LREC 2004* (avec Baker et Sato).

Lesk, M. (1986). **Automatic sense disambiguation using machine readable dictionaries**. *SIGDOC 1986*.

Yarowsky, D. (1995). **Unsupervised word sense disambiguation rivaling supervised methods**. *ACL 1995*.

Banarescu, L., et al. (2013). **Abstract Meaning Representation for Sembanking**. *LAW VII 2013*.

### Pragmatique

Austin, J. L. (1962). *How to Do Things with Words*. Oxford University Press.

Searle, J. R. (1969). *Speech Acts*. Cambridge University Press.

Grice, H. P. (1975). **Logic and Conversation**. In *Syntax and Semantics, vol. 3*.

Brown, P., & Levinson, S. C. (1987). *Politeness: Some Universals in Language Usage*. Cambridge University Press.

Core, M., & Allen, J. (1997). **Coding Dialogs with the DAMSL Annotation Scheme**. *AAAI 1997*.

### Analyse du discours

Mann, W. C., & Thompson, S. A. (1988). **Rhetorical structure theory: Toward a functional theory of text organization**. *Text*, 8(3).

Hearst, M. A. (1997). **TextTiling: Segmenting Text into Multi-Paragraph Subtopic Passages**. *Computational Linguistics*, 23(1).

Barzilay, R., & Lapata, M. (2008). **Modeling Local Coherence: An Entity-Based Approach**. *Computational Linguistics*, 34(1).

Joshi, M., et al. (2020). **SpanBERT: Improving Pre-training by Representing and Predicting Spans**. *TACL*. [arXiv:1907.10529](https://arxiv.org/abs/1907.10529)

### Stylistique et rhétorique computationnelle

Burrows, J. (2002). **'Delta': A Measure of Stylistic Difference and a Guide to Likely Authorship in Uncertain Cases**. *Literary and Linguistic Computing*, 17(3).

Biber, D. (1988). *Variation across Speech and Writing*. Cambridge University Press.

Kincaid, J. P., et al. (1975). **Derivation of New Readability Formulas for Navy Enlisted Personnel**. Research Branch Report 8-75.

Stab, C., & Gurevych, I. (2017). **Parsing Argumentation Structures in Persuasive Essays**. *Computational Linguistics*, 43(3).

Shutova, E., Kiela, D., & Maillard, J. (2016). **Black Holes and White Rabbits: Metaphor Identification with Visual Features**. *NAACL 2016*.

Pang, B., & Lee, L. (2008). **Opinion Mining and Sentiment Analysis**. *Foundations and Trends in Information Retrieval*, 2(1–2).

### Probing et interprétabilité des LLM

Tenney, I., Das, D., & Pavlick, E. (2019). **BERT Rediscovers the Classical NLP Pipeline**. *ACL 2019*. [arXiv:1905.05950](https://arxiv.org/abs/1905.05950)

---

*Document de référence rédigé pour le Master Data/IA · Module NLP · MD5 Volet 2 · 2026. Ce document est indépendant du syllabus et peut être consulté à n'importe quel moment du module. Il sert de guide de lecture pour aller plus loin sur n'importe quelle couche de la pile NLP.*
