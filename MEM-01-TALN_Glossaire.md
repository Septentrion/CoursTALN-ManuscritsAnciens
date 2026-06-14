# Glossaire du module NLP — Traitement automatique des langues

**Module NLP · Master Data/IA · MD5 Volet 2 · 2026**

Ce glossaire rassemble les 170 notions importantes introduites dans les Chapitres 1 à 11 et dans le document de référence sur la pile NLP. Pour chaque terme, la définition est suivie d'une référence au(x) chapitre(s) où il est traité en détail.

---

## A

**Acte de langage** — Notion introduite par Austin (1962) et Searle (1969) selon laquelle dire, c'est faire : une promesse, une question, un ordre ne se contentent pas de décrire le monde, ils accomplissent des actes sociaux. On distingue l'acte locutoire (dire quelque chose), l'acte illocutoire (faire quelque chose en disant — promettre, asserter, ordonner), et l'acte perlocutoire (produire un effet sur l'interlocuteur). *→ Pile NLP, §6 ; Chapitre 5*

**Active learning (apprentissage actif)** — Stratégie d'entraînement dans laquelle le modèle sélectionne activement les exemples les plus informatifs à annoter, plutôt de laisser l'annotateur choisir aléatoirement. Particulièrement utile quand l'annotation est coûteuse, comme pour les corpus de manuscrits médiévaux. *→ Chapitre 9, §5.3*

**Adaptateur LoRA** — Module entraînable de faible rang inséré dans les couches de projection d'un modèle pré-entraîné pour le fine-tuning à moindre coût mémoire. Chaque matrice de poids $\mathbf{W}$ est augmentée par $\Delta\mathbf{W} = \mathbf{B}\mathbf{A}$ avec $\mathbf{A} \in \mathbb{R}^{r \times d}$ et $\mathbf{B} \in \mathbb{R}^{d \times r}$, $r \ll d$. Seuls les paramètres des adaptateurs sont mis à jour ; les poids originaux restent gelés. *→ Chapitre 3, §3*

**Agrégation par page** — Opération qui regroupe les lignes individuelles d'un manuscrit en unités documentaires (pages, actes) avant d'appliquer la modélisation thématique. Nécessaire parce que BERTopic et LDA requièrent des documents d'au moins quelques dizaines de tokens pour estimer des distributions de topics stables. *→ Chapitre 8, §1*

**Alignement subword** — Procédure qui associe les étiquettes NER ou POS (définies au niveau du mot) aux sous-mots produits par la tokenisation BPE ou WordPiece. La stratégie *first-token* attribue l'étiquette du mot au premier sous-mot et ignore les suivants (marqués −100) ; la stratégie *majority* vote entre tous les sous-mots. *→ Chapitre 5, §4.1 ; Chapitre 6, §2.2 ; Chapitre 10, §9*

**Alternance u/v** — Phénomène graphique du moyen français où les lettres *u* et *v* représentent indifféremment le son [u] ou [v] selon leur position dans le mot. En position initiale devant voyelle, *u* représente [v] (*uoir → voir*) ; en position médiale après voyelle, *u* représente aussi [v] (*auoir → avoir*). Cette alternance est résolue par des règles positionnelles déterministes dans le pipeline de normalisation. *→ Chapitre 4, §1.4*

**AMR (Abstract Meaning Representation)** — Formalisme de représentation du sens d'une phrase sous forme de graphe orienté étiqueté, indépendant de la syntaxe de surface. Chaque nœud représente un concept (prédicat ou entité), chaque arc représente une relation sémantique ou un rôle thématique. Introduit par Banarescu et al. (2013), il vise à capturer la structure prédicats-arguments et la coréférence. *→ Pile NLP, §5*

**Analyse du discours** — Couche de la pile NLP qui étudie les unités textuelles qui dépassent la phrase : la cohérence (comment le texte forme un tout cohérent), la cohésion (les mécanismes formels — pronoms, connecteurs — qui relient les phrases), et la structure rhétorique (organisation des arguments). *→ Pile NLP, §7*

**Annotation BIO/BIOES** — Schémas d'étiquetage des séquences pour la NER. BIO utilise trois préfixes : B (début d'entité), I (intérieur d'entité), O (hors entité). BIOES ajoute E (fin d'entité) et S (entité d'un seul token), améliorant la précision du décodage pour les entités mono-token. *→ Chapitre 5, §3.1*

**Antonymie** — Relation lexicale entre deux mots de sens opposés. L'antonymie gradable (*chaud/froid*) admet des degrés intermédiaires, l'antonymie complémentaire (*vivant/mort*) n'en admet pas. En TALN, l'antonymie est encodée dans WordNet et exploitée par les systèmes d'inférence lexicale et d'analyse des sentiments. *→ Pile NLP, §3*

**Arbitraire du signe** — Principe saussurien selon lequel le lien entre le signifiant (image acoustique ou graphique) et le signifié (concept) est arbitraire : il n'existe aucune ressemblance naturelle entre le son [∫a] et l'animal *chat*. L'arbitraire explique la variation interlinguistique et est une des deux propriétés fondamentales du signe linguistique avec la linéarité. *→ Pile NLP, §Panorama*

**Arbre de dépendance** — Représentation syntaxique où chaque token est relié à son *tête* (gouverneur) par une relation typée (sujet, objet, modificateur, etc.). L'arbre est enraciné au verbe principal. Universal Dependencies standardise ces relations sur plus de 70 langues, permettant la comparaison interlinguistique et l'entraînement de parsers multilingues. *→ Pile NLP, §4 ; Chapitre 5*

**Attention (mécanisme d')** — Opération centrale du Transformer qui calcule, pour chaque position de la séquence, une somme pondérée de toutes les autres positions. Le poids attribué à chaque position est calculé par un produit scalaire entre la requête (*query*) de la position courante et les clés (*keys*) de toutes les positions, normalisé par une softmax. Ce mécanisme permet de capturer des dépendances à longue distance sans récurrence. *→ Chapitre 1, §3*

**Attribution d'auteur (stylométrie)** — Application de méthodes quantitatives pour identifier l'auteur probable d'un texte anonyme à partir de son profil stylistique. La méthode Delta de Burrows (2002) compare les profils de fréquences des mots les plus fréquents, normalisés par des scores z. *→ Pile NLP, §8*

**AWQ (Activation-aware Weight Quantization)** — Méthode de quantisation 4 bits qui identifie les poids les plus importants pour la précision (ceux activés par de grandes valeurs d'activation) et les préserve de la quantisation agressive, ne quantisant agressivement que les poids peu importants. Produit une meilleure précision que la quantisation uniforme à même taux de compression. *→ Chapitre 9, §1.3*

## B

**Baseline** — Configuration de référence minimale contre laquelle on mesure les améliorations apportées par un système plus sophistiqué. Dans ce cours, les baselines successives sont : le texte HTR brut (CER initial), la normalisation par règles (après Chapitre 4), et la prédiction par gazetier (pour la NER). *→ Chapitres 4, 5, 6*

**Batching dynamique** — Stratégie de traitement par lots dans laquelle le serveur déclenche l'inférence dès qu'un timeout est atteint ou que la taille maximale du lot est atteinte — selon le premier événement. Permet d'équilibrer la latence (on n'attend pas que le lot soit plein) et le débit (on n'envoie pas chaque requête individuellement). *→ Chapitre 9, §3.3*

**BERT** — Modèle de représentation contextuelle du langage basé sur l'architecture Transformer encoder, pré-entraîné par Devlin et al. (2018) sur deux tâches : la prédiction de mots masqués (MLM, Masked Language Modeling) et la prédiction de la phrase suivante (NSP). BERT est bidirectionnel : il considère le contexte à gauche et à droite simultanément, contrairement aux LM auto-régressifs. *→ Chapitre 1, §8*

**BIO (schéma)** — Schéma d'étiquetage des entités nommées utilisant trois étiquettes par type : B-TYPE (début), I-TYPE (intérieur), O (hors entité). Pour cinq types, BIO produit onze étiquettes. Sa limite principale est de ne pas distinguer les entités mono-token des débuts d'entités multi-tokens, problème résolu par le schéma BIOES. *→ Chapitre 5, §3.1*

**BIOES (schéma)** — Extension du schéma BIO qui ajoute E-TYPE (fin d'entité) et S-TYPE (entité d'un seul token). Pour cinq types, BIOES produit vingt et une étiquettes. Les études empiriques montrent un gain de F1 de 0.5 à 2 points par rapport à BIO, particulièrement sur les entités mono-token fréquentes dans les corpus médiévaux (titres, dates). *→ Chapitre 5, §3.1*

**BLEU** — Métrique d'évaluation automatique de la traduction automatique qui mesure le chevauchement de n-grammes entre la traduction produite et une référence humaine. BLEU-4 combine les précisions de n-grammes de 1 à 4 et applique une pénalité de brièveté. Il est très utilisé mais imparfait : deux traductions correctes de sens différent peuvent avoir des scores très différents selon la référence choisie. *→ Chapitres 4, 11 ; Pile NLP*

**BPE (Byte Pair Encoding)** — Algorithme de tokenisation en sous-mots qui fusionne itérativement la paire de symboles la plus fréquente dans le corpus. Produit un vocabulaire de taille fixe dont les entrées les plus longues correspondent aux mots fréquents et les plus courtes aux caractères. Utilisé par GPT-2/3/4 ; les variantes WordPiece (BERT) et Unigram LM (SentencePiece) suivent un principe similaire. *→ Chapitre 1, §7 ; Pile NLP, §2*

## C

**c-TF-IDF** — Variante de TF-IDF utilisée dans BERTopic pour identifier les mots représentatifs de chaque topic. Le corpus de chaque topic est traité comme un seul document, et le score c-TF-IDF pénalise les mots communs à tous les topics. Elle est plus adaptée que TF-IDF standard pour des corpus où les topics partagent beaucoup de vocabulaire. *→ Chapitre 7, §1.4 ; Chapitre 8, §1*

**CamemBERT** — Modèle de langue BERT entraîné sur des textes français par l'équipe almanach (Martin et al. 2020). Construit sur l'architecture RoBERTa, il a été pré-entraîné sur le corpus OSCAR (données web françaises). C'est le modèle de fondation utilisé dans ce cours pour le fine-tuning NER médiéval. *→ Chapitres 5, 6*

**CER (Character Error Rate)** — Métrique évaluant la qualité d'une transcription ou d'une normalisation en calculant la distance d'édition entre la sortie du système et la référence, normalisée par la longueur de la référence. Un CER de 0 indique une transcription parfaite. Dans ce cours, l'objectif est CER < 10 % après normalisation et CER < 5 % après correction contextuelle. *→ Chapitres 2, 4*

**Chaîne de coréférence** — Ensemble de toutes les expressions dans un texte qui référent au même individu ou entité dans le monde. Dans *"Jean porta les lettres. Le sénéchal les remit au roi. Il ne dit mot."*, *Jean*, *le sénéchal* et *il* forment une chaîne de coréférence dont la mention canonique est *Jean*. *→ Chapitres 7, 8*

**chrF** — Métrique d'évaluation automatique de la traduction basée sur les n-grammes de caractères. Mieux corrélée avec les jugements humains que BLEU pour les langues à morphologie riche, elle pénalise moins les variations morphologiques proches. Le score chrF combine précision et rappel des n-grammes de caractères de longueur 1 à 6. *→ Chapitre 11, §5.2*

**Cohérence du discours** — Propriété d'un texte qui forme un tout sensé et connecté, où les phrases s'enchaînent de façon logique et où les entités sont correctement introduites et maintenues. Distincte de la cohésion (mécanismes formels de liaison), la cohérence est une propriété globale qui requiert l'activation d'une représentation mentale du monde décrit. *→ Pile NLP, §7*

**Cohen (kappa de)** — Mesure de l'accord entre deux annotateurs qui corrige l'accord observé par l'accord dû au hasard. Un kappa supérieur à 0.61 est considéré comme accord substantiel, au-dessus de 0.80 comme accord presque parfait. Pour la NER, le calcul doit tenir compte de la classe O très majoritaire qui gonfle artificiellement l'accord par chance. *→ Chapitre 5, §5 ; Chapitre 6, §5*

**Compétence vs performance** — Distinction chomskienne entre la connaissance implicite de la grammaire (compétence) et les productions linguistiques réelles d'un locuteur (performance), soumises aux contraintes de mémoire, d'attention et des erreurs de parole. En TALN, un corpus est de la performance ; les modèles de langage cherchent à approximer la compétence. *→ Pile NLP, §Panorama*

**Concept drift** — Phénomène par lequel la distribution statistique des données d'entrée d'un modèle en production s'éloigne progressivement de la distribution des données d'entraînement, dégradant les performances. En TALN médiéval, un modèle entraîné sur des chartes normandes peut subir un drift quand on l'applique à des registres paroissiaux gascons. *→ Chapitre 9, §5.2*

**Continuous batching** — Technique de serving qui permet d'insérer de nouvelles séquences dans un batch en cours d'exécution dès qu'une séquence se termine, sans attendre que tout le batch soit traité. Implémentée notamment dans vLLM via l'algorithme PagedAttention, elle maximise l'utilisation des GPU pour les modèles génératifs. *→ Chapitre 9, §3.4*

**CONVENTIONS_NLP.md** — Document de traçabilité scientifique qui enregistre toutes les décisions de normalisation orthographique prises pendant un projet de traitement de manuscrits médiévaux. Il documente les cas ambigus résolus, les règles appliquées et leurs justifications linguistiques, et sert de référence pour toute personne qui utilisera les données annotées. *→ Chapitre 4, §7*

**Coréférence** — Phénomène linguistique par lequel plusieurs expressions d'un texte référent au même individu ou entité dans le monde. La résolution de coréférence — identifier automatiquement ces chaînes — est l'une des tâches les plus difficiles du NLP car elle requiert des informations morphologiques, syntaxiques, sémantiques et pragmatiques simultanément. *→ Chapitres 7, 8*

**Corpus parallèle** — Ensemble de textes dans une langue A alignés avec leurs traductions dans une langue B, au niveau de la phrase, du paragraphe, ou du document. Les corpus parallèles sont indispensables pour l'entraînement des systèmes de traduction automatique neurale ; leur constitution pour le moyen français est un chantier encore incomplet. *→ Chapitre 11, §2*

**CRF (Conditional Random Field)** — Modèle probabiliste de séquence qui prédit une étiquette pour chaque élément d'une séquence en prenant en compte les dépendances entre étiquettes adjacentes. Introduit par Lafferty et al. (2001), il est utilisé en NER pour garantir que des séquences d'étiquettes invalides (I-PER après O) ne soient pas produites, via une matrice de transition apprise. *→ Pile NLP, §1 ; Chapitre 5, §4.2*

**CYK (algorithme de Cocke-Younger-Kasami)** — Algorithme de parsing bottom-up pour les grammaires hors-contexte probabilistes (PCFG), de complexité $O(n^3 \cdot |G|)$. Il remplit une table triangulaire (chart) en calculant les probabilités de toutes les analyses possibles de chaque sous-chaîne, permettant de trouver l'analyse la plus probable. *→ Pile NLP, §4*

## D

**Data contract** — Document structuré (JSON) qui formalise les métadonnées et les annotations associées à chaque ligne d'un corpus de manuscrits numérisés. Il inclut la transcription HTR brute, la forme normalisée, les annotations NER/POS/lemmes, les scores de confiance, et le polygone liant l'annotation à sa position physique dans le manuscrit. Le data contract évolue au fil du pipeline en versions successives. *→ Chapitres 2, 6, 8*

**Delta de Burrows** — Mesure de distance stylométrique utilisée pour l'attribution d'auteur. Elle compare les profils de fréquences des mots les plus fréquents entre un texte inconnu et des textes de référence, après normalisation par scores z. Plus le Delta est petit, plus le texte inconnu ressemble stylistiquement au texte de référence. *→ Pile NLP, §8*

**Désambiguïsation lexicale (WSD)** — Tâche qui consiste à déterminer, pour un mot polysémique dans un contexte donné, lequel de ses sens est actif. *Banque* dans *déposer à la banque* doit être désambiguïsé en "institution financière" plutôt qu'en "banc de sable" ou "rive". L'algorithme de Lesk (1986) et la méthode de Yarowsky (1995) sont les références classiques. *→ Pile NLP, §5*

**Diachronie vs synchronie** — Distinction saussurienne fondamentale : l'étude d'une langue à un moment donné (synchronie) est indépendante de l'étude de son évolution dans le temps (diachronie). La normalisation orthographique du moyen français est une opération de translation diachronique ; la NER sur le texte normalisé opère en synchronie. *→ Pile NLP, §Panorama*

**Distillation de modèles** — Technique d'entraînement qui transfère les connaissances d'un grand modèle (*teacher*) vers un petit modèle (*student*), en utilisant les distributions de probabilités souples du teacher comme signal d'entraînement supplémentaire. La perte de distillation combine une cross-entropie dure (versus les étiquettes réelles) et une divergence KL adoucie (versus le teacher). *→ Chapitre 9, §2*

**DMF (Dictionnaire du Moyen Français)** — Dictionnaire de référence scientifique du français des XIVe et XVe siècles, élaboré par l'ATILF (CNRS – Université de Lorraine). Il couvre plus de 65 000 entrées et intègre LGeRM, un moteur de lemmatisation pour les formes médiévales. Utilisé dans ce cours pour valider les normalisations et constituer le cache lexical. *→ Chapitre 4, §3*

**Docker** — Outil de containerisation qui empaquète une application avec toutes ses dépendances dans une image reproductible. En TALN de production, un Dockerfile fige les versions de toutes les bibliothèques et garantit que le pipeline produit les mêmes résultats sur n'importe quelle machine. *→ Chapitre 10, §3*

**Double quantisation** — Technique introduite dans QLoRA qui quantise les constantes de quantisation elles-mêmes (en plus des poids), économisant environ 0.37 bits par paramètre supplémentaires. Combinée à la quantisation NF4, elle réduit significativement l'empreinte mémoire des modèles volumineux sans dégradation notable des performances. *→ Chapitres 3, 9*

## E

**Encodage positionnel** — Mécanisme qui injecte dans les représentations d'entrée du Transformer une information sur la position de chaque token dans la séquence, compensant l'absence de récurrence. Les encodages sinusoïdaux classiques (Vaswani et al. 2017) sont déterministes ; RoPE (Rotary Position Embedding) et ALiBi sont des variantes récentes qui améliorent la généralisation aux séquences longues. *→ Chapitre 1, §4*

**Entité nommée** — Fragment de texte qui réfère à un objet du monde appartenant à une catégorie prédéfinie : personne (PER), lieu (LOC), organisation (ORG), date (DATE), titre (TITLE) dans ce cours. La reconnaissance d'entités nommées (NER) est la tâche d'identifier automatiquement ces fragments et leur type dans un texte. *→ Chapitres 5, 6*

**Export HuggingFace Hub** — Publication d'un modèle fine-tuné sur la plateforme Hugging Face, accompagné d'une model card et du tokeniseur. L'export garantit que d'autres chercheurs peuvent reproduire exactement le modèle et l'utiliser sans réentraînement, à condition de spécifier le split_hash des données d'entraînement. *→ Chapitre 10, §5.4*

## F

**F1 (score)** — Moyenne harmonique de la précision et du rappel : $F_1 = 2 \cdot P \cdot R / (P + R)$. En NER avec seqeval, il est calculé au niveau des spans (entités complètes), pas des tokens : une entité ne compte comme vrai positif que si elle est correctement délimitée et correctement typée. Le F1 micro pondère par la fréquence des types, le F1 macro leur donne un poids égal. *→ Chapitres 5, 6*

**FastAPI** — Framework Python pour construire des API REST performantes avec validation automatique des données via Pydantic. Dans ce cours, il expose les endpoints `/health`, `/transcribe` et `/analyze` du pipeline NER médiéval, avec documentation OpenAPI générée automatiquement. *→ Chapitres 9, 10*

**Fine-tuning** — Entraînement supplémentaire d'un modèle pré-entraîné sur une tâche ou un domaine spécifique. Dans ce cours, le fine-tuning est réalisé de façon efficace via LoRA (Chapitre 3) pour la normalisation orthographique (mT5) et la NER (CamemBERT), limitant le nombre de paramètres entraînés à moins de 1 % du total. *→ Chapitres 3, 6*

**Flesch-Kincaid (indice de lisibilité)** — Formule qui prédit la difficulté de lecture d'un texte en combinant la longueur moyenne des phrases et la longueur moyenne des mots. Un score élevé (proche de 100) indique un texte facile ; un score faible indique un texte difficile. Utilisé en TALN pour le profilage stylistique et l'adaptation du contenu à un public cible. *→ Pile NLP, §8*

**FP16 / FP32** — Formats de représentation des nombres à virgule flottante sur 16 et 32 bits respectivement. FP32 est le format standard des poids de modèles PyTorch ; FP16 réduit l'empreinte mémoire de moitié au prix d'une légère perte de précision numérique, généralement sans impact sur les performances du modèle. *→ Chapitre 9, §1*

**FrameNet** — Base lexicale hiérarchique fondée sur la théorie des cadres sémantiques (Fillmore 1976). Elle organise les prédicats en *frames* (cadres évoquant une situation type) et leurs *frame elements* (les participants et les circonstances). Le frame GIVING comporte les éléments Donor, Recipient, Theme, etc. Utilisée pour l'étiquetage des rôles sémantiques. *→ Pile NLP, §5*

**FST (Transducteur à états finis)** — Automate qui transforme une séquence d'entrée en une séquence de sortie. En phonologie computationnelle, un FST encode les règles de réécriture phonologique et peut convertir des graphèmes en phonèmes (G2P). Kaplan et Kay (1994) ont montré que toute grammaire phonologique régulière peut être compilée en un FST unique. *→ Pile NLP, §1*

## G

**G2P (Graphème vers Phonème)** — Tâche de conversion d'une séquence de lettres en une séquence de phonèmes, indispensable pour la synthèse vocale et la reconnaissance de la parole. En français, les règles G2P sont relativement régulières mais comportent de nombreuses exceptions (*femme* se prononce [fam], pas [fɛm]). *→ Pile NLP, §1*

**Gazetier** — Liste de termes appartenant à une ou plusieurs catégories d'entités nommées. Un gazetier médiéval peut contenir les prénoms, les noms de lieux normands, les titres de noblesse. En NER, l'annotation par gazetier sert de *weak supervision* (supervision faible) pour initialiser l'entraînement sans annotation manuelle complète. *→ Chapitres 6, 8*

**GPTQ** — Algorithme de quantisation post-entraînement (Frantar et al. 2022) qui quantise les poids d'un réseau couche par couche en minimisant l'erreur de reconstruction sur un ensemble de calibration. GPTQ produit des modèles 4 bits avec une perte de qualité souvent inférieure à 5 % par rapport au modèle FP16, et est adapté aux grands modèles génératifs. *→ Chapitre 9, §1.3*

**Grammaire des constructions** — Théorie linguistique (Goldberg 1995) selon laquelle les unités de base de la grammaire ne sont pas des règles abstraites mais des *constructions* — des paires forme-sens de toute taille, du morphème à la phrase entière. La construction ditransitive [Sujet V Objet1 Objet2] encode le sens de "X cause que Y reçoive Z" indépendamment du verbe spécifique. *→ Pile NLP, §Panorama*

**Grammaire générative** — Programme de recherche inauguré par Chomsky (1957) qui cherche à expliciter la connaissance implicite que les locuteurs ont de leur langue sous forme de règles formelles récursives capables de générer toutes et seulement les phrases grammaticales d'une langue. La hiérarchie de Chomsky classifie les grammaires par leur puissance expressive. *→ Pile NLP, §Panorama*

**Graphe de connaissances** — Structure de données qui représente des entités (nœuds) et les relations entre elles (arêtes typées). Dans ce cours, le graphe médiéval contient des nœuds de types PER, LOC, DATE, ORG, TITLE reliés par des relations comme *porte_titre*, *réside_à*, *agit_lors_de*. Chaque nœud conserve son *polygon_ref* pour la traçabilité vers le document source. *→ Chapitres 7, 8*

## H

**Hallucination** — Phénomène par lequel un modèle de langage génère des informations factuellement incorrectes mais linguistiquement plausibles, sans signal d'incertitude. En traduction automatique historique, un LLM peut inventer une date ou un nom de lieu absent du texte source en "complétant" avec ses connaissances encyclopédiques. *→ Chapitre 11, §6*

**HDBSCAN** — Algorithme de clustering hiérarchique basé sur la densité (*Hierarchical Density-Based Spatial Clustering of Applications with Noise*). Contrairement à K-Means, HDBSCAN ne requiert pas de spécifier le nombre de clusters et identifie les points ne correspondant à aucun cluster (label −1, bruit). Utilisé dans BERTopic après réduction dimensionnelle UMAP. *→ Chapitres 7, 8*

**Hiérarchie de Chomsky** — Classification des grammaires formelles en quatre niveaux (types 0 à 3) par leur puissance expressive : les grammaires régulières (type 3, automates finis) sont les moins puissantes, les grammaires non restreintes (type 0, machines de Turing) les plus puissantes. Les langues naturelles requièrent au moins des grammaires hors-contexte (type 2) pour les structures syntagmatiques de base. *→ Pile NLP, §Panorama*

**HMM (modèle de Markov caché)** — Modèle probabiliste de séquence dans lequel une séquence d'états cachés (non observables) génère une séquence d'observations. L'algorithme de Viterbi trouve la séquence d'états la plus probable ; l'algorithme de Baum-Welch entraîne le modèle. Longtemps standard en reconnaissance vocale et en POS-tagging, les HMM ont été supplantés par les CRF puis par les réseaux profonds. *→ Pile NLP, §1*

**HTR (Handwritten Text Recognition)** — Sous-domaine de la reconnaissance optique de caractères spécialisé dans les manuscrits et l'écriture cursive. Contrairement à l'OCR sur textes imprimés, le HTR doit gérer la variabilité de l'écriture manuscrite, l'encre vieillie, et les ligatures. Le data contract HTR du Volet 1 fournit le point de départ du pipeline NLP de ce cours. *→ Chapitre 2*

**Hyponymie / hyperonymie** — Relation hiérarchique entre deux termes : *pinson* est un hyponyme de *oiseau* (plus spécifique), *oiseau* est un hyperonyme de *pinson* (plus général). Dans WordNet, cette relation forme une hiérarchie de synsets. En TALN, elle est exploitée pour l'expansion de requêtes et l'inférence textuelle. *→ Pile NLP, §3*

## I

**IAA (Inter-Annotator Agreement)** — Mesure de l'accord entre deux annotateurs ou plus sur les mêmes données. Pour la NER, le kappa de Cohen est la mesure standard ; un kappa inférieur à 0.7 signale des consignes d'annotation ambiguës. L'IAA sert aussi de borne supérieure pour évaluer les modèles : un modèle qui atteint le niveau d'accord inter-humains a atteint sa limite pratique. *→ Chapitres 5, 6*

**Icône / Indice / Symbole (Peirce)** — Trichotomie peircéenne des types de signes. Une icône ressemble à ce qu'elle représente (dessin, onomatopée). Un indice est causalement lié à son objet (fumée → feu). Un symbole est relié à son objet par pure convention (le mot *chat*, un drapeau). La plupart des signes linguistiques sont des symboles. *→ Pile NLP, §Panorama*

**Implicature conversationnelle** — Contenu inféré dans un échange mais non exprimé littéralement, découlant de la violation apparente d'une maxime de Grice. *"Peux-tu passer le sel ?"* implique une demande d'action, non une question sur la capacité physique de l'interlocuteur. La violation délibérée de la maxime de manière produit des effets ironiques ou rhétoriques. *→ Pile NLP, §6*

**INT8 (quantisation)** — Représentation des poids ou des activations d'un réseau de neurones sur 8 bits entiers, réduisant l'empreinte mémoire de moitié par rapport à FP16. La quantisation dynamique INT8 ne modifie que les poids (les activations restent en FP16/FP32) ; la quantisation statique quantise aussi les activations après une phase de calibration. *→ Chapitres 9, 10*

## J

**Jensen-Shannon (divergence de)** — Mesure symétrique de la différence entre deux distributions de probabilités, bornée dans [0, 1]. Dérivée de la divergence de Kullback-Leibler, elle est utilisée en production pour la détection de drift : si la distribution courante des types d'entités s'éloigne de la distribution baseline d'un score JSD supérieur à un seuil, une alerte est déclenchée. *→ Chapitre 9, §5.2*

**JSON-LD** — Format d'échange JSON étendu avec un contexte `@context` qui mappe les clés vers des URIs sémantiques (vocabulaires comme schema.org ou un ontologie sur mesure). Utilisé dans ce cours pour sérialiser le graphe de connaissances médiéval de façon interopérable avec le web sémantique, tout en conservant la lisibilité du JSON. *→ Chapitres 7, 8*

## K

**Kappa de Cohen** — *Voir Cohen (kappa de).*

**KV-cache** — Mécanisme de mise en cache des clés (*keys*) et valeurs (*values*) calculées pour les tokens déjà traités dans un modèle Transformer auto-régressif, évitant de les recalculer à chaque étape de génération. Critique pour les LLM (où le cache peut dépasser plusieurs gigaoctets), négligeable pour les encodeurs comme CamemBERT sur des séquences courtes. *→ Chapitre 9, §3.3*

## L

**Langue / parole (distinction de Saussure)** — La *langue* est le système abstrait partagé par une communauté linguistique : l'ensemble des règles et des valeurs qui constituent le code. La *parole* est l'acte individuel de production linguistique, sujet aux accidents de la performance. Reprise par Chomsky sous les termes *compétence* / *performance*, cette distinction structure la réflexion sur ce qu'un modèle de langage apprend réellement. *→ Pile NLP, §Panorama*

**LDA (Latent Dirichlet Allocation)** — Modèle génératif probabiliste pour la modélisation thématique (Blei et al. 2003). Il suppose que chaque document est un mélange de *K* topics, chaque topic étant une distribution sur le vocabulaire. L'inférence (par variationnel Bayes ou échantillonnage de Gibbs) détermine simultanément les topics et leur répartition dans les documents. *→ Chapitre 7, §1.2*

**Lemme / lemmatisation** — Le lemme est la forme canonique d'un mot : l'infinitif pour un verbe (*porter*), le masculin singulier pour un adjectif, la forme du cas régime pour un nom médiéval. La lemmatisation ramène toute forme fléchie à son lemme. Distincte du stemming (qui tronque sans analyse), elle requiert un lexique ou un modèle morphologique. *→ Chapitres 4, 5 ; Pile NLP, §2*

**LGeRM** — Moteur de lemmatisation intégré au DMF, développé par Gilles Souvay (ATILF). Il s'appuie sur une base de formes connues et sur un ensemble de règles graphémiques et morphologiques spécifiques du moyen français (XIVe–XVe siècles). LGeRM est accessible via l'interface web du DMF et a inspiré la conception des règles de normalisation du Chapitre 4. *→ Chapitre 4, §3.2*

**LoRA (Low-Rank Adaptation)** — Méthode PEFT (Hu et al. 2022) qui décompose les mises à jour des matrices de poids en produit de deux matrices de faible rang : $\Delta W = BA$ avec $r \ll d$. Seules $A$ et $B$ sont entraînées ; les poids originaux restent gelés. Le rapport $\alpha/r$ contrôle l'amplitude de l'adaptation. Avec $r = 8$, LoRA entraîne typiquement moins de 1 % des paramètres du modèle. *→ Chapitre 3*

**LLM (Large Language Model)** — Modèle de langage de très grande taille (milliards de paramètres) pré-entraîné sur d'immenses corpus textuels. Les LLM modernes (GPT-4, Claude, Llama) sont capables de réaliser des tâches NLP arbitraires avec quelques exemples (*few-shot*) ou sans exemples (*zero-shot*), mais leur opacité et leur tendance à halluciner limitent leur usage en production scientifique. *→ Chapitre 1, §9*

**Log-vraisemblance pseudo (PLL)** — Approximation de la probabilité d'un texte sous un modèle de langage masqué (comme CamemBERT), calculée en masquant chaque token successivement et en sommant les log-probabilités. La PLL permet d'utiliser CamemBERT comme scorer pour arbitrer entre deux variantes orthographiques d'un même mot. *→ Chapitre 4, §3*

## M

**mBART-50** — Modèle de traduction multilingue seq2seq pré-entraîné sur 50 langues (Tang et al. 2020). Avec 620M paramètres, il produit des traductions plus fluides qu'Opus-MT sur des textes longs et complexes, au prix d'une empreinte mémoire plus grande. Recommandé pour la traduction du moyen français quand le corpus parallèle dépasse 2 000 paires. *→ Chapitre 11, §4.4*

**Maximes de Grice** — Quatre principes qui régissent la coopération dans un échange verbal (Grice 1975) : maxime de quantité (être assez informatif, pas plus), de qualité (ne pas dire le faux), de relation (être pertinent), de manière (être clair et ordonné). Leur violation délibérée produit des implicatures conversationnelles. *→ Pile NLP, §6*

**Mécanisme d'attention** — *Voir Attention (mécanisme d').*

**Méronymie** — Relation lexicale de la partie au tout : *moteur* est une méronyme de *voiture*, *chapitre* est une méronyme de *livre*. L'inverse (le tout par rapport à la partie) est l'holonymie. En inférence textuelle et en construction d'ontologies, la méronymie complète la hiérarchie hyponymique. *→ Pile NLP, §3*

**Métaphore conceptuelle** — Structure cognitive (Lakoff et Johnson 1980) qui projette les inférences d'un domaine concret (source) sur un domaine abstrait (cible). La métaphore *LE DÉBAT EST UNE GUERRE* explique des expressions comme *attaquer un argument*, *défendre sa position*, *capituler*. En TALN, la détection de métaphores vise à distinguer les usages littéraux et métaphoriques. *→ Pile NLP, §3*

**Métonymie** — Figure rhétorique et phénomène cognitif qui substitue un terme par un autre entretenant une relation de contiguïté avec lui (*"Lire Proust"* pour *"lire l'œuvre de Proust"*). Distincte de la métaphore (basée sur la ressemblance), la métonymie est fondamentale en sémantique du discours et en analyse de textes institutionnels. *→ Pile NLP, §9*

**Model card** — Document normalisé qui décrit un modèle de machine learning : tâche, données d'entraînement (avec split_hash), métriques de performance (par catégorie), limitations connues, biais analysés, et usages recommandés ou déconseillés. La model card est une exigence éthique et scientifique pour tout modèle publié ou déployé. *→ Chapitres 9, 10*

**Morphème** — La plus petite unité linguistique porteuse de sens ou de fonction grammaticale. Dans *chantions*, on distingue *chant-* (morphème lexical), *-i-* (morphème d'imparfait) et *-ons* (morphème de 1re personne du pluriel). La morphologie étudie les combinaisons et les modifications des morphèmes. *→ Chapitres 4 ; Pile NLP, §2*

**Morphologie flexionnelle / dérivationnelle** — La morphologie flexionnelle produit les formes d'un même lexème selon les catégories grammaticales (*chante, chantons, chanté* partagent le lemme *chanter*). La morphologie dérivationnelle crée de nouveaux lexèmes par affixation (*chanter → chanteur*), changeant souvent la catégorie grammaticale et le sens. *→ Pile NLP, §2*

**Moyen français** — Étape de l'histoire du français s'étendant approximativement de 1330 à 1500, entre la fin de l'ancien français et le début du français préclassique. Le moyen français se caractérise par une orthographe non standardisée, des traces résiduelles de la déclinaison à deux cas, et un vocabulaire partiellement latinisant. C'est la langue des corpus CREMMA traités dans ce cours. *→ Chapitres 4, 5*

**Multi-head attention** — Extension de l'attention qui calcule plusieurs mécanismes d'attention en parallèle dans des sous-espaces de dimension réduite, puis concatène leurs sorties. Chaque "tête" peut ainsi apprendre à capturer différents types de relations (syntaxiques, sémantiques, coréférentielles). Dans BERT-base, on compte 12 têtes de dimension 64 chacune. *→ Chapitre 1, §3.2*

**mT5** — Version multilingue de T5 (Text-to-Text Transfer Transformer) pré-entraînée sur 101 langues, utilisée dans ce cours pour le fine-tuning de la normalisation orthographique. La variante mT5-small (77M paramètres) est entraînable sur CPU en quelques dizaines de minutes. *→ Chapitres 3, 4*

## N

**NER (Named Entity Recognition)** — Tâche qui consiste à identifier et classifier les entités nommées dans un texte selon un schéma prédéfini de types. Dans ce cours, le schéma comporte cinq types (PER, LOC, DATE, ORG, TITLE) et utilise le schéma BIO ou BIOES. L'évaluation est réalisée au niveau des spans avec seqeval. *→ Chapitres 5, 6*

**NF4 (Normal Float 4-bit)** — Format de quantisation 4 bits optimisé pour des distributions de poids normales, introduit dans QLoRA (Dettmers et al. 2023). Contrairement à la quantisation entière uniforme, NF4 place ses 16 niveaux de quantification aux quantiles de la distribution normale, minimisant l'erreur de quantisation pour les poids pré-entraînés. *→ Chapitres 3, 9*

**NMF (Non-negative Matrix Factorization)** — Technique de décomposition matricielle qui factorise une matrice TF-IDF $X \approx WH$ en deux matrices non négatives : $H$ contient les topics (distributions mot-topic), $W$ les poids de chaque topic dans chaque document. NMF produit souvent des topics plus "purs" que LDA sur les petits corpus. *→ Chapitre 7, §1.3*

**NMT (Neural Machine Translation)** — Approche de la traduction automatique basée sur des réseaux de neurones, en particulier des architectures encoder-decoder avec mécanisme d'attention. Les modèles NMT génèrent la traduction mot à mot de façon auto-régressive, en conditionnant chaque token généré sur les tokens précédents et sur la représentation complète de la source. *→ Chapitre 11*

**Normalisation orthographique** — Opération qui convertit les formes graphiques médiévales non standardisées vers leurs équivalents en français moderne normalisé. Elle comprend la résolution des alternances phonographiques (u/v, i/j), la correction des terminaisons, la résolution des abréviations, et le lookup dans le DMF pour les formes rares. *→ Chapitre 4*

**Normalisation Unicode** — Première étape du pipeline de normalisation orthographique : unification des représentations composées et décomposées des caractères accentués (NFC), et translittération des caractères médiévaux spéciaux (p barré, ligatures æ/œ). *→ Chapitre 4, §1*

## O

**Observabilité** — Capacité à comprendre l'état interne d'un système en production à partir de ses sorties externes : logs structurés, métriques de latence, distribution des prédictions. En TALN de production, l'observabilité permet de détecter le concept drift, les dégradations de performance, et les comportements anormaux avant qu'ils n'impactent les utilisateurs. *→ Chapitre 9, §5*

**ONNX Runtime** — Moteur d'inférence open-source indépendant du framework d'entraînement. Un modèle PyTorch exporté en format ONNX peut être exécuté sur n'importe quel hardware (CPU, GPU, NPU) sans dépendance à PyTorch, avec des optimisations de graphe qui accélèrent l'inférence de 1.5× à 4× selon la configuration. *→ Chapitre 9, §3.2*

**Opus-MT** — Suite de modèles de traduction automatique neurale (MarianMT) développée par Helsinki-NLP, couvrant des centaines de paires de langues. Dans ce cours, Helsinki-NLP/opus-mt-fr-ROMANCE est le modèle de départ pour le fine-tuning vers la traduction moyen français → français moderne. *→ Chapitre 11, §4.3*

**OOV (Out-Of-Vocabulary)** — Terme ou token absent du vocabulaire du modèle, ce qui empêche le modèle de le représenter directement. La tokenisation par sous-mots (BPE, WordPiece) réduit pratiquement le problème OOV à zéro en découpant tout mot en caractères connus, au prix d'une fragmentation des unités linguistiques. *→ Chapitre 1, §7*

## P

**PAGE XML** — Format de description structurée d'une page de document numérisé, développé dans le cadre du projet READ. Il encode la structure physique de la page (zones de texte, lignes, mots) avec leurs polygones de localisation, leurs transcriptions, et leurs scores de confiance. C'est le format de sortie du Volet 1 (HTR), et les polygones constituent le lien vers l'image source conservé tout au long du pipeline. *→ Chapitre 2*

**Parsing syntaxique** — Construction automatique de la structure syntaxique d'une phrase sous forme d'arbre syntagmatique ou de graphe de dépendances. Les algorithmes classiques (CYK, Earley, arc-eager) opèrent sur des grammaires formelles ; les parsers neuronaux modernes (biaffine) apprennent directement les structures depuis des corpus annotés. *→ Pile NLP, §4*

**PEFT (Parameter-Efficient Fine-Tuning)** — Famille de méthodes qui adaptent un modèle pré-entraîné à une tâche en n'entraînant qu'un petit sous-ensemble de ses paramètres. LoRA, les adapters, le prefix tuning et le prompt tuning sont les principales méthodes PEFT. Elles permettent de fine-tuner des modèles de milliards de paramètres avec des ressources GPU limitées. *→ Chapitre 3*

**Perplexité** — Métrique d'évaluation des modèles de langage, définie comme l'exponentielle de l'entropie croisée : $\text{PPL} = e^{-\frac{1}{N}\sum_i \log p(w_i | w_{<i})}$. Une perplexité plus basse indique un modèle qui prédit mieux le texte. Elle est peu informative pour la comparaison de modèles sur des tokenisations différentes. *→ Chapitre 1, §6*

**Phonème** — La plus petite unité sonore distinctive d'une langue. Deux sons qui forment une paire minimale (*pain / bain*) correspondent à deux phonèmes distincts (/p/ et /b/ en français). Un phonème est une abstraction — il peut être réalisé de différentes façons phonétiques selon le contexte (*allophone*). *→ Pile NLP, §1*

**Phonologie** — Couche inférieure de la pile NLP qui étudie l'organisation des sons en systèmes fonctionnels. Elle distingue les phonèmes (unités contrastives) des allophones (variantes non contrastives), et étudie les processus phonologiques (assimilation, élision, liaison) qui régissent les interactions entre sons en contexte. *→ Pile NLP, §1*

**pie-extended** — Outil de lemmatisation et d'étiquetage morphosyntaxique pour le français médiéval (Camps, Clérice et al. 2021), basé sur l'architecture Pie. Il produit des annotations dans le schéma Cattex, convertibles vers Universal Dependencies. C'est l'outil de référence pour le POS-tagging des corpus CREMMA dans ce cours. *→ Chapitres 5, 6*

**PMI (Pointwise Mutual Information)** — Mesure de l'association statistique entre deux événements : $\text{PMI}(w_1, w_2) = \log \frac{P(w_1,w_2)}{P(w_1)P(w_2)}$. Une PMI positive indique que deux mots cooccurrent plus souvent qu'attendu par hasard, signal d'une collocation ou d'une relation sémantique. La PPMI (PMI positive) supprime les valeurs négatives peu fiables. *→ Pile NLP, §3*

**Polysémie** — Propriété d'un mot qui possède plusieurs sens liés par une relation étymologique ou métonymique (*canal* : voie d'eau, conduit, chaîne de communication). Distincte de l'homonymie (deux mots de formes identiques mais d'origines indépendantes), la polysémie est résolue par la désambiguïsation lexicale (WSD). *→ Pile NLP, §3*

**Post-édition** — Révision humaine des sorties automatiques d'un système de traduction ou de normalisation pour corriger les erreurs résiduelles. En traduction automatique historique, la post-édition est indispensable car les faux amis sémantiques (comme *liez* = *libres*) ne sont pas toujours détectés automatiquement. *→ Chapitre 11, §7*

**POS-tagging** — Tâche d'étiquetage morphosyntaxique qui assigne à chaque token d'un texte sa catégorie grammaticale (nom, verbe, adjectif, etc.) selon un schéma d'étiquettes. Universal Dependencies définit 17 étiquettes UPOS universelles et des étiquettes spécifiques à la langue (XPOS). Sur le moyen français normalisé, Stanza `frm` est le modèle recommandé. *→ Chapitre 5, §2*

**Pragmatique** — Couche de la pile NLP qui étudie le sens en contexte — ce que le locuteur veut dire, pas seulement ce que les mots signifient. Elle comprend l'étude des actes de langage, des implicatures, de la politesse, et des maximes conversationnelles. *→ Pile NLP, §6*

**Pré-entraînement / fine-tuning** — Paradigme dominant en NLP moderne : un modèle est d'abord pré-entraîné sur un immense corpus général (modèle de langue, MLM), puis adapté (fine-tuné) sur une tâche spécifique avec un corpus plus petit. Le pré-entraînement apprend des représentations générales du langage ; le fine-tuning spécialise ces représentations. *→ Chapitres 1, 3*

**Présupposition** — Contenu sémantique que l'on tient pour acquis avant même d'asserter la phrase, et qui survit à la négation. *"Jean a arrêté de fumer"* présuppose que Jean fumait — présupposition qui est maintenue même sous la forme négative *"Jean n'a pas arrêté de fumer"*. *→ Pile NLP, §5*

**Probing** — Méthode d'interprétabilité qui entraîne des classifieurs linéaires simples sur les représentations internes d'un modèle pré-entraîné pour déterminer quelle information linguistique ces représentations encodent. Tenney et al. (2019) ont montré que les couches inférieures de BERT encodent les informations morphologiques, les couches médianes les informations syntaxiques, et les couches supérieures les informations sémantiques. *→ Pile NLP, §Conclusion*

**Prosodie** — Composante de la phonologie qui étudie les propriétés suprasegmentales : l'accent, le ton, l'intonation, le rythme, et la durée. En français, l'intonation montante transforme une assertion en question. La prosodie peut signaler des frontières syntaxiques et pragmatiques que la grammaire laisserait ambiguës. *→ Pile NLP, §1*

**pytest** — Framework de test Python utilisé pour écrire et exécuter des tests automatisés. Dans ce cours, la suite pytest du pipeline NER vérifie le schéma JSON des réponses, la non-régression du F1 sur un corpus de test fixe, les contraintes de latence (p99 < 500ms), et l'idempotence des prédictions. *→ Chapitre 10, §4*

## Q

**QLoRA** — Variante de LoRA qui combine la quantisation NF4 des poids du modèle de base (4 bits) avec l'entraînement des adaptateurs LoRA en précision complète (BF16), avec pagination mémoire pour les gradients. Permet de fine-tuner des modèles de plusieurs milliards de paramètres sur un seul GPU grand public. *→ Chapitre 3, §4*

**Quantisation** — Famille de techniques qui réduisent la précision numérique des poids (et éventuellement des activations) d'un modèle, réduisant son empreinte mémoire et accélérant l'inférence. Les principaux formats sont FP16, INT8, INT4 (NF4), avec un trade-off entre compression et qualité. *→ Chapitre 9, §1*

## R

**RAG (Retrieval-Augmented Generation)** — Architecture qui combine un modèle génératif avec un index de récupération d'information. À chaque requête, le système récupère les passages les plus pertinents depuis une base de connaissance (textes, triplets du graphe), les injecte dans le prompt du LLM, et génère une réponse contextualisée. *→ Chapitre 7, §3.3*

**RDF / Turtle** — Resource Description Framework est le modèle de données du web sémantique W3C, représentant les connaissances sous forme de triplets (sujet, prédicat, objet) avec des URIs. Turtle est une syntaxe compacte et lisible pour sérialiser du RDF. Utilisé pour exporter le graphe de connaissances médiéval vers des formats interopérables. *→ Chapitre 7, §3.2*

**Rang LoRA (r)** — Hyperparamètre de LoRA qui contrôle la dimension des matrices d'adaptateurs. Un rang faible ($r = 4$ ou $8$) entraîne très peu de paramètres mais limite la capacité d'adaptation ; un rang élevé ($r = 64$) se rapproche du full fine-tuning en coût mémoire. L'hypothèse du faible rang intrinsèque suggère que $r = 8$ ou $16$ suffit pour la plupart des tâches NLP. *→ Chapitre 3, §3.3*

**Registre (stylistique)** — Ensemble des choix linguistiques (lexicaux, syntaxiques, morphologiques) adaptés à une situation communicative donnée. Un même contenu peut être exprimé en registre soutenu, courant ou familier. En TALN, la détection du registre est utile pour la génération de texte adapté à un public cible et pour l'analyse des corpus hétérogènes. *→ Pile NLP, §8*

**Relation d'extraction** — Tâche qui identifie les relations sémantiques entre paires d'entités nommées dans un texte. Dans ce cours, les relations extraites incluent *porte_titre* (PER–TITLE), *réside_à* (PER–LOC), *agit_lors_de* (PER–DATE). L'extraction peut être réalisée par règles syntaxiques ou par prompting LLM avec un schéma structuré. *→ Chapitres 7, 8*

**Réseau de neurones** — Modèle computationnel composé de couches de neurones artificiels (unités de calcul non linéaires) reliées par des poids ajustables. En NLP, les architectures fondamentales sont les réseaux récurrents (LSTM, GRU) pour les séquences, les réseaux convolutifs (CNN) pour les caractéristiques locales, et les Transformers pour les dépendances à longue distance. *→ Chapitre 1*

**Résolution de coréférence** — Tâche qui identifie automatiquement toutes les expressions d'un texte référant au même individu ou entité, et les regroupe en chaînes. SpanBERT (Joshi et al. 2020) est le modèle de référence. Elle est indispensable avant la construction d'un graphe de connaissances pour éviter de créer des nœuds distincts pour le même individu. *→ Chapitre 7*

**Rhétorique** — Couche supérieure de la pile NLP, discipline héritée d'Aristote, qui étudie les stratégies discursives de persuasion et d'expression. Les trois modes de preuve rhétorique — logos (argument rationnel), éthos (crédibilité), pathos (appel aux émotions) — structurent l'argumentation mining. Les figures de style (métaphore, métonymie, chiasme, etc.) sont ses objets classiques. *→ Pile NLP, §9*

**Rôle thématique** — Relation sémantique entre un prédicat et ses arguments : Agent (qui fait l'action), Patient (qui subit), Thème (ce qui est déplacé), Instrument, Localisation, etc. Introduits par Fillmore (1968), les rôles thématiques sont à la base de l'étiquetage des rôles sémantiques (SRL) et des bases FrameNet et PropBank. *→ Pile NLP, §5*

**RST (Rhetorical Structure Theory)** — Théorie (Mann et Thompson 1988) qui décrit la structure d'un texte comme un arbre de relations rhétoriques entre segments. Chaque relation (élaboration, contraste, cause, concession, etc.) distingue un noyau (segment principal) et un satellite (segment dépendant). RST est la base de l'analyse automatique de la structure discursive. *→ Pile NLP, §7*

## S

**Scaled dot-product attention** — Formule de calcul de l'attention dans le Transformer : $\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$. Le facteur $\sqrt{d_k}$ normalise les produits scalaires pour éviter que la softmax ne sature dans des régions à gradient quasi-nul lorsque la dimension des vecteurs est grande. *→ Chapitre 1, §3.1*

**seqeval** — Bibliothèque Python d'évaluation standard pour les tâches d'étiquetage de séquences (NER, chunking). Elle évalue au niveau des spans (entités complètes) plutôt qu'au niveau des tokens, et reconnaît les schémas BIO, BIOES, etc. Dans ce cours, elle est utilisée pour calculer le F1 micro et macro par type d'entité. *→ Chapitre 6, §4*

**Sémantique** — Couche de la pile NLP qui étudie le sens : ce que les expressions linguistiques signifient, les relations de référence entre les expressions et le monde, et la façon dont le sens se calcule compositionnellement depuis les parties. Elle se distingue de la pragmatique (sens en contexte) et englobe la WSD, le SRL, la logique de premier ordre, et les représentations vectorielles. *→ Pile NLP, §5*

**Sémantique compositionnelle** — Principe selon lequel le sens d'une expression complexe est une fonction du sens de ses parties et de leurs relations syntaxiques. Formalisé par Montague (1970) via le λ-calcul, il est à la base des systèmes de compréhension du langage naturel qui construisent des représentations logiques depuis les phrases. *→ Pile NLP, §5*

**Sémiose / sémiotique** — La sémiose est le processus général par lequel quelque chose (un signe) représente quelque chose d'autre pour un interprétant — concept central de la philosophie du signe de Peirce. La sémiotique est la science des signes et des processus de signification, fondée par Peirce et Saussure. Morris (1938) en a formalisé les trois branches : syntaxe, sémantique, pragmatique. *→ Pile NLP, §Panorama*

**Sentence embeddings (SBERT)** — Plongements vectoriels de phrases entières (par opposition aux plongements de tokens) produits par un modèle Sentence-BERT, entraîné à produire des représentations similaires pour des phrases sémantiquement proches. Utilisés dans BERTopic pour calculer la similarité entre documents avant le clustering HDBSCAN. *→ Chapitre 7, §1.4*

**Serving** — Ensemble des techniques et infrastructure permettant de mettre un modèle ML en production de façon fiable, performante et scalable. Les serveurs d'inférence spécialisés (vLLM, TGI, ONNX Runtime) optimisent le batching, la gestion mémoire, et la quantisation pour maximiser le débit et minimiser la latence. *→ Chapitre 9, §3*

**Signifiant / signifié** — Dichotomie saussurienne au cœur de la conception du signe linguistique. Le signifiant est l'image acoustique ou graphique (la séquence de sons ou de lettres) ; le signifié est le concept mental auquel le signifiant renvoie. Leur relation est arbitraire (pas motivée par ressemblance) et différentielle (chaque signe vaut par opposition aux autres). *→ Pile NLP, §Panorama*

**Softmax** — Fonction d'activation qui transforme un vecteur de scores réels en une distribution de probabilités dont la somme vaut 1 : $\text{softmax}(z_i) = e^{z_i} / \sum_j e^{z_j}$. Utilisée dans la couche de classification NER (sur les étiquettes) et dans le mécanisme d'attention (sur les poids). La température dans la distillation "adoucit" la softmax en divisant les logits par $T > 1$. *→ Chapitres 1, 5*

**SPARQL** — Langage de requête W3C pour les graphes RDF. Permet d'interroger le graphe de connaissances médiéval avec des requêtes comme *"tous les sénéchaux ayant signé des actes en Normandie entre 1340 et 1360"*. SPARQL 1.1 supporte les requêtes SELECT, CONSTRUCT, ASK et DESCRIBE, ainsi que les opérations ensemblistes et les agrégats. *→ Chapitre 7, §3.3*

**Stanza** — Bibliothèque NLP de l'Université de Stanford avec des modèles pré-entraînés pour de nombreuses langues. Pour le moyen français, le modèle `frm` (entraîné sur l'Arboratoire du français médiéval) couvre la tokenisation, le POS-tagging, la lemmatisation et le parsing de dépendances. Distinct de `fro` (vieux français), `frm` est adapté aux textes des XIVe–XVe siècles. *→ Chapitre 5, §2.2*

**Stemming** — Opération de réduction d'un mot à sa racine morphologique par troncature heuristique, sans analyse linguistique. L'algorithme de Porter (1980) est le stemmer anglais de référence. Moins précis que la lemmatisation (qui nécessite un lexique et produit un vrai lemme), le stemming est plus rapide et suffisant pour certaines tâches de recherche d'information. *→ Pile NLP, §2*

**Stylométrie** — Application de méthodes statistiques à l'analyse du style linguistique, principalement pour l'attribution d'auteur. Elle mesure des traits comme les fréquences des mots fonctionnels, la longueur moyenne des phrases, la richesse lexicale (hapax ratio), et la distribution des n-grammes. La mesure Delta de Burrows (2002) est la référence de la stylométrie computationnelle. *→ Pile NLP, §8*

**Subword tokenisation** — Stratégie de tokenisation qui découpe les mots en sous-mots fréquents plutôt que de les traiter comme des unités atomiques. BPE, WordPiece et Unigram LM sont les trois algorithmes principaux. Elle résout le problème OOV et permet de représenter des mots rares ou nouveaux à partir de sous-mots connus, au prix d'une fragmentation des unités morphologiques. *→ Chapitre 1, §7*

**Synonymie** — Relation lexicale entre deux mots ou expressions de sens équivalent ou très proche. La synonymie absolue (sens identiques dans tous les contextes) est rare ; la synonymie partielle (sens voisins, différences de registre ou de nuance) est courante. *Commencer* et *débuter* sont des synonymes partiels avec une différence de registre. *→ Pile NLP, §3*

## T

**TALN (Traitement automatique des langues)** — Discipline à l'intersection de la linguistique, de l'informatique et de l'intelligence artificielle dont l'objet est de concevoir des algorithmes et des systèmes capables de traiter, analyser, comprendre et générer le langage humain. Ses applications incluent la traduction automatique, la reconnaissance vocale, l'extraction d'information, la génération de texte et les interfaces conversationnelles. *→ Tout le cours*

**TEI-XML** — Text Encoding Initiative, standard d'encodage XML des textes en humanités numériques depuis 1987. Il définit des centaines de balises pour représenter la structure physique et logique des textes, les annotations linguistiques (POS, lemmes, entités nommées), et les métadonnées. Dans ce cours, `<lb facs="polygon_ref">` ancre chaque ligne annotée à sa position physique dans le manuscrit. *→ Chapitres 7, 8*

**TF-IDF (Term Frequency-Inverse Document Frequency)** — Pondération classique en recherche d'information qui combine la fréquence d'un terme dans un document (TF) et l'inverse de la proportion de documents contenant ce terme (IDF). Un terme fréquent dans un document mais rare dans le corpus reçoit un score élevé — il est caractéristique de ce document. *→ Pile NLP, §3*

**TGI (Text Generation Inference)** — Serveur d'inférence haute performance développé par Hugging Face, optimisé pour les modèles génératifs. Il implémente le continuous batching, la quantisation INT8, le parallel tensor et d'autres optimisations pour maximiser le débit sur GPU. Recommandé pour la mise en production des LLM dans des contextes à fort trafic. *→ Chapitre 9, §3.1*

**Token** — Unité atomique de traitement dans un pipeline NLP. Selon le système, un token peut être un mot, un sous-mot (BPE), un caractère, ou une ponctuation. Dans les Transformers modernes, un token est une unité du vocabulaire BPE ; en français, la correspondance approximative est 1 token ≈ 0.75 mot. *→ Chapitre 1*

**Tokenisation** — Opération qui segmente un texte brut en tokens. La tokenisation par sous-mots (BPE, WordPiece) est standard pour les Transformers ; la tokenisation par règles (espaces et ponctuation) est adaptée pour des langues à orthographe régulière ; des approches spécialisées existent pour les manuscrits médiévaux où les espaces sont irréguliers. *→ Chapitres 1, 5 ; Pile NLP, §2*

**Topic modelling** — Famille de méthodes non supervisées qui inférent des thèmes latents (topics) depuis la distribution des mots dans un corpus. LDA (Blei et al. 2003) est le modèle probabiliste classique ; BERTopic (Grootendorst 2022) utilise des embeddings contextuels. Utilisé dans ce cours pour analyser la structure thématique du corpus de chartes médiévales. *→ Chapitres 7, 8*

**Trait distinctif (phonologie)** — Propriété binaire ou multivaluée qui définit un phonème en le distinguant des autres. Les traits binaires classiques de Jakobson, Halle et Fant (1952) incluent [±voisé], [±nasal], [±occlusive], etc. Chaque phonème est défini par son ensemble de traits. Les processus phonologiques (assimilation, mutation) se décrivent comme des changements de valeurs de traits. *→ Pile NLP, §1*

**Transformer (architecture)** — Architecture de réseau de neurones introduite par Vaswani et al. (2017) qui repose exclusivement sur le mécanisme d'attention multi-têtes, sans récurrence ni convolution. Elle comprend un encodeur (stacks de couches d'auto-attention et de réseaux feed-forward) et un décodeur (avec en plus une attention croisée sur l'encodeur). BERT, GPT, T5, mT5 et CamemBERT sont tous des variantes de cette architecture. *→ Chapitre 1*

## U

**UMAP (Uniform Manifold Approximation and Projection)** — Algorithme de réduction de dimensionnalité non linéaire (McInnes et al. 2018) qui préserve la structure locale et globale des données mieux que t-SNE, et est plus scalable que PCA. Dans BERTopic, UMAP réduit les embeddings 768D en 3–5 dimensions avant le clustering HDBSCAN. *→ Chapitres 7, 8*

**Universal Dependencies (UD)** — Schéma d'annotation syntaxique multilingue standardisé (Nivre et al. 2016, 2020) qui définit 17 catégories UPOS et un ensemble de relations de dépendance comparables entre langues. Il couvre plus de 100 langues et constitue la ressource de référence pour le POS-tagging et le parsing interlinguistiques. Dans ce cours, les annotations UD de Stanza `frm` sont exportées en CoNLL-U. *→ Chapitres 5 ; Pile NLP, §4*

## V

**Valence verbale** — Propriété lexicale d'un verbe qui spécifie le nombre et le type de ses compléments obligatoires (actants). Un verbe monovalent (*dormir*) n'a qu'un actant (le dormeur) ; un verbe trivalent (*donner*) en a trois (donateur, donataire, chose donnée). La notion est due à Tesnière (1959) et est centrale dans les grammaires de dépendances. *→ Pile NLP, §4*

**Variation graphique** — Phénomène par lequel un même mot est représenté par plusieurs orthographes différentes dans un corpus médiéval. Le mot *roi* apparaît sous les formes *roi, roy, roys, rei, rois*, etc. Cette variation est la principale source de difficulté du TALN médiéval : elle multiplie les types hors-vocabulaire pour un modèle entraîné sur le français moderne. *→ Chapitre 4, §1.1*

**Viterbi (algorithme de)** — Algorithme de programmation dynamique qui trouve la séquence d'états cachés la plus probable dans un HMM ou la séquence d'étiquettes de score maximal dans un CRF, avec une complexité $O(n \cdot T^2)$ où $n$ est la longueur de la séquence et $T$ le nombre d'étiquettes. Il est utilisé dans les modèles CRF-NER pour garantir la cohérence des séquences d'étiquettes BIO. *→ Chapitre 5, §4.2*

**vLLM** — Serveur d'inférence pour les grands modèles de langage développé par Kwon et al. (2023), qui implémente PagedAttention — une gestion paginée du KV-cache inspirée de la pagination mémoire des OS. PagedAttention permet le partage de mémoire entre séquences avec un préfixe commun et le continuous batching, multipliant le débit par rapport aux approches naïves. *→ Chapitre 9, §3.4*

## W

**Weak supervision (supervision faible)** — Approche qui génère des annotations automatiques de qualité imparfaite (par règles, gazetier, ou modèles de bruit) pour créer des données d'entraînement sans annotation humaine exhaustive. Dans ce cours, l'annotation par gazetier est la forme de weak supervision utilisée pour initialiser le corpus NER médiéval. *→ Chapitre 6, §1.2*

**Word embeddings** — Représentations vectorielles denses de mots dans un espace de faible dimension, telles que les mots sémantiquement proches ont des représentations voisines. Word2Vec (Mikolov et al. 2013), GloVe (Pennington et al. 2014) et FastText (Bojanowski et al. 2017) sont des embeddings statiques ; BERT et CamemBERT produisent des embeddings contextuels qui varient selon le contexte d'occurrence du mot. *→ Pile NLP, §3 ; Chapitre 1*

**Word2Vec** — Modèle d'apprentissage de plongements lexicaux (Mikolov et al. 2013) disponible en deux architectures : Skip-gram (prédire les mots du contexte depuis le mot central) et CBOW (prédire le mot central depuis son contexte). Sa propriété d'arithmétique vectorielle (*roi − homme + femme ≈ reine*) a popularisé l'idée de représenter le sens comme un vecteur. *→ Pile NLP, §3*

**WordNet** — Base lexicale hiérarchique pour l'anglais (Miller 1995) qui organise les mots en synsets (ensembles de synonymes) reliés par des relations lexicales (hyponymie, méronymie, antonymie). WOLF et JeuxDeMots sont des équivalents pour le français. WordNet est la ressource de référence pour la désambiguïsation lexicale et l'inférence textuelle. *→ Pile NLP, §3*

**WordPiece** — Algorithme de tokenisation en sous-mots utilisé dans BERT et CamemBERT. Contrairement à BPE (qui fusionne les paires les plus fréquentes), WordPiece choisit les fusions qui maximisent la vraisemblance du corpus d'entraînement. Les tokens intérieurs d'un mot sont préfixés par `##` pour indiquer qu'ils ne sont pas des débuts de mot. *→ Chapitres 1 ; Pile NLP, §2*

## Z

**Zero-shot** — Capacité d'un modèle à réaliser une tâche pour laquelle il n'a vu aucun exemple lors de son entraînement, en s'appuyant uniquement sur les instructions fournies dans le prompt. Les LLM modernes réussissent souvent des tâches zero-shot grâce à leur exposition massive à des descriptions de tâches pendant le pré-entraînement. Dans ce cours, le zero-shot LLM pour la traduction médiévale est démonstratif mais non recommandé en production faute de reproductibilité. *→ Chapitre 11, §6*

---

*Glossaire rédigé pour le Master Data/IA · Module NLP · MD5 Volet 2 · 2026. Les renvois entre parenthèses indiquent les chapitres du cours où chaque notion est traitée en détail. Les définitions sont intentionnellement synthétiques ; pour l'approfondissement, consulter les bibliographies des chapitres correspondants.*
