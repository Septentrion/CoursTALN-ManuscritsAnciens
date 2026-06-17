# Traitement de la langue (ancien français)

Script Python montrant divers aspects de l'analyse linguistique d'un langue (naturelle) :

1. Tokenisation
2. Lemmatisation
3. Parties du langage (ou _parts-of-speech_, ou **POS**)
4. Analyse des dépendances syntaxiques (ou arbre syntaxique)
5. Analyse de cohérence (non présente pour l'ancien français)
6. Extraction des entités nommées (non présente pour l'ancien français, «&nbsp;patchée&nbsp;» ici par gazetier)

Vous trouverez aussi une liste des caractèristiques morphologiques reconnues par Stanza.

## Notion de `treebank`

Un **treebank** (littéralement « banque d'arbres ») est un corpus de phrases annotées syntaxiquement, où chaque phrase est représentée sous forme d'un arbre (ou d'un graphe, dans le formalisme des dépendances) qui encode sa structure grammaticale complète : découpage en mots, catégories grammaticales, traits morphologiques, et relations de dépendance entre les mots.

## Dans le contexte spécifique de Stanza

Stanza s'appuie exclusivement sur le format **Universal Dependencies (UD)**, un projet international qui vise à annoter des treebanks dans des dizaines de langues avec un schéma d'annotation commun (les mêmes catégories **UPOS**, les mêmes relations de dépendance comme `nsubj`, `obj`, `root`, etc.), pour que les modèles et les outils soient comparables d'une langue à l'autre. Chaque treebank UD est un ensemble de fichiers `.conllu` — exactement le format que l'on retrouve dans le carnet de fine-tuning pour la préparation des données.

## La convention de nommage `UD_Langue-NomDuTreebank`

C'est ce que nous avons utilisé directement dans votre notebook. Chaque treebank UD porte un nom à deux parties : la langue, puis un nom propre qui identifie la source ou le projet d'annotation, parce qu'une même langue peut avoir plusieurs treebanks distincts (annotés par des équipes différentes, sur des types de textes différents). Pour le français moderne, il existe par exemple `UD_French-GSD`, `UD_French-Sequoia`, `UD_French-ParTUT` — trois treebanks différents, donc potentiellement trois modèles `fr` légèrement différents selon celui choisi à l'entraînement.

Pour l'ancien français, le treebank que `fro` utilise s'appelle `UD_Old_French-PROFITEROLE` — c'est le nom que nous avons rencontré en inspectant `resources.json`, où Stanza l'identifie sous le nom de code `profiterole`. C'est ce treebank précis, avec ses ~227 000 tokens et ses 19 765 phrases couvrant douze textes du IXe au XIIIe siècle, qui constitue toutes les connaissances que le modèle `fro` possède sur la syntaxe et la morphologie de l'ancien français — c'est exactement la limite structurelle qu'on a discutée pour expliquer le faible taux de couverture du lemmatiseur, et les scores modestes de votre premier fine-tuning sur 8 phrases.

## Pourquoi c'est le treebank, et pas le modèle, qui porte ce nom

C'est une distinction importante dans l'architecture de Stanza : le treebank est la *donnée* (les phrases annotées), tandis que le modèle neuronal (tagger, parser, etc.) est *entraîné sur* ce treebank. Le shorthand `fro_custom` que nous utilisons dans le carnet consacré au fine-tuning suit exactement cette même logique : nous créons un treebank personnalisé (`UD_Old_French-Custom`), distinct du treebank officiel `Profiterole`, à partir des phrases auto-annotées de CATMuS — c'est pour cela que `prepare_pos_treebank` exige cette convention de nommage précise, et pourquoi le `shorthand` se retrouve ensuite dans le nom des fichiers `.conllu` (`fro_custom-ud-train.conllu`, etc.) que le script d'entraînement va chercher.
