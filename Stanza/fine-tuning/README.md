# Réentraînement (fine-tuning) du modèle de l'ancien français

Les expérimentations sont sous forme de **carnet Jupyter**, pour être exécutées avec le **GPU T4** de Google Colab.

Il existe deux versions :
1. une version jouet, entraînée sur un micro-corpus, uniquement pour les besoins de la démonstration : `stanza_medieval_colab_toy.ipynb`.
2. une version plus réaliste, mais plus lourdre, utilisant des données réelles du jeu de données CATMuS : `stanza_medieval_colab_catmus.ipynb`

Les carnets sont auto-suffisants. vVous pouvez les lancer directement (en vérifiant que le GPU est activé).

Au besoin, vous pouvez récupérer le jeu de données qui aura été engendré par le script d'analyse linguistique (étape 1).

Vous trouverez également, à titre d'exemple, des fichiers au format `CoNLL-U` qui est le format d'entrée du réentraînement de Stanza.
