# Syllabus

## Fondements des Transformers & prise en main du corpus

### Chapitre 1 - Architecture Transformer de A à Z
- Limites des RNN/LSTM : vanishing gradient, séquentialité
- Self-attention : queries, keys, values — formule scaled dot-product
- Positional encoding : sinusoïdal vs appris (RoPE, ALiBi)
- Multi-head attention : pourquoi plusieurs têtes, interprétation
- Encoder-only (BERT), decoder-only (GPT), encoder-decoder (T5, BART)
- Vocabulaire BPE, tokenisation SentencePiece — enjeux sur le vieux français
- Cas pratique : visualiser les têtes d'attention sur un texte médiéval

### Chapitre 2 - Ingestion du data contract HTR → NLP
- Lecture du JSON livré par Volet 1 : structure, transcriptions, scores de confiance **par caractère**, candidats `[a/b]`, needs_review,  polygones
- Filtrage des lignes `needs_review` et stratégie de gestion
- EDA : longueurs, distribution des confiances, taux d'abréviations résiduelles, types de documents (chartes, registres, romans)
- Analyse exploratoire : distribution des longueurs, taux d'abréviations, langues
- Tokenisation de textes en vieux/moyen français avec CamemBERT et RoBERTa
	- taux OOV de CamemBERT sur CREMMA, stratégies character-level vs BPE
- Visualisation des embeddings avec UMAP sur un sous-corpus
- Inventaire des abréviations non résolues par la CV → priorisation pour le Jour 2
- Fixer le split train/val/test (SHA-256)

### Point de vigilance
- Le vieux français ne bénéficie pas d'un tokeniseur dédié : quantifier la perte de couverture du vocabulaire BPE sur votre corpus avant de choisir le modèle de base.

## Fine-tuning efficace — PEFT, LoRA, QLoRA

### Chapitre 3 - Adapter les LLMs avec peu de ressources
- Full fine-tuning vs parameter-efficient - Full FT vs LoRA vs QLoRA: coûts comparés (GPU-hours, mémoire)
- LoRA : décomposition de rang faible, hyperparamètres r, alpha, dropout
	- LoRA sur T5/mT5 : configuration, target_modules, scheduler, arrêt prématuré
- QLoRA : quantisation 4-bit NF4 + LoRA — recette bitsandbytes
- Autres PEFT : prefix tuning, prompt tuning, adapters, (IA)³
- **Approche par règles et lexiques DMF/TLFi d'abord** : normalisation Unicode, tables de substitution graphiques (roy→roi, u/v, j/i), priorité traçabilité pour les humanités
- Limites des règles → besoin d'un modèle seq2seq pour les cas ambigus
- Choix des couches à adapter : query/value vs toutes les projections
- Arrêt prématuré, scheduler cosine, warmup — bonnes pratiques
- Démo live : fine-tuning LoRA de CamemBERT sur NER médiévale
	- Démo live : règles + T5 LoRA en pipeline, comparaison des résultats

### Chapitre 4 - Normalisation orthographique du vieux français + abréviations résiduelles + correction guidée
- Formulation comme tâche seq2seq : token brut → forme normalisée
- Fine-tuning LoRA de T5 / mT5 avec la bibliothèque PEFT
- Données : CREMMA + éventuelles règles manuelles pour les abréviations
- Évaluation : accuracy token, CER normalisé, BLEU sur les formes cibles
- Ablation : impact de r=8 vs r=16 vs full fine-tuning sur la validation
- Journal d'expériences : enregistrement systématique dans experiments/journal.jsonl

- **Étape 1 — Règles** : implémenter un module de normalisation par règles (DMF lookup + regex graphiques). Mesurer le CER avant/après sur 200 lignes.
- **Étape 2 — Abréviations résiduelles** : résoudre `q~`→_que_, `pñ`→_prison_ via tables contextuelles (formules latines, titres, monnaies). Annoter les cas ambigus.
- **Étape 3 — Correction guidée par confiance** : pour les caractères confidence < 0.7 avec candidats `[a/b]`, utiliser un modèle de langue (CamemBERT MLM) pour arbitrer. Ablation : confiance seule vs MLM seul vs combiné.
- **Étape 4 — Fine-tuning T5 LoRA** : entraîner sur les paires (brut→normalisé). Comparer r=8 vs r=16. Journal d'expériences.
- Outil t9n comme référence pour les choix de normalisation

### Lien Volet 1
- Les scores de confiance HTR alimentent les poids de perte : lignes incertaines pondérées moins fortement à l'entraînement.
- Les candidats `[a/b]` et les scores par caractère fournis par la CV sont exploités dans l'étape 3. Un pipeline qui ignore ces métadonnées ne remplit pas le cahier des charges.

### Ressources
- DMF — Dictionnaire du Moyen Français (atilf.fr)
- t9n (normalisation médiévale)
- google/mt5-base · peft · editdistance

## NER & annotation morpho-syntaxique sur manuscrits

### Chapitre 5 - Segmentation en mots, NER et POS-tagging pour langues historiques
- **Restauration des frontières de mots** : les espaces manuscrits sont irréguliers. 
	- Approche : tokenisation sur règles + modèle CRF entraîné sur corpus CREMMA. 
	- Évaluation : accuracy sur 100 tokens annotés manuellement.
- Token classification avec BERT : architecture CRF vs softmax en sortie
- POS et lemmatisation : **pie-extended** (modèle medieval-fr) et Stanza "frm" — différence avec "fro" (vieux français)
- Schémas d'annotation : BIO, BIOES — choix pour entités nichées
- Entités cibles : PER, LOC, DATE, ORG, TITLE — spécificités médiévales
- Annotation morpho-syntaxique : POS, lemmatisation, dépendances (Universal Dependencies)
- Outils spécialisés : Stanza, Trankit, pie-extended pour le vieux français
- Mesure de l'IAA (Inter-Annotator Agreement) : Cohen's kappa sur NER
	- - IAA sur 100 lignes annotées croisées : kappa de Cohen sur NER et POS
- Propagation d'erreurs HTR → NER : impact du CER sur le F1-NER
	- simulation CER 0/5/10/12% → impact sur F1-NER

### Chapitre 6 - Pipeline NER bout en bout sur le corpus médiéval
- Fine-tuning CamemBERT-NER ou BERT-medieval sur annotations existantes
- Stratégie de données : utiliser les transcriptions normalisées (sortie Jour 2)
- Annotation POS/lemme avec pie-extended medieval-fr, export CoNLL-U, IAA comparée au modèle
- Évaluation : seqeval F1 micro/macro par type d'entité, rapport de classification
- Analyse des erreurs : confusion PER/LOC, entités multi-tokens, abbréviations, confusion TITLE/PER, lieux non répertoriés, abréviations résiduelles manquées
- Annotation morpho-syntaxique avec Stanza (modèle fro/frm)
- Export au format CoNLL-2003 et Intégration dans le data contract NLP : champs ner_spans, pos_tags, lemmas, polygon_ref

### Ressources
- pie-extended (GitHub Ponteineptique)
- Stanza "frm" · seqeval
- Universal Dependencies — Middle French

### Point éthique +1 bonus

- Analyser les biais de représentation NER : quels types de personnes, lieux et dates sont surreprésentés ? Quelles conséquences sur la portée du modèle ?

## JourModélisation thématique, extraction de relations & base de connaissances

### Chapitre 7 - Du texte à la connaissance structurée

- Modélisation thématique : LDA, NMF, BERTopic — comparaison sur corpus court
- BERTopic sur corpus de pages agrégées : embeddings SBERT + UMAP + HDBSCAN + c-TF-IDF
- Extraction de relations : RE classique vs prompting LLM avec schema
- Résolution de coréférence : neuralcoref, SpanBERT, approches by LLM
- Construction de graphes de connaissances : triplets (sujet, relation, objet), triplets, JSON-LD, SPARQL
- **Export TEI** : structuration des annotations NLP en TEI-XML (balises \<persName>, \<placeName>, \<date>, \<w> pour POS). Standard attendu en humanités numériques.
- **Boucle de rétroaction NLP → HTR** : les corrections produites par le module NLP (abréviations résolues, graphies normalisées) constituent une nouvelle vérité terrain pour réentraîner le modèle HTR. Protocole de réinjection, risques de biais.
- Stockage : RDF/Turtle, Neo4j, ou JSON-LD — choix selon cas d'usage
- Interrogation : SPARQL, Cypher, RAG (Retrieval-Augmented Generation)

### Chapitre 8 - Base de connaissances médiévale interrogeable  + TEI + data contract final

- BERTopic sur le corpus - agrégé, interprétation des topics (liturgique, juridique, narratif)
- Extraction de relations simples (PER — lieu — DATE) - par prompting, validation manuelle, précision calculée
- Export TEI : chaque page → fichier .xml avec annotations NER/POS imbriquées
- Résolution de coréférence sur les entités nommées détectées (Jour 3)
- Construction d'un graphe de connaissances (NetworkX + export JSON-LD)
- Interface de requête minimale : recherche plein-texte + filtre par entité/topic
- Documentation du schéma de données NLP (data contract Volet 2)
- Data contract NLP complet : JSON enrichi + TEI + polygones Volet 1 (ancrage spatial des entités)
- Protocole de rétroaction : générer un delta de corrections pour le modèle HTR (fichier diff JSON)

### Ressources
- TEI Guidelines (tei-c.org)
- lxml (génération XML)
- BERTopic · networkx · json-ld.org

### Lien Volet 1 pipeline

- Les polygones PAGE XML du Volet 1 permettent d'ancrer chaque entité nommée à sa localisation physique sur le manuscrit — conserver cet ancrage dans le graphe.

## Déploiement, traduction historique (bonus), optimisation & restitution

### Chapitre 9 - NLP en production : du modèle au service
- Quantisation : INT8, INT4, GPTQ, AWQ — trade-off vitesse/qualité
- Distillation de modèles : teacher-student, DistilBERT, TinyBERT
- Serving : vLLM, Text Generation Inference (TGI), ONNX Runtime
- Batching dynamique, KV-cache, continuous batching — impact throughput
- Évaluation production : latence p50/p99, throughput req/s, mémoire GPU
- Observabilité : logging des inférences, détection de drift, monitoring
- FastAPI endpoint `/analyze`, benchmark latence p50/p99
- Spécificité humanités numériques : cas d'usage faible trafic vs API patrimoniale
	- Spécificité humanités : traçabilité des versions de modèle > throughput

### Chapitre 10 - Optimisation et packaging du pipeline NLP
- Quantisation INT8 du modèle NER avec bitsandbytes — mesure du F1 post-quant
- Benchmark latence avant/après quantisation (100 requêtes, percentiles)
- Quantisation INT8 du modèle NER, benchmark, Dockerfile, pytest complet
- Packaging : FastAPI + Docker, endpoint /transcribe et /analyze
- Tests pytest : schéma JSON sortie, non-régression F1, temps de réponse
- Pour les équipes bonus : TP OpenNMT-py ou HuggingFace Seq2SeqTrainer sur corpus parallèle fourni (500 paires) — BLEU > 10 pour valider le bonus
- Finalisation README, model cards, DATA_SOURCES, CONVENTIONS_NLP.md  DATA_SOURCES.md, export HuggingFace

## Chapitre 11 - Traduction automatique historique bonus
- Corpus parallèles moyen français / français moderne : éditions critiques bilingues, données liturgiques et juridiques
- Approche règles + lexiques (DMF→TLFi) : pédagogique mais limitée sur les structures complexes
- NMT : fine-tuning d'Opus-MT ou mBART-50 sur corpus parallèle. Pipeline : normalisation → traduction → post-édition
- Évaluation : BLEU, chrF sur corpus de test aligné. Évaluation humaine sur un échantillon (préservation du sens, hallucinations)
- Approche zero-shot LLM (démonstration uniquement — non recommandée pour production)

### Livrable final
- Pipeline NLP dockerisé + tests pytest
- Article scientifique (8–12 pages, format ArXiv)
- Dépôt GitHub public documenté
- Export TEI-XML + data contract NLP
- Pipeline NLP dockerisé + tests pytest
- [Bonus] Script NMT + résultats BLEU
- Soumission Hetic Learn : [groupe]_nlp_manuscrits_MD5.zip

### Cible Performance NLP
- CER < 10% -> après normalisation (seuil de validation)
- CER < 5% -> après correction contextuelle (seuil bonus)
- F1 NER -> 0.65 micro sur PER, LOC, DATE
- BLEU > 10 -> traduction moyen fr. → fr. moderne (bonus)

## Ressources recommandées

| Modèles             | almanach/camembert-base, microsoft/deberta-v3-base, google/mt5-base, sentence-transformers/paraphrase-multilingual-mpnet-base-v2, HuggingFace BERT-medieval |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Données**         | CREMMA Medieval (GitHub), CATMuS/medieval (HuggingFace), corpus NER Ancien français (Zenodo), Universal Dependencies Old French                             |
| **Bibliothèques**   | transformers, peft, trl, bitsandbytes, BERTopic, stanza, pie-extended, spacy-transformers, fastapi, jsonschema, pytest<br>                                  |
| **Références clés** | Hu et al. 2022 (LoRA), Dettmers et al. 2023 (QLoRA), Grootendorst 2022 (BERTopic), Clerice 2023 (pie-extended pour vieux français)<br>                      |
| **Lien Volet 1**    | JSON data contract HTR · PAGE XML avec polygones · hash SHA-256 du test set · scores de confiance par ligne                                                 |
