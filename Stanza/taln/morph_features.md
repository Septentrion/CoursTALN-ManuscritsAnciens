Le module mockup est conçu pour les cartes et interfaces, pas pour les tableaux de données — d'après les instructions mêmes, ce contenu doit rester en markdown dans ma réponse plutôt que dans un widget. Je présente donc la liste sous forme de tableau markdown.

| Catégorie | Trait UD | Valeurs typiques | Exemple |
|---|---|---|---|
| Nominal | `Gender` | Masc, Fem, Neut | `Gender=Fem` |
| Nominal | `Number` | Sing, Plur, Dual | `Number=Plur` |
| Nominal | `Case` | Nom, Acc, Gen, Dat, Voc... | `Case=Gen` |
| Nominal | `Definite` | Def, Ind | `Definite=Def` |
| Nominal | `Degree` | Pos, Cmp, Sup | `Degree=Cmp` |
| Verbal | `VerbForm` | Fin, Inf, Part, Ger | `VerbForm=Part` |
| Verbal | `Mood` | Ind, Sub, Imp, Cnd | `Mood=Sub` |
| Verbal | `Tense` | Pres, Past, Fut, Imp | `Tense=Past` |
| Verbal | `Aspect` | Perf, Imp, Hab | `Aspect=Perf` |
| Verbal | `Voice` | Act, Pass, Mid | `Voice=Pass` |
| Verbal | `Evident` | Fh, Nfh | `Evident=Fh` |
| Verbal | `Polarity` | Pos, Neg | `Polarity=Neg` |
| Verbal | `Person` | 1, 2, 3 | `Person=3` |
| Pronominal | `PronType` | Prs, Dem, Int, Rel, Ind | `PronType=Rel` |
| Pronominal | `Poss` | Yes | `Poss=Yes` |
| Pronominal | `Reflex` | Yes | `Reflex=Yes` |
| Pronominal | `Foreign` | Yes | `Foreign=Yes` |
| Structurel | `NumType` | Card, Ord | `NumType=Ord` |
| Structurel | `Animacy` | Anim, Inan | `Animacy=Anim` |
| Structurel | `Clusivity` | In, Ex | `Clusivity=In` |

Chaque ligne du champ `feats` d'un objet `Word` Stanza combine plusieurs de ces traits, séparés par une barre verticale — par exemple `Gender=Masc|Number=Sing|Case=Nom`. Le treebank Profiterole (modèle `fro`) n'utilise vraisemblablement qu'un sous-ensemble de cette liste, concentré sur les traits nominaux et verbaux pertinents pour la morphologie de l'ancien français ; pour le vérifier précisément sur votre corpus, le plus fiable reste d'inspecter `word.feats` directement sur les sorties du pipeline.
