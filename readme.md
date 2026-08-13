# **Rapport de Projet — Atlantic Haven Hotels**

## **Examen Final Machine Learning & Data Science — M1**

Réalisé au sein de [**ISPM — Madagascar**](https://www.ispm-edu.com)

---

### **1. Informations sur le Groupe**

#### Membre 1

- nom : ARISOA
- prénom(s) : Aika Fitia
- classe : IGGLIA 4
- numéro : 9
- rôle : *développeur, analyste, responsable de la modélisation*

### **2. Résumé du Travail**

#### Problématique

*Atlantic Haven Hotels subit des annulations tardives qui laissent des chambres inoccupées et perturbent la planification opérationnelle sur ses dix régions italiennes. L'objectif est de prédire, au moment de la réservation ou en amont de l'arrivée, la probabilité qu'une réservation soit annulée (`reservation_annulee`), afin de permettre une intervention commerciale ciblée plutôt qu'une gestion réactive. Une détection précoce et suffisamment précise permet d'arbitrer entre surbooking raisonné, relance client et réallocation de chambres.*

#### Méthodologie adoptée

- **EDA** : cible déséquilibrée (25,8 % d'annulations sur l'ensemble train), valeurs manquantes concentrées sur `marche_origine`, `enfants`, `prix_moyen_nuit_eur`, `demandes_speciales` et `agent_id` (ce dernier étant en réalité un indicateur de réservation directe, recodé en `"DIRECT"` plutôt qu'imputé).
- **Validation temporelle** : tri des données par `date_reservation`, coupure au 80ᵉ centile temporel — split au **2024-11-28**, donnant 6 395 réservations d'entraînement et 1 605 de validation, avec des taux d'annulation proches (25,6 % vs 26,9 %) confirmant la stabilité de la cible dans le temps.
- **Baseline** : régression logistique (`class_weight="balanced"`), prétraitements (imputation médiane/mode, standardisation, one-hot) appris uniquement sur le train via `ColumnTransformer`.
- **Feature engineering** : variables calendaires sur la date d'arrivée, ratio délai de réservation/nuits, prix par personne, taux d'annulation historique du client, et une variable composite `sans_engagement` (aucun acompte + tarif remboursable).
- **Modèles comparés** : régression logistique (baseline et avec FE), Random Forest, HistGradientBoosting — tous évalués sur le même jeu de validation temporelle.
- **Seuil de décision** : optimisé par recherche sur grille (pas de 0,01) pour maximiser le F1 sur la validation, plutôt que le seuil par défaut de 0,5.

#### Résultats obtenus

Meilleur modèle retenu : **Random Forest**, seuil de décision **0,42**, **F1 = 0,4755** sur la validation temporelle (precision 0,345 / recall 0,764 / ROC-AUC 0,649).

Découverte importante : les variables les plus déterminantes ne sont pas les caractéristiques du séjour (nuits, adultes, catégorie hôtel) mais les **conditions commerciales de la réservation** — `type_acompte`, `tarif_remboursable`, et leur combinaison `sans_engagement` dominent l'importance des variables, devant le prix et le délai de réservation.

#### Mots-clés

*Classification binaire, annulation hôtelière, validation temporelle, F1-score, feature engineering, déséquilibre de classes, seuil de décision, importance des variables.*

---

### **3. Contenu du Repository**

Voici la liste des fichiers et liens importants permettant d’évaluer votre travail :

- **reponse.ipynb** : code complet de l’EDA, du prétraitement, de la modélisation et de l’évaluation ;
- **submission.csv** : prédictions sur `reservations_test.csv` ;
- **README.md** : présent rapport ;

**🔗 Liens utiles :**

- [Lien vers le dépôt GitHub](https://github.com/aikafitia/ML-Exam-S2.git)

### **4. Résultats de Modélisation**

Tous les modèles sont évalués sur le même jeu de validation temporel (1 605 réservations, split au 2024-11-28). Les scores "@0,5" utilisent le seuil par défaut ; les scores "optimisé" utilisent le seuil recherché sur grille.

| Modèle | Paramètres principaux | F1-score | Précision | Rappel | ROC-AUC |
|---|---|---:|---:|---:|---:|
| Régression logistique — baseline | `class_weight=balanced`, seuil 0,5 | 0,4533 | 0,3619 | 0,6065 | 0,6466 |
| Régression logistique — baseline, seuil optimisé | seuil = 0,40 | 0,4685 | — | — | 0,6466 |
| Régression logistique + feature engineering, seuil optimisé | seuil = 0,40 | 0,4713 | 0,3360 | 0,7894 | 0,6457 |
| Random Forest + FE, seuil optimisé | `n_estimators=400, max_depth=10`, seuil = 0,42 | **0,4755** | 0,3452 | 0,7639 | 0,6490 |
| HistGradientBoosting + FE, seuil optimisé | `max_iter=300, max_depth=6, lr=0,05`, seuil = 0,23 | 0,4705 | 0,3615 | 0,6736 | 0,6384 |

**Seuil de décision retenu :** 0,42 (Random Forest).

**Justification du choix du modèle final :**

Le Random Forest obtient le meilleur F1 après optimisation du seuil, avec un ROC-AUC légèrement supérieur aux autres modèles. L'écart avec la régression logistique + FE (0,4755 vs 0,4713) reste toutefois modeste : sur ce dataset, la majorité du signal est déjà linéaire et capturée par `type_acompte` et `tarif_remboursable`, ce qui explique pourquoi un modèle simple reste compétitif. Le Random Forest est retenu pour son léger gain de performance et parce qu'il fournit une hiérarchie d'importance des variables directement exploitable pour la recommandation métier, sans sacrifier l'interprétabilité au point de devenir une boîte noire (comparé au gradient boosting, plus difficile à régler ici avec un jeu d'entraînement de cette taille).

**Limite honnête à signaler** : au seuil optimal pour le F1, le modèle prédit une annulation dans **58 %** des cas sur le jeu de test, alors que le taux réel observé en train est d'environ 26 %. Optimiser uniquement le F1 pousse le modèle à privilégier fortement le rappel (peu de faux négatifs) au prix de nombreux faux positifs. Ce compromis doit être assumé explicitement dans la recommandation opérationnelle (section 6) plutôt que présenté comme un résultat sans contrepartie.

---

### **5. Réponses aux Questions d’Analyse**

#### **Q1. Pourquoi utilise-t-on principalement le F1-score plutôt que l'accuracy pour cette tâche ?**

La cible est déséquilibrée (25,8 % d'annulations, 74,2 % de maintenues). Un modèle qui prédirait systématiquement "maintenue" atteindrait déjà 74 % d'accuracy sans détecter la moindre annulation, ce qui est inutile opérationnellement. Le F1-score, moyenne harmonique de la précision et du rappel sur la classe minoritaire ("annulation"), pénalise à la fois les annulations manquées (faux négatifs) et les fausses alertes (faux positifs), ce qui reflète mieux l'enjeu métier réel.

#### **Q2. Dans ce contexte, qu'est-ce qui est le plus grave : un faux positif ou un faux négatif ?**

Un **faux positif** (le modèle prédit une annulation qui n'arrive pas) entraîne une action commerciale inutile — relance, éventuel surbooking préventif — coûteuse en image client si elle est intrusive, mais rarement catastrophique si l'action reste légère (email, appel).

Un **faux négatif** (le modèle ne détecte pas une annulation réelle) prive l'hôtel de la fenêtre d'action : la chambre reste bloquée jusqu'à l'annulation tardive, sans possibilité de réallocation.

Dans un contexte hôtelier avec un delai_reservation généralement de plusieurs semaines, le faux négatif est en général **plus coûteux** car il empêche toute anticipation, alors qu'un faux positif bien géré (action légère et non intrusive) a un coût limité. C'est cette asymétrie qui justifie de privilégier un modèle à fort rappel plutôt qu'à forte précision, quitte à accepter plus de fausses alertes — d'où le choix d'un seuil de décision qui favorise le recall (0,76) au F1 optimal.

#### **Q3. Quelles variables créées par feature engineering ont le plus amélioré votre modèle par rapport à la régression logistique de référence ?**

- `sans_engagement` (aucun acompte ET tarif remboursable) : combinaison des deux variables commerciales les plus discriminantes, apparaît en 4ᵉ position dans l'importance des variables du Random Forest.
- `ratio_delai_nuits` (délai de réservation rapporté à la durée du séjour) : 6ᵉ variable la plus importante.
- `prix_par_personne` : capture un signal différent du prix moyen brut.

Le gain mesuré est **modeste** : F1 optimal de 0,4685 (baseline, seuil optimisé) à 0,4713 (baseline + FE, seuil optimisé), soit +0,6 point. Le feature engineering apporte surtout un gain d'interprétabilité (`sans_engagement` résume une interaction facilement actionnable commercialement) plutôt qu'un gain brut de score, car les variables originales `type_acompte` et `tarif_remboursable` portaient déjà l'essentiel du signal.

#### **Q4. Pourquoi un découpage aléatoire simple peut-il produire une évaluation trompeuse sur ce dataset ?**

Les dates de réservation du train (2023-01-01 à 2025-05-24) et du test (2025-05-24 à 2025-12-31) montrent que le test est strictement postérieur au train. Un découpage aléatoire mélangerait des réservations passées et futures dans le même pli de validation, ce qui permettrait au modèle de "voir" indirectement des tendances commerciales ou saisonnières issues de périodes qu'il ne devrait pas connaître au moment de l'évaluation — donnant un score optimiste non représentatif de la performance réelle sur des données futures. Nous avons donc trié les données par `date_reservation` et coupé au 80ᵉ centile temporel (2024-11-28), pour obtenir 6 395 réservations d'entraînement et 1 605 de validation, avec des taux d'annulation cohérents (25,6 % vs 26,9 %).

#### **Q5. Quels profils ou scénarios de réservation sont les plus fréquemment associés aux annulations dans vos analyses ?**

- Réservations **sans acompte** (34,0 % d'annulation) contre 10,4 % pour un acompte total.
- Réservations à **tarif remboursable** (31,3 % d'annulation) contre 14,0 % pour un tarif non remboursable.
- Réservations faites via une **plateforme en ligne** (30,4 %) contre 14,5 % pour un canal entreprise.
- La combinaison **aucun acompte + tarif remboursable** (`sans_engagement`) cumule ces effets et ressort comme un signal fort dans le modèle final.

*Ces patterns décrivent des conditions commerciales et un canal de distribution, pas une caractéristique intrinsèque à une région ou une population.*

#### **Q6. Comment votre pipeline traite-t-il les valeurs manquantes et les catégories jamais observées pendant l'entraînement ?**

Toutes les transformations sont encapsulées dans un `ColumnTransformer` intégré à un `Pipeline` scikit-learn, **entraîné uniquement sur le jeu d'entraînement** :
- variables numériques : imputation par la médiane du train, puis standardisation ;
- variables catégorielles : imputation par le mode du train, puis `OneHotEncoder(handle_unknown="ignore")`, qui encode toute catégorie inconnue au moment du test comme un vecteur nul plutôt que de lever une erreur ;
- cas particulier : `agent_id` manquant est recodé en `"DIRECT"` avant le pipeline, car l'absence de valeur correspond à une réservation directe (information, pas une donnée manquante).

Aucune statistique (médiane, mode, catégories) n'est calculée sur validation ou test — le `.fit()` du pipeline n'est appelé que sur `X_train`, garantissant l'absence de fuite.

#### **Q7. Selon vous, quelle action l'hôtel devrait-il entreprendre lorsqu'une réservation en cours présente une forte probabilité d'annulation ?**

Une intervention **proportionnée et non intrusive** : contact préventif (email ou SMS) proposant une flexibilité (changement de dates, upsell sur une offre non remboursable avec réduction) plutôt qu'une annulation ou une réattribution automatique de la chambre. Pour les probabilités les plus élevées, un signalement à l'équipe de gestion des réservations pour anticiper un éventuel surbooking raisonné sur cette date, sans jamais pénaliser unilatéralement le client.

#### **Q8. Votre modèle présente-t-il des performances comparables selon les régions ou les types de destination ?**

F1 par région sur la validation (régions avec ≥ 50 observations) :

| Région | N | F1 |
|---|---:|---:|
| Sicilia | 121 | 0,571 |
| Trentino-Alto Adige | 141 | 0,550 |
| Lombardia | 218 | 0,508 |
| Puglia | 99 | 0,494 |
| Lazio | 258 | 0,460 |
| Veneto | 215 | 0,460 |
| Liguria | 127 | 0,453 |
| Sardegna | 68 | 0,456 |
| Toscana | 189 | 0,438 |
| Campania | 169 | 0,380 |

L'écart entre la meilleure région (Sicilia, 0,571) et la moins bonne (Campania, 0,380) est notable (~0,19 point de F1). Avec des effectifs de 68 à 258 observations par région, ces estimations restent bruitées — un intervalle de confiance serait large pour les sous-groupes les plus petits (Sardegna, Puglia). Cette hétérogénéité mérite d'être surveillée en production plutôt que traitée comme un biais structurel avéré à ce stade.

#### **Q9. Analyse des erreurs**

Analysez au minimum :

- cinq faux positifs ;
- cinq faux négatifs ;
- les raisons possibles de ces erreurs ;
- une piste d’amélioration des données ou du modèle.

**5 faux positifs** (prédits annulés, en réalité maintenus) — tous caractérisés par `type_acompte=aucun` et `tarif_remboursable=oui`, ex. réservations R000308, R001943, R009259, R008402, R007351, avec des probabilités prédites entre 0,44 et 0,62. Le modèle suit ici le pattern commercial dominant, mais ces clients ont finalement maintenu leur séjour — les variables commerciales ne suffisent pas à elles seules à capturer l'intention réelle du client.

**5 faux négatifs** (prédits maintenus, en réalité annulés) — ex. R000609 (acompte total, non remboursable, probabilité 0,33), R001806 (acompte partiel, délai de réservation de 140 jours, probabilité 0,40). Ces cas correspondent à des réservations avec des garanties commerciales fortes (acompte total ou partiel, non remboursable) qui, statistiquement, annulent peu — mais l'ont fait ici, échappant au signal principal du modèle.

**Raisons possibles** : le modèle s'appuie fortement sur les conditions commerciales, qui sont de bons prédicteurs en moyenne mais n'expliquent pas les décisions individuelles (imprévus personnels, changements de plans non capturés par les variables disponibles).

**Piste d'amélioration** : enrichir les données avec des signaux comportementaux plus fins (historique de communication, modifications récentes de la réservation dans les jours précédant l'arrivée) qui ne sont pas disponibles dans ce jeu de données synthétique.

---

### **6. Conclusion et Recommandations**

Le modèle Random Forest final atteint un F1 de 0,4755 sur la classe "annulation", avec un rappel élevé (0,76) obtenu au prix d'une précision modérée (0,35). Cette performance reste limitée en absolu (ROC-AUC de 0,649), ce qui reflète la difficulté intrinsèque de prédire un comportement individuel à partir de variables essentiellement commerciales et déclaratives. Le modèle est fiable pour hiérarchiser les réservations par risque relatif, mais ne doit pas être utilisé pour des décisions automatiques et irréversibles.

**Recommandation opérationnelle finale :** utiliser la probabilité produite comme un score de priorisation pour une intervention commerciale légère et graduée (contact préventif, offre de flexibilité) sur les réservations à risque, jamais pour une réattribution automatique de chambre. Étant donné le fort taux de faux positifs au seuil retenu, prévoir une revue humaine avant toute action à coût significatif.

---

### **7. Reproductibilité**

- version de Python : 3.12.7
- principales bibliothèques et versions : : pandas, scikit-learn
- graine(s) aléatoire(s) : `RANDOM_STATE = 42`
- commande ou procédure d’exécution : exécuter `reponse.ipynb` de bout en bout depuis un noyau vierge
- durée approximative d’entraînement : quelques secondes à quelques minutes selon la machine (Random Forest à 400 arbres sur ~6 400 lignes)
- environnement utilisé : *local*

---

### **8. Bibliographie**

- Documentation officielle scikit-learn (`Pipeline`, `ColumnTransformer`, `OneHotEncoder`, `RandomForestClassifier`, `HistGradientBoostingClassifier`).
- Sujet d'examen fourni par l'ISPM (RABOANARY Heriniaina Andry).
- Assistant IA (Claude, Anthropic) utilisé pour la construction du pipeline de prétraitement, la comparaison de modèles, la recherche du seuil de décision optimal et la rédaction assistée de ce rapport à partir des résultats réellement obtenus sur les données du projet.

- Référence 1 : scikit-learn
- Référence 2 : [Claude AI](https://www.claude.ai)
