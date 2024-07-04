# PLAN D'ANALYSE DES DONNEES DE CETTE ETUDE

Pour réaliser une courbe de survie des patients drépanocytaires en fonction des statuts vaccinaux avec détermination de la p-value, les variables nécessaires dans votre jeu de données incluront :

1. **ID du Patient** : Un identifiant unique pour chaque patient.
2. **Temps de Suivi** : La durée de suivi pour chaque patient (en mois, jours, ou années).
3. **Événement de Survie** : Une variable binaire indiquant si l'événement de survie (décès ou une autre condition d'intérêt) s'est produit (1) ou non (0) pendant la période de suivi.
4. **Statut Vaccinal** : Une variable indiquant le statut vaccinal des patients (vacciné ou non vacciné).
5. **Date de Début du Suivi** : La date à laquelle le suivi du patient a commencé.
6. **Covariables Potentielles** : D'autres variables cliniques ou démographiques qui pourraient influencer la survie (âge, sexe, type de drépanocytose, traitements reçus, comorbidités, etc.)

Avec nos variables, nous allons utiliser des méthodes statistiques telles que l'analyse de survie de Kaplan-Meier pour estimer la courbe de survie, et le test du log-rank pour comparer les courbes de survie entre les groupes de patients vaccinés et non vaccinés, ce qui permettra de déterminer la p-value associée à cette comparaison.

Pour aller plus loin, nous allons envisager d'utiliser un modèle de régression de Cox (Cox proportional hazards model) pour ajuster les effets d'autres covariables sur la survie et obtenir des estimations plus robustes.

## Determination des covariables

Identifier les covariables importantes dans notre jeu de données est crucial pour obtenir des résultats précis et robustes dans vos analyses statistiques. Voici les étapes et méthodes que nous allons utiliser pour déterminer l'importance des covariables :

### 1. **Exploration Préliminaire des Données**

- Cliquez sur `*code*` pour vous rendre vers l'analyse exploratoire des données [code](descriptive_analysis.py#L4-L29)
- Résumé Statistique : Utilisez des fonctions comme describe() de pandas pour obtenir un résumé statistique de vos données.

```python
# Loading cleaned dataset
df = load_cleaned_data()

# Filter key variables
df = df[['CODE', 'AGE', 'SEXE', 'NIVEAUSCOLAIRE', 'STATUTMARITAL', 'ETHNIE', 'AGE_DE_DIAGNOSTIC', 'DECEDES', 'VACCINSAJOUR',
            'AGEAUDECES']]

# Preliminary exploration of dataset

# Summarize the descriptive analysis
print(df.describe())
```

Resultats
![Screenshot of result of a describe() method on the dataframe](result_desc_ana_summ.png)

- Visualisation : Utilisez des graphiques comme les diagrammes de dispersion, les histogrammes, et les boîtes à moustaches pour visualiser les relations entre les variables

#### 1.1 Diagramme de Dispersion (Scatter Plots)

Le diagrammes de dispersion a été utiliser pour visualiser la relation entre deux variables continues qui sont l'age auquel le patient a été diagnostiqué et l'age de décès du patient.

```python
# Draw a scatter plot to visualize the distribution of 2 continuous variables
scatter_plot(df, 'AGE_DE_DIAGNOSTIC', 'AGEAUDECES', "Age at diagnostic vs Age at death", var3="VACCINSAJOUR")
```

Resultats
![Text](result_desc_ana_scatter_plot.png)