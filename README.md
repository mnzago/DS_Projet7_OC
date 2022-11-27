### Projet 7: Implentez un modèle de scoring

Je suis Data Scientist au sein d'une société financière, nommée **"Prêt à dépenser"**, qui propose des crédits à la consommation pour des personnes ayant peu ou pas du tout d'historique de prêt.

### Objectifs 

L’entreprise souhaite mettre en œuvre un outil de “scoring crédit” pour calculer la probabilité qu’un client rembourse son crédit, puis classifie la demande en crédit accordé ou refusé. Elle souhaite donc développer un algorithme de classification en s’appuyant sur des sources de données variées (données comportementales, données provenant d'autres institutions financières, etc.).

De plus, les chargés de relation client ont fait remonter le fait que les clients sont de plus en plus demandeurs de transparence vis-à-vis des décisions d’octroi de crédit. Cette demande de transparence des clients va tout à fait dans le sens des valeurs que l’entreprise veut incarner.

Prêt à dépenser décide donc de développer un dashboard interactif pour que les chargés de relation client puissent à la fois expliquer de façon la plus transparente possible les décisions d’octroi de crédit, mais également permettre à leurs clients de disposer de leurs informations personnelles et de les explorer facilement. 

### Missions

  1) Construire un modèle de scoring qui donnera une prédiction sur la probabilité de faillite d'un client de façon automatique.
  2) Construire un dashboard interactif à destination des gestionnaires de la relation client permettant d'interpréter les prédictions faites par le modèle, et d’améliorer la connaissance client des chargés de relation client.
  
### Contenu du répertoire

Ce répertoire contiendra 3 sous-répertoires:
  * data_modelisation : la table nettoyer après preprocessing, le modèle et l'explainer (pour l'interprétation des features importances)
  * flask_api : qui contiendra les fichiers permettant le bon fonctionnement de l'herbergement sur Heroku
  * streamlit_dashboard : contiendra les codes et fichiers pour avoir notre visualiation dashbord web
