import pandas as pd
import flask
from flask import Flask, request, jsonify
import pickle
from sklearn.preprocessing import StandardScaler, RobustScaler

# Initialisation de l'application Flask
app = Flask(__name__)

DATA_LOCATION = 'data_final'
DATA_NAME = 'X_test_sample'
MODEL_LOCATION = 'data_model_explain'
MODEL_NAME = 'model'
EXPLAINER_NAME = 'explainer'

# Fonction permettant de charger un fichier pickle
def load(filename, filepath='./'):
    with open(f'{filepath}/{filename}.pickle', 'rb') as f:
        return pickle.load(f)
    
# Chargement des données des clients, du modèle et de l'explainer SHAP
data = load("DATA_NAME", f'./{"DATA_LOCATION"}')
model = load("MODEL_NAME", f'./{"MODEL_LOCATION"}')
explainer = load("EXPLAINER_NAME", f'./{"MODEL_LOCATION"}')

# Traitement des données pour les préparer à leur utilisation dans les différents contextes de l'application
customer_ids = list(data['SK_ID_CURR'])
data_customers = data[[c for c in data.columns if c not in ['TARGET', 'SK_ID_CURR']]]
scaler = RobustScaler()
scaled_data_customers = pd.DataFrame(scaler.fit_transform(data_customers),
                                     columns=data_customers.columns,
                                     index=data_customers.index)

# Récupération du modèle dans le pipeline sauvegardé
model = pipeline.steps[1][1]

# Classe héritant de Exception qui sera raise en cas de comportement inattendu de l'application
# Cette classe aura comme arguments le status de l'erreur survenue, un message associé et des données le cas échéant
class InvalidUsage(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, data=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.data = data

    def to_dict(self):
        rv = dict(self.data or ())
        rv['erreur'] = self.message
        return rv


# Classe permettant de structurer la réponse d'une requête en format json
class Response:

    def __init__(self, message, data, status_code=200):
        self.json = jsonify({"message": message, "data": data})
        self.json.status_code = status_code


@app.route('/customer/<id_>', methods=['GET'])
def customer_controller(id_):
    """
    Route permettant de récupérer les données concernant les clients.
    Cette route prend pour paramètre "id_" qui peut valoir:
    - soit -1 : dans ce cas la route renvoie la liste des identifiants de tous les clients
    - soit l'identifiant du client : dans ce cas la route renvoie les données du client demandé
    Si l'id donné n'existe pas, alors la requête renverra une erreur 404.
    Si il se produit une erreur inattendue alors la requête renverra une erreur 500 avec le message d'erreur.
    :param id_: str Identifiant du client ou -1
    :return: Response
    """

    try:
        id_ = int(id_)

        if id_ not in customer_ids + [-1]:
            raise InvalidUsage(message="Le client n'a pas été trouvé.", status_code=404)

        if id_ == -1:  # on veut récupérer la liste des clients
            return Response(message="Identifiants de tous les clients récupérés.", status_code=200,
                            data=customer_ids).json
        else:  # on veut récupérer les données d'un client
            customer_data = data_customers[data_customers.index == id_].T.to_dict()
            return Response(message="Données du client récupérées.", status_code=200, data=customer_data).json

    except InvalidUsage as e:
        raise InvalidUsage(message=e.message, status_code=e.status_code)

    except Exception as e:
        raise InvalidUsage(message=f"Quelque chose s'est mal passé: {e}.", status_code=500)


@app.route('/predict/<id_>', methods=['GET'])
def predict_controller(id_):
    """
    Route permettant de prédire si un client va rembourser son prêt.
    Cette route prend pour paramètre "id_" qui correspond à l'identifiant du client.
    Un argument de requête "threshold" doit être obligatoirement ajouté. Il s'agit du seuil de tolérance à appliquer.
    Si l'id donné n'existe pas, alors la requête renverra une erreur 404.
    Si le "threshold" n'est pas présent alors la requête renverra une erreur 400.
    Si il se produit une erreur inattendue alors la requête renverra une erreur 500 avec le message d'erreur.
    :param id_: str Identifiant du client
    :return: Response
    """

    try:
        id_ = int(id_)

        if id_ not in customer_ids:
            raise InvalidUsage(message="Le client n'a pas été trouvé.", status_code=404)

        customer_data = scaled_data_customers[scaled_data_customers.index == id_]
        threshold = request.args.get('threshold')

        if threshold is None:
            raise InvalidUsage(message=f"Le paramètre `threshold` est absent de la requête.", status_code=400)

        threshold = float(threshold)

        y_probability = model.predict_proba(customer_data)[:, 1][0]
        y_predict = int((y_probability > threshold) * 1)

        return Response(message=f"La prédiction a été réalisée avec succès.",
                        status_code=200,
                        data={"id_": id_,
                              "threshold": threshold,
                              "predict": y_predict,  # renvoie la décision d'accorder le prêt ou non
                              "probability": y_probability  # renvoie la probabilité de ne pas rembourser le prêt
                              }).json

    except InvalidUsage as e:
        raise InvalidUsage(message=e.message, status_code=e.status_code)

    except Exception as e:
        raise InvalidUsage(message=f"Quelque chose s'est mal passé: {e}.", status_code=500)


@app.route('/interp/<id_>', methods=['GET'])
def interp_controller(id_):
    """
    Route permettant de réaliser l'interprétabilité globale ou locale pour le modèle choisi.
    Cette route prend pour paramètre "id_" qui peut valoir:
    - soit -1 : dans ce cas la route se charge de l'interprétabilité globale sur un échantillon de clients,
    - soit l'identifiant du client : dans ce cas la route se charge de l'interprétabilité locale du client demandé.
    Un argument de requête "n_customers" doit être obligatoirement ajouté. Il de la taille de l'échantillon sur lequel
    effectuer l'interprétabilité globale.
    Si l'id donné n'existe pas, alors la requête renverra une erreur 404.
    Si le "n_customers" n'est pas présent alors la requête renverra une erreur 400.
    Si le "n_customers" est plus élevé que la taille maximale des données alors la requête renverra une erreur 500.
    Si il se produit une erreur inattendue alors la requête renverra une erreur 500 avec le message d'erreur.
    :param id_: str Identifiant du client ou -1
    :return: Response
    """

    try:
        id_ = int(id_)

        if id_ not in customer_ids + [-1]:
            raise InvalidUsage(message="Le client n'a pas été trouvé.", status_code=404)

        if id_ == -1:  # interprétabilité globale

            n_customers = request.args.get('n_customers')

            if n_customers is None:
                raise InvalidUsage(message=f"Le paramètre `n_customers` est absent de la requête.", status_code=400)

            n_customers = int(n_customers)

            try:
                sampled_data = scaled_data_customers.sample(n=n_customers, random_state=42)
            except ValueError as e:
                raise InvalidUsage(message=f"Quelque chose s'est mal passé: {e}.", status_code=500)

            interp_data = {
                "interp_data": sampled_data.to_dict(),  # renvoie les données pour l'interprétabilité globale
                "shap_values": explainer.shap_values(sampled_data, check_additivity=False).tolist(),
                # renvoie les valeurs de Shapely
                "expected_value": explainer.expected_value  # renvoie la valeur de base
            }

            return Response(message="Données nécessaires à l'interprétabilité globale récupérées.",
                            status_code=200,
                            data=interp_data).json

        else:  # interprétabilité locale

            customer_data = scaled_data_customers[scaled_data_customers.index == id_]

            interp_data = {
                "id_": id_,
                "interp_data": customer_data.to_dict(),  # renvoie les données pour l'interprétabilité locale
                "shap_values": explainer.shap_values(customer_data, check_additivity=False).tolist(),
                # renvoie les valeurs de Shapely
                "expected_value": explainer.expected_value  # renvoie la valeur de base
            }

            return Response(message="Données nécessaires à l'interprétabilité locale récupérées.",
                            status_code=200,
                            data=interp_data).json

    except InvalidUsage as e:
        raise InvalidUsage(message=e.message, status_code=e.status_code)

    except Exception as e:
        raise InvalidUsage(message=f"Quelque chose s'est mal passé: {e}.", status_code=500)


@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    """
    Cette partie permet de catch les erreurs renvoyées à l'intérieur des différents controlleurs pour les convertir en
    réponse adaptée en format json
    :param error:
    :return:
    """
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response