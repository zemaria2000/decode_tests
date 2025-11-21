"""
    It just defines a set of useful functions that will be leveraged by all the remaining test scripts
"""

import os
import yaml
from sklearn.metrics import matthews_corrcoef, r2_score

from pre_process_reg import *
from pre_process_clf import *

# Importing all models
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.base import clone

# XAI related libraries
from lime.lime_tabular import LimeTabularExplainer
import alibi
import shap


# Defining a set of default models to be used
DEFAULT_MODELS = {
    # Classifiers
    'logreg': LogisticRegression(max_iter = 100, C = 1.5, solver = 'lbfgs'),
    'svm': SVC(C = 1, degree = 3, probability = True),
    'knn': KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2),
    'mlp': MLPClassifier(hidden_layer_sizes = (16, )),
    # Regresors
    'linreg': LinearRegression(),
    'svr': SVR(degree = 3),
    'knn_r': KNeighborsRegressor(n_neighbors = 5, metric = 'minkowski', p = 2),
    'mlp_r': MLPRegressor(hidden_layer_sizes = (16, )) 
}



def choose_model(model_name: str, seed: int = 17):
    """
        Choosing the appropriate model
        Args:
            - model_name (str): must be compliant with one of the models in DEFAULT_MODELS.keys()
            - seed (int): just for reproducibility
        Returns:
            - model: model instance to then be fit, etc
    """

    if model_name not in DEFAULT_MODELS.keys():
        raise KeyError(f"Model name is incompatible with the models currently defined to be used. Models list of names: {list(DEFAULT_MODELS.keys())}")
    model = clone(DEFAULT_MODELS[model_name])
    try:
        model.set_params(random_state = seed)
    except:
        pass

    return model


def pre_process(dataset_name, seed: int = 17):
    """
        Pre-processing all currently available datasets for testing
        Args:
            - dataset_name (str): the name of the dataset to be processed
    """

    if dataset_name not in ['iris', 'hawks', 'penguins', 'mushrooms', 'diamonds', 'housing']:
        raise KeyError(f"The dataset name is not compatible with the currently available datasets. Dataset names: ['iris', 'hawks', 'penguins', 'mushrooms', 'diamonds', 'housing']")

    # Classification datasets
    if dataset_name == 'iris':
        trainX, testX, trainY, testY, trainX_scaled, testX_scaled, scaler, feature_names, target_names, feature_types = pre_process_iris(seed = seed)
        return trainX, testX, trainY, testY, trainX_scaled, testX_scaled, scaler, feature_names, target_names, feature_types
    if dataset_name == 'hawks':
        trainX, testX, trainY, testY, trainX_scaled, testX_scaled, scaler, feature_names, target_names, feature_types = pre_process_hawks(seed = seed)
        return trainX, testX, trainY, testY, trainX_scaled, testX_scaled, scaler, feature_names, target_names, feature_types
    elif dataset_name == 'penguins':
        trainX, testX, trainY, testY, trainX_scaled, testX_scaled, scaler, feature_names, target_names, feature_types = pre_process_penguins(seed = seed)
        return trainX, testX, trainY, testY, trainX_scaled, testX_scaled, scaler, feature_names, target_names, feature_types        
    elif dataset_name == 'mushrooms':
        trainX, testX, trainY, testY, trainX_scaled, testX_scaled, scaler, feature_names, target_names, feature_types = pre_process_mushrooms(seed = seed)
        return trainX, testX, trainY, testY, trainX_scaled, testX_scaled, scaler, feature_names, target_names, feature_types   
    # Regression datasets
    elif dataset_name == 'diamonds':
        trainX, testX, trainY, testY, trainX_scaled, testX_scaled, trainY_scaled, testY_scaled, scalerX, scalerY, feature_names, target_names, feature_types = pre_process_diamonds(seed = seed)
        return trainX, testX, trainY, testY, trainX_scaled, testX_scaled, trainY_scaled, testY_scaled, scalerX, scalerY, feature_names, target_names, feature_types
    elif dataset_name == 'housing':
        trainX, testX, trainY, testY, trainX_scaled, testX_scaled, trainY_scaled, testY_scaled, scalerX, scalerY, feature_names, target_names, feature_types = pre_process_housing(seed = seed)
        return trainX, testX, trainY, testY, trainX_scaled, testX_scaled, trainY_scaled, testY_scaled, scalerX, scalerY, feature_names, target_names, feature_types


def retrieve_optimised_hyperparams(dataset_name: str, model_name: str, model_type: str):
    """
        Retrieving the optimisation parameters for a specific optimisation session
        Args:
            - dataset_name (str): target dataset
            - model_name (str): target classifier/regressor
            - model_type (str): model type, 'classification' or 'regression'
    """

    # Check dataset + model names
    if model_name not in DEFAULT_MODELS.keys():
        raise KeyError(f"Model name is incompatible with the models currently defined to be used. Models list of names: {list(DEFAULT_MODELS.keys())}")
    if dataset_name not in ['iris', 'hawks', 'penguins', 'mushrooms', 'diamonds', 'housing']:
        raise KeyError(f"The dataset name is not compatible with the currently available datasets. Dataset names: ['iris', 'hawks', 'penguins', 'mushrooms', 'diamonds', 'housing']")
    
    # Check current optimisation folders + load hyperparams
    current_optimisations = os.listdir('./optimisations/')
    opt_name = f'{dataset_name}_{model_name}'
    if opt_name not in current_optimisations:
        print(f"Optimisation for dataset {dataset_name} with model {model_name} has not yet been conducted/finished. As such, resetting to the default hyperparameters..")
        # Retrieving the default hyperparameters
        with open('default_hyperparams.yaml', 'r') as f:
            return yaml.safe_load(f)[model_type]
    else:
        hyperparams_file_path = os.path.join('./optimisations', f'{dataset_name}_{model_name}', 'hyperparams.yaml')
        with open(hyperparams_file_path, 'r') as f:
            opt_hyperparams = yaml.safe_load(f)
        return opt_hyperparams['decode_params']


###############################################
# EXPLAINABILITY FUNCTIONS FOR TIME-ASSESSMENTS
###############################################

## LIME - LIME is a local methodology. However, I have created for global feature importance a loop that iterates all samples, and then averages the feature importances from each...
def lime_values(input_data_sample, model, explainer, model_type: str):
    """
        Extracting a normalised + ordered feature importance ranking for a single sample
        Args:
            - input_data_sample (np.ndarray): The input data sample for which the LIME analysis will be conducted
            - model: model instance to conduct predictions
            - explainer: previously started explainer object, so that I don't have to always instantiate it 
            - model_type: 'clf' or 'reg'
        Returns:
            - ordered_lime_values (np.ndarray): The importance values for each feature, ordered according to their index
    """

    # Apply the explainer to a specific instance
    explanation = explainer.explain_instance(input_data_sample, predict_fn = model.predict_proba if model_type == 'clf' else model.predict, num_features = len(input_data_sample))
    # lime_class_probs = explanation.predict_proba
    explanation = np.array(explanation.local_exp[1])

    # Extract the explanation values + the indices of the features (both ordered according to the importances) 
    lime_values = explanation[:, 1].astype(float)
    lime_indices = explanation[:, 0].astype(int)
    
    # Ordering lime values according to the indices
    ordered_lime_values = np.zeros(len(input_data_sample))
    for index, value in zip (lime_indices, lime_values):
        ordered_lime_values[index] = value

    return ordered_lime_values


def lime_global(samples, model, explainer, model_type):
    """
        Custom loop that conducts LIME local explanations to all samples in an input dataset, averaging those to give a "global" feature ranking
        Args:
            - inputs: the input dataset to compute importances
            - model: model instance to conduct predictions
            - explainer: previously started explainer object, so that I don't have to always instantiate it 
            - model_type: 'clf' or 'reg'
        Returns:
            - norm_lime_importances: normalised averaged global importances for each feature in a dataset
    """
    lime_local_importances = []
    for sample in samples:
        lime_local_importances.append(lime_values(input_data_sample = sample, model = model, explainer = explainer, model_type = model_type))

    # Computing a global set of importances
    lime_local_importances = np.array(lime_local_importances)
    lime_importances = np.sum(np.abs(lime_local_importances), axis = 0)/lime_local_importances.shape[0]
    norm_lime_importances = lime_importances/np.sum(lime_importances)

    return norm_lime_importances


## SHAP - it can be either applied globally or locally

def global_shap_values(samples, explainer):

    """
        Computing global SHAP values for a specific dataset (inadvertently, it creates also individual local values for each specific sample)
        Args:
            samples (np.ndarray): The input data samples
            explainer: shap explainer instance
        Returns:
            global_shap_values (np.ndarray): The global SHAP values for the input data
            local_shap_values (np.ndarray): The local SHAP values for the input data
    """

    # Get the shap_values
    shap_values = explainer(samples)
    shap_values = np.squeeze(shap_values.values)
    # NOTE: the shape of the shap_values will essentially be (N_samples(input_data), N_features(input_data), N_classes output by model)

    # Get the average shap_values per class + global shap values for the dataset
    global_shap_values = np.sum(np.abs(shap_values), axis = 0)/shap_values.shape[0]
    local_shap_values = shap_values

    return global_shap_values, local_shap_values


# NOTE: I can just do exectimeit of the explainer function directly!!
def local_shap_values(sample, explainer):
    """
        With a previsouly instantiated SHAP instance, it computes local SHAP values for a single data sample
        Args:
            - sample: individual sample to evaluate
            - explainer: shap explainer instance
        Returns:
            - shap_values: individual shap values
    """
    # Get the shap_values
    shap_values = explainer(sample.reshape(1, -1)).values[0]

    return shap_values


# Applying Permutation Feature Importance
def permutation_feature_importance(input_X, input_Y, explainer):
    """
        Function which conducts PFI global analysis
        Args:
            input_X (np.ndarray): The input data ssamples - the entire dataset, as this is a global method
            input_Y (np.ndarray): The target labels for the dataset
            explainer: explainer instance 
        Returns:
            global_importances (np.ndarray): The feature importances for each feature
    """

    # Conduct explanations
    explanation = explainer.explain(input_X, input_Y)

    # Get the global average feature importances 
    global_importances = np.zeros(input_X.shape[1])
    for i, item in enumerate(explanation.feature_importance[0]):
        global_importances[i] = item['mean']
    
    return global_importances


# Auxiliary scoring function for the PFI method (for classification!!)
def mcc_scoring_function(y_true, y_pred):
    """
    Scoring function for MCC (Matthews Correlation Coefficient).
    
    Parameters:
        - y_true (array-like): Ground truth labels.
        - y_pred (array-like): Predicted labels.
    
    Returns:
        - float: MCC score.
    """
    return matthews_corrcoef(y_true, y_pred)

# Auxiliary scoring function for the PFI method (for regression!!)
def r2_scoring_function(y_true, y_pred):
    """
    Scoring function for MCC (Matthews Correlation Coefficient).
    
    Parameters:
        - y_true (array-like): Ground truth labels.
        - y_pred (array-like): Predicted labels.
    
    Returns:
        - float: MCC score.
    """
    return r2_score(y_true, y_pred)

##################################################
# EXPLAINABILITY FUNCTION FOR IMPORTANCE ANALYSIS
##################################################

def get_ranks(importances):
    """
        Function that accepts as inputs the rankings provided by a specific method, and ranks those according to their importance to the ranking approach
        Args:
            - importances (np.ndarray): array with all normalised importances for a particular set of features
    """

    arr = np.array(importances)
    
    # Base ranks (descending order)
    ranks = arr.argsort()[::-1].argsort() + 1
    
    # Identify zero-importance features
    zero_mask = (arr == 0)
    
    if np.any(zero_mask):
        # Assign the same rank (max rank + 1) to all zeros
        max_nonzero_rank = ranks[~zero_mask].max() if np.any(~zero_mask) else 0
        ranks[zero_mask] = max_nonzero_rank + 1
    
    return ranks