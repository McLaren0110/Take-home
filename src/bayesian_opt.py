# src/hyperparameter_tuning.py

# TODO 1: Clean up integer conversion in hyperparameter tuning/objective functions

import numpy as np
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from data_preprocessing import load_data, preprocess_data

def evaluate_model(model, X, y, cv=5):
    """
    Evaluate the performance of a machine learning model using cross-validation.

    Args:
        model (object): The machine learning model to evaluate.
        X (DataFrame): The input features.
        y (Series): The target variable.
        cv (int, optional): The number of cross-validation folds. Default is 5.

    Returns:
        dict: A dictionary containing the evaluation scores for each fold.
            The dictionary has the following keys: 'f1', 'precision', 'recall', 'accuracy'.
            The values are lists of scores for each fold.
    """
    cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = {
        'f1': [],
        'precision': [],
        'recall': [],
        'accuracy': [],
        'roc_auc': []
    }
    for train_idx, val_idx in cv.split(X, y):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        scores['f1'].append(f1_score(y_val, y_pred))
        scores['precision'].append(precision_score(y_val, y_pred))
        scores['recall'].append(recall_score(y_val, y_pred))
        scores['accuracy'].append(accuracy_score(y_val, y_pred))
        scores['roc_auc'].append(roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]))
    return scores

def objective_decision_tree(params, X, y):
    """
    Objective function for tuning the hyperparameters of a decision tree classifier.

    Args:
        params (dict): A dictionary containing the hyperparameters to be tuned.
        X (DataFrame): The input features.
        y (Series): The target variable.

    Returns:
        dict: A dictionary containing the loss and status of the objective function evaluation.
            The dictionary has the following keys: 'loss', 'status'.
    """
    params['max_depth'] = int(params['max_depth'])
    params['min_samples_split'] = int(params['min_samples_split'])
    params['min_samples_leaf'] = int(params['min_samples_leaf'])
    model = DecisionTreeClassifier(**params)
    scores = evaluate_model(model, X, y)
    best_score = np.mean(scores['recall'])
    return {'loss': -best_score, 'status': STATUS_OK}

def tune_decision_tree(X, y):
    space = {
        'max_depth': hp.quniform('max_depth', 1, 15, 1),
        'min_samples_split': hp.quniform('min_samples_split', 2, 10, 1),
        'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 4, 1),
        'random_state': 42
    }
    trials = Trials()
    # fmin expects a single argument, hence the lambda function - maybe should define outside but works for now
    best = fmin(fn=lambda params: objective_decision_tree(params, X, y),
                space=space,
                algo=tpe.suggest,
                max_evals=200,
                trials=trials)
    print(best)
    best_params = {k: int(best[k]) for k in ['max_depth', 'min_samples_split', 'min_samples_leaf']}
    save_hyperparameters("config/decision_tree_config.json", best_params)
    return DecisionTreeClassifier(**best_params, random_state=42)

def objective_random_forest(params, X, y):
    """
    Objective function for tuning the hyperparameters of a random forest classifier.

    Args:
        params (dict): A dictionary containing the hyperparameters to be tuned.
        X (DataFrame): The input features.
        y (Series): The target variable.

    Returns:
        dict: A dictionary containing the loss and status of the objective function evaluation.
            The dictionary has the following keys: 'loss', 'status'.
    """
    # Convert float values to int for integer parameters
    params['max_depth'] = None if params['max_depth'] < 1 else int(params['max_depth'])
    params['min_samples_split'] = int(params['min_samples_split'])
    params['min_samples_leaf'] = int(params['min_samples_leaf'])
    params['n_estimators'] = int(params['n_estimators'])
    
    model = RandomForestClassifier(**params)
    scores = evaluate_model(model, X, y)
    best_score = np.mean(scores['recall'])
    return {'loss': -best_score, 'status': STATUS_OK}

def tune_random_forest(X, y):
    space = {
        'n_estimators': hp.quniform('n_estimators', 50, 1000, 50),
        'max_depth': hp.quniform('max_depth', 3, 9, 2),
        'min_samples_split': hp.quniform('min_samples_split', 2, 10, 1),
        'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 4, 1),
        'random_state': 42
    }

    trials = Trials()
    best = fmin(fn=lambda params: objective_random_forest(params, X, y),
                space=space,
                algo=tpe.suggest,
                max_evals=200,
                trials=trials)
    print(best)
    best_params = {
        'n_estimators': int(best['n_estimators']),
        'max_depth': None if best['max_depth'] < 1 else int(best['max_depth']),
        'min_samples_split': int(best['min_samples_split']),
        'min_samples_leaf': int(best['min_samples_leaf']),
        'random_state': 42
    }
    save_hyperparameters("config/random_forest_config.json", best_params)

    return RandomForestClassifier(**best_params)

def objective_catboost(params, X, y, categorical_features):
    """
    Objective function for tuning the hyperparameters of a catboost gbm classifier.

    Args:
        params (dict): A dictionary containing the hyperparameters to be tuned.
        X (DataFrame): The input features.
        y (Series): The target variable.
        categorical_features (list): List of categorical features.

    Returns:
        dict: A dictionary containing the loss and status of the objective function evaluation.
            The dictionary has the following keys: 'loss', 'status'.
    """
    # Convert float values to int for integer parameters
    params['iterations'] = int(params['iterations'])
    params['depth'] = int(params['depth'])
    
    model = CatBoostClassifier(**params, cat_features=categorical_features, random_seed=42, verbose=False)
    scores = evaluate_model(model, X, y)
    best_score = np.mean(scores['recall'])
    return {'loss': -best_score, 'status': STATUS_OK}

def tune_catboost(X, y, categorical_features):
    space = {
        'iterations': hp.quniform('iterations', 100, 1000, 100),
        'learning_rate': hp.loguniform('learning_rate', -4.6, -2.3),  # log10(0.01) to log10(0.1)
        'depth': hp.quniform('depth', 3, 11, 2),
        'l2_leaf_reg': hp.uniform('l2_leaf_reg', 1, 10),
    }

    trials = Trials()
    best = fmin(fn=lambda params: objective_catboost(params, X, y, categorical_features),
                space=space,
                algo=tpe.suggest,
                max_evals=200,
                trials=trials)
    print(best)
    best_params = {
        'iterations': int(best['iterations']),
        'learning_rate': best['learning_rate'],
        'depth': int(best['depth']),
        'l2_leaf_reg': best['l2_leaf_reg'],
        'cat_features': categorical_features,
        'random_seed': 42,
        'verbose': False
    }
    save_hyperparameters("config/catboost_config.json", best_params)

    return CatBoostClassifier(**best_params)

def save_hyperparameters(file_path, hyperparameters):
    # Convert int64 values to int - not getting caught out again
    hyperparameters = {k: int(v) if isinstance(v, np.int64) else v for k, v in hyperparameters.items()}
    with open(file_path, 'w') as f:
        json.dump(hyperparameters, f)

if __name__ == "__main__":
    data = load_data("data/dataset_v2.csv")
    X, y, categorical_features = preprocess_data(data)
    # Get best parameters for models and create model objects
    decision_tree = tune_decision_tree(X, y)
    random_forest = tune_random_forest(X, y)
    catboost = tune_catboost(X, y, categorical_features)

    models = {
        'Decision Tree': decision_tree,
        'Random Forest': random_forest,
        'CatBoost': catboost
    }
    # Evaluate models and print metrics
    for model_name, model in models.items():
        scores = evaluate_model(model, X, y)
        print(f"{model_name} - F1: {np.mean(scores['f1']):.3f}, Precision: {np.mean(scores['precision']):.3f}, "
              f"Recall: {np.mean(scores['recall']):.3f}, Accuracy: {np.mean(scores['accuracy']):.3f}, "
              f"ROC AUC: {np.mean(scores['roc_auc']):.3f}")
    

