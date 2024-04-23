# src/visualisation.py

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from catboost import CatBoostClassifier
from data_preprocessing import load_data, preprocess_data
from bayesian_opt import evaluate_model


def load_hyperparameters(file_path):
    with open(file_path, 'r') as f:
        hyperparameters = json.load(f)
    return hyperparameters

def plot_metrics(models, metrics):
    """
    Plots the comparison of model metrics.

    Args:
        models (dict): A dictionary containing the model names as keys and the model objects as values.
        metrics (list): A list of dictionaries where each dictionary contains the metrics for a specific model.

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.2
    x = np.arange(len(metrics[0]))
    metric_names = list(metrics[0].keys())
    colors = plt.cm.rainbow(np.linspace(0, 1, len(models)))

    for i, (model_name, model_metrics) in enumerate(zip(models.keys(), metrics)):
        mean_metric_values = [np.mean(model_metrics[metric_name]) for metric_name in metric_names]
        std_metric_values = [np.std(model_metrics[metric_name]) for metric_name in metric_names]
        ax.bar(x + (i - len(models) / 2 + 0.5) * width, mean_metric_values, width, yerr=std_metric_values,label=model_name, color=colors[i])

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    # Set y_ticks to range 0 - 1, inclusive, in steps of 0.1
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_ylabel('Score')
    ax.set_title('Model Metrics Comparison')
    # Place the legend outside the plot
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    data = load_data("data/dataset_v2.csv")
    X, y, categorical_features = preprocess_data(data)
    

    # Load hyperparameters for each model
    catboost_params = load_hyperparameters("config/catboost_config.json")
    decision_tree_params = load_hyperparameters("config/decision_tree_config.json")
    random_forest_params = load_hyperparameters("config/random_forest_config.json")

    # Instantiate models and unpack hyperparameters
    models = {
        'CatBoost': CatBoostClassifier(**catboost_params),
        'Decision Tree': DecisionTreeClassifier(**decision_tree_params),
        'Random Forest': RandomForestClassifier(**random_forest_params)
    }
    # Create and add ensemble model - all 3 models this time, should check combinations - soft voting, can't use roc_auc with hard
    ensemble = VotingClassifier([(name, model) for name, model in models.items()], voting='soft')
    models['Ensemble'] = ensemble 

    # Evaluate models
    metrics = []
    for model in models.values():
        scores = evaluate_model(model, X, y)
        metrics.append(scores)

    plot_metrics(models, metrics)

# Print feature name - feature importance for all models except ensemble, ordered by importance

    for model_name, model in models.items():
        if model_name != 'Ensemble':
            if model_name == 'CatBoost':
                feature_importances = model.get_feature_importance()
            else:
                feature_importances = model.feature_importances_
            # Indices that would sort the array
            sorted_indices = np.argsort(feature_importances)[::-1]
            sorted_features = [categorical_features[i] if i in categorical_features else X.columns[i] for i in sorted_indices]
            sorted_importances = feature_importances[sorted_indices]
            
            print(f"Model: {model_name}")
            for feature, importance in zip(sorted_features, sorted_importances):
                print(f"Feature: {feature}, Importance: {importance}")
            print()
