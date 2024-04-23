# Financial Arrangement Termination Prediction

This project aims to predict the likelihood of termination for financial arrangements using machine learning models. The dataset contains detailed records of financial arrangements, and the target variable is the 'Terminated' feature, a binary variable indicating whether an arrangement was terminated.

## Project Structure

- `data/`: Directory containing the dataset (dataset_v2.csv)
- `src/`: Directory containing the source code
 - `data_preprocessing.py`: Script for data preprocessing
 - `bayesian_opt.py`: Script for hyperparameter tuning using Bayesian optimization
 - `visualisation.py`: Script for visualizing model metrics and feature importances
- `config/`: Directory containing the configuration files for each model
- `EDA_Notebook.ipynb`: Jupyter notebook for exploratory data analysis
- `requirements.txt`: File listing the required Python packages

## Setup

1. Install the required packages:

```python
pip install -r requirements.txt
```
2. Place `dataset_v2.csv` within the data/ directory. 

3. Run the scripts in the following order:

```python
python src/data_preprocessing.py
python src/bayesian_opt.py
python src/visualisation.py
```

## Data Preprocessing

The `data_preprocessing.py` script performs the following steps:
- Drops features with ethical/GDPR implications
- Creates binary variables for missed payments and never paid
- Fills missing values in certain features
- Removes an outlier in the 'NoMonth_FirstMissedPayment' feature
- Splits the data into feature and target variables

## Hyperparameter Tuning

The `bayesian_opt.py` script performs hyperparameter tuning for three models:
- Decision Tree
- Random Forest
- CatBoost

Bayesian optimization is used to find the best hyperparameters for each model. The tuned hyperparameters are saved in the `config/` directory.

## Visualization

The `visualisation.py` script visualizes the model metrics and feature importances. It loads the tuned hyperparameters, trains the models, and compares their performance using a bar plot. It also prints the feature importances for each model.

## Exploratory Data Analysis

The `EDA_Notebook.ipynb` notebook contains exploratory data analysis, including:
- Checking for missing values and duplicates
- Analyzing the target variable distribution
- Examining correlations between features
- Visualizing the distribution of categorical and numerical features
- Comparing feature distributions based on the target variable

Please refer to the notebook for detailed insights and observations.