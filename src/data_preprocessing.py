# src/data_preprocessing.py

import pandas as pd

def load_data(path):
    try:
        data = pd.read_csv(path)
    except FileNotFoundError:
        print("File not found - check project structure and file path.")
        return None
    return data

def preprocess_data(data):
    """
    Preprocesses the data. Features with ethical / GDPR implications are dropped. Null values in 
    NoMonth_FirstMissedPayment and NoMonth_FirstPayment are replaced by -1, with missed_payment and never_paid
    created as binary variables. Splits data into feature and target variables. 

    Args:
        data (pd.DataFrame): Data to be preprocessed.
    
    Returns:
        X (pd.DataFrame): Feature variables.
        y (pd.DataFrame): Target variable.
        categorical_variables: (list): List of categorical variables.

    """
    columns_to_drop = ['Partner_Gender', 
           'Partner_Employment_Status',
           'Arrears_Category',
           'Output_Area_Classification_Code',
           'Lower_Super_Output_Area_Code',
           'AB',
           'C1',
           'C2',
           'DE', 
           'DOB_Year', 
           'DOB_Month',
           'Gender',
           'Partner_Gender',
           'Partner_Employment_Status',
           'under_18',
           'Marital_Status',
           'Physical_Disability_Vulnerability',
           'Illness_Vulnerability',
           'Addiction_Vulnerability',
           'Mental_Health_Vulnerability',
           'age',
           'age_partner',
           'DOB_Year',
           'DOB_Month',
           'household_DI']
    
    # Drop columns with ethical / GDPR implications
    data = data.drop(columns=columns_to_drop)
    
    # Create binary variables for missed_payment and never_paid
    data['missed_payment'] = data['NoMonth_FirstMissedPayment'].notna().astype(int)
    data['never_paid'] = data['NoMonths_FirstPayment'].isna().astype(int)

    # Create ratio variables for household_income to household_expenses
    data['income_expenses_ratio'] = data['household_income'] / data['household_expenses']

    # TotalPaymentsDue to Total_Expected_Duration
    data['TotalPaymentsDue_to_Total_Expected_Duration'] = data['TotalPaymentsDue'] / data['Total_Expected_Duration']    

    # Fill NaN values
    data[['NoMonth_FirstMissedPayment', 'NoMonths_FirstPayment', 'arrears_months']] = data[
        ['NoMonth_FirstMissedPayment', 'NoMonths_FirstPayment', 'arrears_months']
    ].fillna(-1)

    # Remove outlier in NoMonth_FirstMissedPayment
    data = data[data['NoMonth_FirstMissedPayment'] >= -1]

    # Split data into feature and target variables
    X = data.drop('Terminated', axis=1)
    y = data['Terminated']

    # List of categorical variables
    categorical_features = ['Employment_Status', 
                        'home_owner_flag', 
                        'agreed_missed_flag',
                        'missed_payment',
                        'never_paid',]
    
    return X, y, categorical_features

if __name__ == "__main__":
    data = load_data("data/dataset_v2.csv")
    # Check everything works as expected
    if data is not None: 
        X, y, categorical_features = preprocess_data(data)
        print(len(X))
        assert X.shape[0] == y.shape[0]
        assert X.isnull().sum().sum() == 0




