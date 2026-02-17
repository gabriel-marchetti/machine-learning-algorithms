import numpy as np 
import pandas as pd 
import os

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def preprocess_german(german_path):
    print(f'Starting preprocess_german')
    columns = [
        'CheckingAccount',
        'Duration',
        'CreditHistory',
        'Purpose',
        'CreditAmount',
        'SavingsAccount',
        'EmploymentSince',
        'InstallmentRate',
        'PersonalStatusAndSex',
        'OtherDebtors',
        'ResidenceSince',
        'Property',
        'Age',
        'OtherInstallmentPlans',
        'Housing',
        'ExistingCredits',
        'Job',
        'Dependents',
        'Telephone',
        'ForeignWorker',
        'Target'
    ]

    try:
        print(f'Importing Data from German Dataset...')
        df = pd.read_csv(
            german_path,
            header=None,
            sep=r'\s+',
            engine='python',
            names=columns
        )
        y_raw = df['Target']
        y = y_raw - 1
        X = df.drop(columns=['Target'])

        numerical_cols = X.select_dtypes(include=['number']).columns
        categorical_cols = X.select_dtypes(exclude=['number']).columns

        # Perform One-Hot Encode
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

        # One-Hot Encode return True/False dtype, so we correct with:
        bool_cols = [col for col in X.columns if X[col].dtype == 'bool']
        for col in bool_cols:
            X[col] = X[col].astype(float)

        # Scale Numerical Values
        scaler = StandardScaler()
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

        X['Target'] = y

        german_preprocessed_path = os.path.join('..', 'data', 'preprocessed', 'german.csv')
        output_dir = os.path.dirname(german_preprocessed_path)
        os.makedirs(output_dir, exist_ok=True)

        X.to_csv(german_preprocessed_path, index=False)
    except FileNotFoundError:
        raise ValueError(f'Path not found for German Dataset: {german_path}')