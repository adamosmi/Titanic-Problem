import pandas as pd
import numpy as np

def load_data(path):
    """
    Load the dataset from the given path.
    """
    return pd.read_csv(path)

def fill_missing_values(df):
    """
    Fill missing values in the dataframe.
    """
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df.drop('Cabin', axis=1, inplace=True)
    return df

def transform_features(df):
    """
    Perform feature transformation.
    """
    df['FamilySize'] = df['SibSp'] + df['Parch']
    df = df.drop(['SibSp', 'Parch'], axis=1)
    df['Fare'] = df['Fare'].apply(lambda x: np.log(x) if x > 0 else 0)
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df = df.drop(['Name'], axis=1)
    df['Ticket_Prefix'] = df['Ticket'].apply(lambda x: x.split()[0] if not x.split()[0].isdigit() else 'NoPrefix')
    df = df.drop(['Ticket'], axis=1)
    return df

def encode_categorical_features(df):
    """
    Perform one-hot encoding on categorical features.
    """
    df = pd.get_dummies(df, columns=['Embarked', 'Pclass'])
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df = pd.get_dummies(df, columns=['Title'])
    df = pd.get_dummies(df, columns=['Ticket_Prefix'])
    return df

def save_data(df, path):
    """
    Save the processed data to a csv file.
    """
    df.to_csv(path, index=False)

def main():
    # Load the data
    df = load_data('data/raw/train.csv')

    # Preprocess the data
    df = fill_missing_values(df)
    df = transform_features(df)
    df = encode_categorical_features(df)

    # Save the processed data
    save_data(df, 'data/processed/train_processed.csv')

if __name__ == '__main__':
    main()
