import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
import joblib

def load_data(path):
    """
    Load the dataset from the given path.
    """
    return pd.read_csv(path)

def split_data(df, test_size=0.2, random_state=42):
    """
    Split the data into train and test sets.
    """
    X = df.drop("Survived", axis=1)
    y = df["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, n_estimators=100, random_state=42):
    """
    Train a RandomForestClassifier model.
    """
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    clf.fit(X_train, y_train)
    # Define the parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Initialize the GridSearchCV object
    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', random_state=random_state)

    # Fit the GridSearchCV object to the data
    grid_search.fit(X_train, y_train)

    # Print the best parameters
    print(grid_search.best_params_)

    # Train a new model with the best parameters
    clf_best = RandomForestClassifier(**grid_search.best_params_)
    clf_best.fit(X_train, y_train)
    return clf_best

def evaluate_model(clf, X_test, y_test):
    """
    Evaluate the model and print the results.
    """
    y_pred = clf.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print('\nConfusion Matrix:')
    print(confusion_matrix(y_test, y_pred))
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred))

def save_model(clf, path):
    """
    Save the trained model to a pickle file.
    """
    joblib.dump(clf, path)

def main():
    # Load the data
    df = load_data('../../data/processed/train_processed.csv')

    # Split the data
    X_train, X_test, y_train, y_test = split_data(df)

    # Train the model
    clf = train_model(X_train, y_train)

    # Evaluate the model
    evaluate_model(clf, X_test, y_test)

    # Save the model
    save_model(clf, '../../models/random_forest.pkl')

if __name__ == '__main__':
    main()
