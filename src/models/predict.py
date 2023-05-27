import pandas as pd
import joblib

def load_data(path):
    """
    Load the dataset from the given path.
    """
    return pd.read_csv(path)

def load_model(path):
    """
    Load the trained model from a pickle file.
    """
    return joblib.load(path)

def make_predictions(clf, data):
    """
    Use the trained model to make predictions on the data.
    """
    return clf.predict(data)

def main():
    # Load the data
    df = load_data('data/processed/test_processed.csv')

    # Load the trained model
    clf = load_model('models/random_forest.pkl')

    # Make predictions
    predictions = make_predictions(clf, df)

    # Create a DataFrame for the predictions
    submission_df = pd.DataFrame({
        "PassengerId": df["PassengerId"],  # assumes that df includes "PassengerId"
        "Survived": predictions
    })

    # Save the predictions to a CSV file
    submission_df.to_csv('data/submission.csv', index=False)

if __name__ == '__main__':
    main()
