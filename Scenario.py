import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from joblib import dump, load

def load_instagram_data():
    """Loads the Instagram data from a CSV file.

    Returns:
        A Pandas DataFrame containing the Instagram data.
    """

    # Load the Instagram data from the CSV file.
    data = pd.read_csv('instagram_data.csv')

    # Return the data.
    return data

def split_instagram_data(data):
    """Splits the Instagram data into training and testing sets.

    Args:
        data: A Pandas DataFrame containing the Instagram data.

    Returns:
        Four Pandas DataFrames containing the training and testing data and labels.
    """

    # Split the data into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(data.drop('did_vote', axis=1), data['did_vote'], test_size=0.25, random_state=42)

    # Return the training and testing data and labels.
    return X_train, X_test, y_train, y_test

def train_logistic_regression_model(X_train, y_train):
    """Trains a logistic regression model on the Instagram data.

    Args:
        X_train: A Pandas DataFrame containing the training data.
        y_train: A Pandas Series containing the training labels.

    Returns:
        A trained logistic regression model.
    """

    # Create a logistic regression model.
    model = LogisticRegression()

    # Fit the model to the training data.
    model.fit(X_train, y_train)

    # Return the trained model.
    return model

def evaluate_logistic_regression_model(model, X_test, y_test):
    """Evaluates the performance of a logistic regression model on the testing data.

    Args:
        model: A trained logistic regression model.
        X_test: A Pandas DataFrame containing the testing data.
        y_test: A Pandas Series containing the testing labels.

    Returns:
        The accuracy of the model on the testing data.
    """

    # Make predictions on the testing data.
    y_pred = model.predict(X_test)

    # Calculate the accuracy of the model.
    accuracy = accuracy_score(y_test, y_pred)

    # Return the accuracy of the model.
    return accuracy

def save_model(model, filename):
    """Saves a model to a file.

    Args:
        model: A trained model.
        filename: The path to the file to save the model to.
    """

    # Save the model to the file.
    dump(model, filename)

def main():
    """Loads the Instagram data, trains a logistic regression model, evaluates the performance of the model, saves the model, and uploads the code to GitHub."""

    # Load the Instagram data.
    data = load_instagram_data()

    # Split the data into training and testing sets.
    X_train, X_test, y_train, y_test = split_instagram_data(data)

    # Train a logistic regression model on the training data.
    model = train_logistic_regression_model(X_train, y_train)

    # Evaluate the performance of the model
