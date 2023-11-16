import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Load the training data.
data = pd.read_csv('data.csv')

if 'label' not in data.columns:
    # Handle the case where the 'label' column is missing
    print("The label column does not exist in the data.")
    exit()

# Split the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(data.drop('label', axis=1), data['label'], test_size=0.25, random_state=42)


# Train a logistic regression model on the training data.
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the model to a file.
pickle.dump(model, open('model.pkl', 'wb'))

# Load the new data.
new_data = pd.read_csv('new_data.csv')

# Make predictions on the new data.
y_pred = model.predict(new_data)

# Print the predictions.
print(y_pred)
