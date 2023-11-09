import numpy as np
from sklearn.tree import DecisionTreeRegressor

class GradientBoostingClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []

    def fit(self, X, y):
        # Initialize the prediction with the mean of y
        initial_prediction = np.mean(y)
        prediction = np.full_like(y, initial_prediction, dtype=float)

        for _ in range(self.n_estimators):
            # Compute the negative gradient (residuals)
            residuals = y - 1 / (1 + np.exp(-prediction))

            # Fit a decision tree to the negative gradient
            tree = DecisionTreeRegressor(max_depth=3)
            tree.fit(X, residuals)

            # Update the prediction with the output of the decision tree
            prediction += self.learning_rate * tree.predict(X)

            # Store the trained tree
            self.models.append(tree)

    def predict_proba(self, X):
        # Initialize predictions with zeros
        predictions = np.zeros(X.shape[0])

        # Aggregate predictions from all trees
        for tree in self.models:
            predictions += self.learning_rate * tree.predict(X)

        # Convert to probabilities using the sigmoid function
        probabilities = 1 / (1 + np.exp(-predictions))

        # Return probabilities for both classes
        return np.column_stack((1 - probabilities, probabilities))

    def predict(self, X, threshold=0.5):
        # Predict probabilities
        probabilities = self.predict_proba(X)[:, 1]

        # Convert probabilities to binary predictions
        return (probabilities >= threshold).astype(int)

# Let's create some dummy data for demonstration
X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y_train = np.array([0, 0, 1, 1, 1])

# Create an instance of GradientBoostingClassifier
model_gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)

# Fit the model to the training data
model_gb.fit(X_train, y_train)

# Make a prediction
print("Prediction for [5, 6]:", model_gb.predict(np.array([[5, 6]]), threshold=0.5))
