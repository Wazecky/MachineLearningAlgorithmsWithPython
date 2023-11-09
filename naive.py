import numpy as np

class NaiveBayes:
    def __init__(self):
        self.class_probs = None
        self.mean = None
        self.variance = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.class_probs = {
            0: np.sum(y == 0) / n_samples,
            1: np.sum(y == 1) / n_samples
        }
        self.mean = {
            0: np.mean(X[y == 0], axis=0),
            1: np.mean(X[y == 1], axis=0)
        }
        self.variance = {
            0: np.var(X[y == 0], axis=0),
            1: np.var(X[y == 1], axis=0)
        }
    
    def predict_prob(self, X):
        prob_0 = np.prod(1 / np.sqrt(2 * np.pi * self.variance[0]) * np.exp(-(X - self.mean[0]) ** 2 / (2 * self.variance[0])), axis=1)
        prob_1 = np.prod(1 / np.sqrt(2 * np.pi * self.variance[1]) * np.exp(-(X - self.mean[1]) ** 2 / (2 * self.variance[1])), axis=1)
        
        prob_0 *= self.class_probs[0]
        prob_1 *= self.class_probs[1]
        
        return prob_0 / (prob_0 + prob_1), prob_1 / (prob_0 + prob_1)
    
    def predict(self, X, threshold):
        prob_0, prob_1 = self.predict_prob(X)
        return (prob_1 >= threshold).astype(int)

# Let's create some dummy data for demonstration
X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y_train = np.array([0, 0, 1, 1, 1])

# Create an instance of NaiveBayes
model_nb = NaiveBayes()

# Fit the model to the training data
model_nb.fit(X_train, y_train)

# Make a prediction
print("Prediction for [5, 6]:", model_nb.predict(np.array([[5, 6]]), threshold=0.5))
