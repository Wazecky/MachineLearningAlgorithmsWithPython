import numpy as np
from sklearn import svm

class SVM:
    def __init__(self, kernel='rbf', C=1.0, gamma='scale'):
        self.clf = svm.SVC(kernel=kernel, C=C, gamma=gamma)
    
    def fit(self, X, y):
        self.clf.fit(X, y)
    
    def predict(self, X):
        return self.clf.predict(X)

# Let's create some dummy data for demonstration
X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y_train = np.array([0, 0, 1, 1, 1])

# Create an instance of SVM
model = SVM(kernel='linear', C=1.0, gamma='scale')

# Fit the model to the training data
model.fit(X_train, y_train)

# Make a prediction
print("Prediction for [5, 6]:", model.predict(np.array([[5, 6]])))
