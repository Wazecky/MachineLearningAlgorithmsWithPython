from sklearn import tree

class DecisionTree:
    def __init__(self, criterion='gini', splitter='best', max_depth=None):
        self.model = tree.DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth)
    
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)

# Let's create some dummy data for demonstration
X_train = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
y_train = [0, 0, 1, 1, 1]

# Create an instance of DecisionTree
model = DecisionTree()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make a prediction
print("Prediction for [5, 6]:", model.predict([[5, 6]]))
