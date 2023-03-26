import numpy as np

# Task 1

def mse(y_true:np.ndarray, y_predicted:np.ndarray):
    return np.sum((y_true-y_predicted)**2) / len(y_true)

def r2(y_true:np.ndarray, y_predicted:np.ndarray):
    u=np.sum((y_true-y_predicted)**2)
    v=np.sum((y_true-np.mean(y_true))**2)
    return 1-u/v

# Task 2

class NormalLR:
    def __init__(self):
        self.weights = None # Save weights here
    
    def fit(self, X:np.ndarray, y:np.ndarray):
        X=np.hstack((np.array([[1]*X.shape[0]]).T, X))
        self.weights=np.linalg.inv(X.T@X)@X.T@y
    
    def predict(self, X:np.ndarray) -> np.ndarray:
        X=np.hstack((np.array([[1]*X.shape[0]]).T, X))
        return(X @ self.weights)
        
    
# Task 3

class GradientLR:
    def __init__(self, alpha:float, iterations=10000, l=0.):
        self.weights = None # Save weights here
        self.iterations=iterations
        self.reg_coef=l
        self.learn_rate=alpha
    
    def fit(self, X:np.ndarray, y:np.ndarray):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        weights= np.zeros(X.shape[1])
        for _ in range(self.iterations):
            grad=2*(X.T @ (X @ weights - y))/X.shape[0] + self.reg_coef*np.sign(weights)
            weights=weights-self.learn_rate*grad
        self.weights=weights

    def predict(self, X:np.ndarray):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        return(X @ self.weights)

# Task 4

def get_feature_importance(linear_regression):
    return list(np.abs(linear_regression.weights[1:]))

def get_most_important_features(linear_regression):
    return list(np.argsort(np.abs(linear_regression.weights[1:])))[::-1]


