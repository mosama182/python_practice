
# This script defines a simple model class that uses least squares to fit 
# a linear model to data

import numpy as np

class ModelLinearRegression():
    """
    A simple model class that uses
    least squares to fit a linear model
    """

    def __init__(self, d=2):
        self.d = d # dimension of parameter
        self.parm = np.random.normal(loc=0, scale=1, size=(d, 1))

    
    def fit(self, X, y):
        """
        X : n x d numpy array
        y: n x 1 numpy array
        """
        self.parm = np.linalg.inv(X.T @ X) @ (X.T @ y)

    
    def predict(self, Xtest):
        """
        X : ntest x d numpy array
        """
        self.ypred = Xtest @ self.parm
        
        return self.ypred
    

    def error(self, Xtest, ytest):
        ypred = self.predict(Xtest)
        error = np.mean((ypred - ytest)**2)

        return error



if __name__ == "__main__":

    # generate data
    theta_star = np.array([1, 2])
    ndata = 1000
    X = np.concatenate([np.ones((ndata, 1)), np.linspace(start=-5, stop=10, num=ndata).reshape(-1, 1)], axis=1)
    y = X @ theta_star
    y = y.reshape(-1, 1)
    y += np.random.normal(loc=0, scale=0.2, size=(ndata, 1))

    # split training and test
    indices = np.arange(ndata)
    np.random.shuffle(indices)
    train_indices = indices[:800]
    test_indices = indices[800:]
    Xtrain, ytrain = X[train_indices], y[train_indices]
    Xtest, ytest = X[test_indices], y[test_indices]

    # define model
    linear_model = ModelLinearRegression(d=2)
    linear_model.fit(Xtrain, ytrain)
    print(f"True parameters: {theta_star}, estimated parameters: {linear_model.parm}")

    mse_test = linear_model.error(Xtest, ytest)
    print(f"MSE on test set: {mse_test}")