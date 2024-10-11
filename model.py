
# This script defines a simple model class that uses least squares to fit 
# a linear model to data

import numpy as np
import torch.nn as nn

from data import LinearDataGen

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


class ModelNNLinearRegression(nn.Module):

    def __init__(self, d_features=1, output_size=1):
        super().__init__()
        self.d = d_features # number of parameters 
        self.net = nn.Linear(in_features=d_features, out_features=output_size)

    
    def forward(self, X):
        """
        X : n x d_features tensor
        """
        return self.net(X)
    

if __name__ == "__main__":

    linear_data = LinearDataGen(ndata=1000)
    Xtrain, ytrain, Xtest, ytest = linear_data.get_train_test_data()

    # define least squares model
    linear_model = ModelLinearRegression(d=2)
    linear_model.fit(Xtrain, ytrain)
    print(f"True parameters: {linear_data.theta_star}, estimated parameters: {linear_model.parm}")

    mse_test = linear_model.error(Xtest, ytest)
    print(f"MSE on test set: {mse_test}")
    

    

