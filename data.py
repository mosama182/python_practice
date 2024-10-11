import numpy as np

class LinearDataGen():

    def __init__(self, ndata=1000):
        # generate data
        self.theta_star = np.array([1, 2]).reshape(-1, 1)
        self.ndata = ndata
        self.X = np.concatenate([np.ones((ndata, 1)), np.linspace(start=-5, stop=10, num=ndata).reshape(-1, 1)], axis=1)
        self.y = self.X @ self.theta_star
        self.y = self.y.reshape(-1, 1)
        self.y += np.random.normal(loc=0, scale=0.2, size=(ndata, 1))


    def get_train_test_data(self):
        # split training and test
        indices = np.arange(self.ndata)
        np.random.shuffle(indices)
        train_indices = indices[:800]
        test_indices = indices[800:]
        Xtrain, ytrain = self.X[train_indices], self.y[train_indices]
        Xtest, ytest = self.X[test_indices], self.y[test_indices]

        return Xtrain, ytrain, Xtest, ytest