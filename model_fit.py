import torch.nn as nn
import torch.optim as optim
import torch

from model import ModelNNLinearRegression
from data import LinearDataGen


def fit_epoch(model, epoch, X, y, loss_func, optimizer):
    """
    X: n x d tensor
    y: n x 1 tensor
    Perform model fit for one epoch for
    the NN linear model
    """

    ypred = model.forward(X)
    loss = loss_func(ypred, y)

    optimizer.zero_grad() 
    loss.backward()
    optimizer.step()

    if (epoch % 10) == 0:
        print(f"Epoch: {epoch}, Train loss: {loss.item(): 0.4f}")


def fit_using_nn(xtrain, ytrain, nepochs=100):
    """
    xtrain: n x d tensor
    ytrain: n x 1 tensor
    """

    d = xtrain.shape[1]
    model = ModelNNLinearRegression(d_features=d)

    loss_func = nn.MSELoss()

    lr = 0.01
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    for epoch in range(nepochs):
        fit_epoch(model, epoch, xtrain, ytrain, loss_func, optimizer)

    return model


def model_fit_using_nn():

    linear_data = LinearDataGen(ndata=1000)
    Xtrain, ytrain, Xtest, ytest = linear_data.get_train_test_data()

    # fit using linear NN
    xtrain = torch.tensor(Xtrain[:, 1], dtype=torch.float32).reshape(-1, 1)
    ytrain = torch.tensor(ytrain, dtype=torch.float32).reshape(-1, 1)
    model = fit_using_nn(xtrain, ytrain, nepochs=200)
    
    # print params and test error
    print(f"True parameters: {linear_data.theta_star}")
    print("Estimated parameters")
    with torch.no_grad():
        for parm in model.parameters():
            print(parm)
        
        xtest = torch.tensor(Xtest[:, 1], dtype=torch.float32).reshape(-1, 1)
        ytest = torch.tensor(ytest, dtype=torch.float32).reshape(-1, 1)
        loss_func = nn.MSELoss()
        ypred = model(xtest)
        loss = loss_func(ypred, ytest)
        print(f"Test error: {loss.item():.4f}")



if __name__ == "__main__":

    model_fit_using_nn()
