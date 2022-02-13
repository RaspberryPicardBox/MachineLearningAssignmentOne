import numpy as np


def polyMatrix(x, degree):
    # Expands the dataframe by a degree
    X = np.ones(x.shape)
    for i in range(1, degree + 1):
        X = np.column_stack((X, x ** i))
    return X


def pol_regression(features_train, y_train, degree):
    # Take the data and return polynomial weights
    if degree == 0:
        parameters = sum(features_train)/len(features_train)
    else:
        X = polyMatrix(features_train, degree)
        XX = X.transpose().dot(X)
        parameters = np.linalg.solve(XX, X.transpose().dot(y_train))
    return parameters


def eval_pol_regression(parameters, x, y, degree):
    # Evaluates the error of the regression using root-mean-square error
    Xtest = polyMatrix(x, degree)
    ytest = Xtest.dot(parameters)

    rmse = np.sqrt(np.square(y - ytest).sum() / np.square(y - ytest).size)

    return rmse

