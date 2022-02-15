from PolynomialRegression import pol_regression, polyMatrix, eval_pol_regression
from kMeans import kmeans
from HIVAnalysis import summary, normalise, plot, ANN, randomForest
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


def cleanDataFrame(df):
    # Remove null values and duplicate rows
    df = df.drop_duplicates()
    df = df.dropna()
    return df


def splitDataFrame(df, testSize, shuffleBool):
    # Split the data into a train test split
    train, test = train_test_split(df, test_size=testSize, shuffle=shuffleBool)
    return train, test


def runPolynomialRegression():
    pol = pd.read_csv('datasets/Task1 - dataset - pol_regression.csv', index_col=0)  # Read the CSV file
    pol = cleanDataFrame(pol)  # Clean the dataframe
    pol = pol.sort_values(by=['x'])  # Sort the data frame so that the line draws nicely

    x = pol['x']
    y = pol['y']

    # Split into test and train for each dimension
    train, test = splitDataFrame(pol, 0.3, False)

    x_train = train['x'].to_numpy()
    y_train = train['y'].to_numpy()
    x_test = test['x'].to_numpy()
    y_test = test['y'].to_numpy()

    # PLot the test data in green, and the train data in blue
    plt.figure()
    plt.title("Polynomial Regression")
    plt.plot(x_train, y_train, 'go')
    plt.plot(x_test, y_test, 'bo')

    # For each given degree, train the model, then plot the result of testing on x
    w0 = pol_regression(x, y, 0)
    Xtest0 = polyMatrix(x, 0)
    ytest0 = Xtest0.dot(w0)
    plt.plot(x, ytest0, 'r')

    w1 = pol_regression(x, y, 1)
    Xtest1 = polyMatrix(x, 1)
    ytest1 = Xtest1.dot(w1)
    plt.plot(x, ytest1, 'r--')

    w2 = pol_regression(x, y, 2)
    Xtest2 = polyMatrix(x, 2)
    ytest2 = Xtest2.dot(w2)
    plt.plot(x, ytest2, 'g')

    w3 = pol_regression(x, y, 3)
    Xtest3 = polyMatrix(x, 3)
    ytest3 = Xtest3.dot(w3)
    plt.plot(x, ytest3, 'm--')

    w6 = pol_regression(x, y, 6)
    Xtest6 = polyMatrix(x, 6)
    ytest6 = Xtest6.dot(w6)
    plt.plot(x, ytest6, 'c')

    w10 = pol_regression(x, y, 10)
    Xtest10 = polyMatrix(x, 10)
    ytest10 = Xtest10.dot(w10)
    plt.plot(x, ytest10, 'c--')

    plt.legend(('training points', 'ground truth', '$x0$', '$x^1$', '$x^2$', '$x^3$', '$x^6$', '$X^10$'), loc='lower right')
    plt.xlim([-5, 5])
    plt.show()

    plt.figure()
    plt.title("Polynomial Square Root Mean Error")
    plt.xlabel("Degrees")
    plt.ylabel("Error")

    degrees = [0, 1, 2, 3, 6, 10]
    trainError = []
    testError = []

    for degree in degrees:
        w = pol_regression(x_train, y_train, degree)
        error = eval_pol_regression(w, x_train, y_train, degree)
        trainError.append(error)

    for degree in degrees:
        w = pol_regression(x_test, y_test, degree)
        error = eval_pol_regression(w, x_test, y_test, degree)
        testError.append(error)

    plt.plot(trainError, 'g')
    plt.plot(testError, 'r')
    plt.legend(('training error', 'test error'),
               loc='upper right')
    plt.show()


def runKmeansClustering():
    kdf = pd.read_csv('datasets/Task2 - dataset - dog_breeds.csv')  # Read the CSV file
    kdf = cleanDataFrame(kdf)  # Clean the dataframe
    kdf = kdf.values

    # k=2
    centroidsk2, cluster_assignedk2 = kmeans(kdf, 2)

    # k=3
    centroidsk3, cluster_assignedk3 = kmeans(kdf, 3)

    # For k=2 - height x and tail length y
    plt.scatter(cluster_assignedk2[:, 0], cluster_assignedk2[:, 1], c=cluster_assignedk2[:, 4], s=10)  # Plot the values with the cluster value as colour
    plt.scatter(centroidsk2[:, 0], centroidsk2[:, 1], marker='*', s=200)  # Plot the centroid values
    plt.title("Dog Breeds - Height and Tail")
    plt.xlabel("Height")
    plt.ylabel("Tail Length")
    plt.show()

    # For k=3 - height x and tail length y
    plt.scatter(cluster_assignedk3[:, 0], cluster_assignedk3[:, 1], c=cluster_assignedk3[:, 4], s=10)
    plt.scatter(centroidsk3[:, 0], centroidsk3[:, 1], marker='*', s=200)
    plt.title("Dog Breeds - Height and Tail")
    plt.xlabel("Height")
    plt.ylabel("Tail Length")
    plt.show()

    # For k=2 - height x and leg length y
    plt.scatter(cluster_assignedk2[:, 0], cluster_assignedk2[:, 2], c=cluster_assignedk2[:, 4], s=10)
    plt.scatter(centroidsk2[:, 0], centroidsk2[:, 2], marker='*', s=200)
    plt.title("Dog Breeds - Height and Leg")
    plt.xlabel("Height")
    plt.ylabel("Leg Length")
    plt.show()

    # For k=3 - height x and leg length y
    plt.scatter(cluster_assignedk3[:, 0], cluster_assignedk3[:, 2], c=cluster_assignedk3[:, 4], s=10)
    plt.scatter(centroidsk3[:, 0], centroidsk3[:, 2], marker='*', s=200)
    plt.title("Dog Breeds - Height and Leg")
    plt.xlabel("Height")
    plt.ylabel("Leg Length")
    plt.show()


def runHivAnalysis():
    hiv = pd.read_csv('datasets/Task3 - dataset - HIV RVG.csv')
    hiv = cleanDataFrame(hiv)  # Remove duplicate and null values

    summary(hiv)  # Print a summary of the dataset

    hivNormalised = normalise(hiv)  # Normalise the dataset
    plot(hivNormalised)

    train, test = splitDataFrame(hivNormalised, 0.1, True)  # Split the data into 90% training and 10% testing, shuffled

    acuracy, plt = ANN(train, test, showPlot=True)  # Run an artificial neural network with 500 neurons in each hidden layer
    plt.show()
    print("Accuracy of Random Forest Classifier with 1000 trees and 5 samples per node: {}".format(randomForest(train, test)))  # Run a random forest classifier with 1000 trees and 5 samples per node
    print("Accuracy of Random Forest Classifier with 1000 trees and 10 samples per node: {}".format(randomForest(train, test, numSamples=10)))  # Run a random forest classifier with 1000 trees and 10 samples per node

    splits = []

    for i in range(10):
        splits.append(hivNormalised.sample(frac=0.1))

    variations = [50, 500, 1000]
    accuracyANN = []
    accuracyRFC = []

    for value in variations:
        for split in splits:
            train, test = splitDataFrame(split, 0.1, True)
            accuracyANN.append(ANN(train, test, value)[0])
            accuracyRFC.append(randomForest(train, test, value, 10))
        meanAccuracyANN = sum(accuracyANN)/len(accuracyANN)
        meanAccuracyRFC = sum(accuracyRFC) / len(accuracyRFC)
        print("Mean accuracy of ANN with {} neurons and 10 folds is {}".format(value, meanAccuracyANN))
        print("Mean accuracy of RFC with {} trees and 10 folds is {}".format(value, meanAccuracyRFC))


if __name__ == "__main__":
    # runPolynomialRegression()  # Run polynomial regression task
    runKmeansClustering()  # Run k-means clustering task
    # runHivAnalysis()
