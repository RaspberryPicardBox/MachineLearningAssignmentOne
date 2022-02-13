from PolynomialRegression import pol_regression, polyMatrix, eval_pol_regression
from kMeans import compute_euclidean_distance, initialise_centroids, kmeans
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


def cleanDataFrame(df):
    # Remove null values and duplicate rows
    df = df.drop_duplicates()
    df = df.dropna()
    return df


def splitDataFrame(df):
    # Split the data into a train test split
    train, test = train_test_split(df, test_size=0.2, shuffle=False)
    return train, test


def runPolynomialRegression():
    pol = pd.read_csv('datasets/Task1 - dataset - pol_regression.csv', index_col=0)  # Read the CSV file
    pol = cleanDataFrame(pol)  # Clean the dataframe
    pol = pol.sort_values(by=['x'])  # Sort the data frame so that the line draws nicely

    x = pol['x']
    y = pol['y']

    # Split into test and train for each dimension
    train, test = splitDataFrame(pol)

    x_train = train['x'].to_numpy()
    y_train = train['y'].to_numpy()
    x_test = test['x'].to_numpy()
    y_test = test['y'].to_numpy()

    # PLot the test data in green, and the train data in blue
    plt.figure()
    plt.title("Polynomial Regression")
    plt.plot(x_test, y_test, 'go')
    plt.plot(x_train, y_train, 'bo')

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

    plt.show()

    plt.figure()
    plt.title("Polynomial Square Root Mean Error")
    plt.xlabel("Degrees")
    plt.ylabel("Error")

    degrees = [0, 1, 2, 3, 6, 10]
    trainError = []
    testError = []

    for degree in [x for x in range(11) if x in degrees]:
        w = pol_regression(x_train, y_train, degree)
        error = eval_pol_regression(w, x_train, y_train, degree)
        trainError.append(error)

    for degree in [x for x in range(11) if x in degrees]:
        w = pol_regression(x_test, y_test, degree)
        error = eval_pol_regression(w, x_test, y_test, degree)
        testError.append(error)

    plt.plot(trainError, 'g')
    plt.plot(testError, 'r')
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


if __name__ == "__main__":
    # runPolynomialRegression()  # Run polynomial regression task
    runKmeansClustering()  # Run k-means clustering task
