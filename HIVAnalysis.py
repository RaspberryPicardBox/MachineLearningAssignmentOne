import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing, neural_network, ensemble
import seaborn as sns
import warnings

from sklearn.exceptions import ConvergenceWarning


def summary(df):
    # Show summary of the data
    df = df.drop('Participant Condition', axis=1)

    for col in df:
        print("Summary of Column {}:".format(col))
        print("Max = {}".format(df[col].max()))
        print("Min = {}".format(df[col].min()))
        print("Mean = {}".format(df[col].mean()))
        print("Median = {}".format(df[col].median()))
        print("Mode = {}".format(df.mode()[col][0]))
        print("Standard Deviation = {}".format(df[col].std()))
        print("\n")


def normalise(df):
    # Drop the string columns, normalise data, and rejoin
    partCon = df['Participant Condition']
    df = df.drop('Participant Condition', axis=1)
    names = df.columns
    normalised = pd.DataFrame(preprocessing.normalize(df, axis=0), columns=names).join(partCon)
    return normalised


def plot(df):
    # Plot the participant condition against Alpha as a box plot, and Beta as a density graph
    partCon = df['Participant Condition']
    sns.boxplot(x=df['Participant Condition'], y=df['Alpha'])
    plt.show()
    sns.kdeplot(df['Beta'])
    plt.show()


def ANN(train, test, numNeurons=500, showPlot=False):
    # Fit a multi-layer ANN classifier and predict values of participant condition for test set

    lEnc = preprocessing.LabelEncoder()  # Preprocess the label as encoded int
    lEnc.fit(['Patient', 'Control'])

    y_train = lEnc.transform(train['Participant Condition'])
    X_train = train.drop('Participant Condition', axis=1)  # Remove the label from X

    y_test = lEnc.transform(test['Participant Condition'])
    X_test = test.drop('Participant Condition', axis=1)  # Remove the labels from the test set

    clf = neural_network.MLPClassifier(hidden_layer_sizes=(numNeurons, numNeurons), activation='relu')
    clf.fit(X_train, y_train)

    num_epochs = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 1000]
    accuracies = []
    if showPlot == True:
        for max_epoch in num_epochs:
            clf = neural_network.MLPClassifier(hidden_layer_sizes=(numNeurons, numNeurons), activation='logistic', max_iter=max_epoch)
            warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
            clf.fit(X_train, y_train)
            accuracies.append(clf.score(X_test, y_test))
        plt.plot(num_epochs, accuracies)
        plt.title("ANN with {} Neurons".format(numNeurons))
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")

    accuracy = clf.score(X_test, y_test)
    return accuracy, plt


def randomForest(train, test, numTrees=1000, numSamples=5):
    # Fit a random forest classifier and predict values of participant condition for test set

    lEnc = preprocessing.LabelEncoder()  # Preprocess the label as encoded int
    lEnc.fit(['Patient', 'Control'])

    y_train = lEnc.transform(train['Participant Condition'])
    X_train = train.drop('Participant Condition', axis=1)  # Remove the label from X

    y_test = lEnc.transform(test['Participant Condition'])
    X_test = test.drop('Participant Condition', axis=1)  # Remove the labels from the test set

    clf = ensemble.RandomForestClassifier(n_estimators=numTrees, min_samples_leaf=numSamples)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)

    return accuracy
