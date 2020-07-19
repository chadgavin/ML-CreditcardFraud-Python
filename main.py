import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest

def visualiseFeatures(data):
    print(data.shape)
    print(data.describe())
    # distribution of anomalous features
    features = data.iloc[:,0:28].columns
    plt.figure(figsize=(12,28*4))
    gs = gridspec.GridSpec(28, 1)
    
    return features, gs

def determineFraud(data):
    Fraud = data[data['Class'] == 1]
    Valid = data[data['Class'] == 0]
    outlier_fraction = len(Fraud)/float(len(Valid))
    print(outlier_fraction)
    print('Fraud Cases: {}'.format(len(data[data['Class'] == 1])))
    print('Valid Transactions: {}'.format(len(data[data['Class'] ==0])))
    print("Amount details of fraudulent transaction")
    
    return Fraud

def heatmap(data):
    corrmat = data.corr()
    fig = plt.figure(figsize = (12, 9))
    sns.heatmap(corrmat, vmax = .8, square = True)
    plt.show()
    
def divideToFeaturesNTarget(data):
    X=data.drop(["Class"], axis=1)
    Y=data['Class']
    print(X.shape)
    print(Y.shape)
    #getting just the values for the sake of processing (its a numpy array with no columns)
    X_data=X.values
    Y_data=Y.values
    X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size = 0.2, random_state = 42)

def main():
    data = pd.read_csv('creditcard.csv')
    features, gs = visualiseFeatures(data)
    # for i, c in enumerate(data[features]):
    #     ax = plt.subplot(gs[i])
    #     sns.distplot(data[c][data.Class == 1], bins=50)
    #     sns.distplot(data[c][data.Class == 0], bins=50)
    #     ax.set_xlabel('')
    #     ax.set_title('histogram of feature: ' + str(c))
    # plt.show()

    Fraud=determineFraud(data)
    print(Fraud.Amount.describe())
    heatmap(data)

    

if __name__ == "__main__":
    main()
