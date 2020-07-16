import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec

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
    
    return outlier_fraction

def main():
    data = pd.read_csv('Credit-card-dataset/creditcard.csv')
    features, gs = visualiseFeatures(data)
    for i, c in enumerate(data[features]):
        ax = plt.subplot(gs[i])
        sns.distplot(data[c][data.Class == 1], bins=50)
        sns.distplot(data[c][data.Class == 0], bins=50)
        ax.set_xlabel('')
        ax.set_title('histogram of feature: ' + str(c))
    plt.show()

    outlier_fraction = determineFraud(data)
    print(outlier_fraction)
    print('Fraud Cases: {}'.format(len(data[data['Class'] == 1])))
    print('Valid Transactions: {}'.format(len(data[data['Class'] ==0])))
if __name__ == "__main__":
    main()
