import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import seaborn as sns
from math import ceil


def plot_distrib(features, labels):
    """ Plots the distibution of every feature """
    with plt.style.context('seaborn-whitegrid'):
        plt.figure(figsize=(8, 6))
        for i, column in enumerate(features.columns):
            plt.subplot(4,2,i+1)
            feat = features.loc[labels==0, column]
            plt.hist(feat,
                        label=0,
                        bins=10,
                        alpha=0.5, 
                        color="red", 
                        weights=[1/float(len(feat))]*len(feat))
            feat = features.loc[labels==1, column]
            plt.hist(feat,
                        label=1,
                        bins=10,
                        alpha=0.5, 
                        color="green", 
                        weights=[1/float(len(feat))]*len(feat))
            plt.xlabel(column)
        plt.legend(('Died', 'Survived'), loc="upper right", fancybox=True, fontsize=8)
        plt.tight_layout()
        plt.show()

def number_encode_features(df):
    """ Encodes every object into an integer """
    result = df.copy()
    encoders = {}
    for column in result.columns:
        if result.dtypes[column] == np.object:
            encoders[column] = preprocessing.LabelEncoder()
            result[column] = encoders[column].fit_transform(result[column])
    return result, encoders

def plot_correlation(df):
    """ Plots the corelation between all datapoints """
    encoded_data, _ = number_encode_features(df)
    sns.heatmap(encoded_data.corr(), square=True, cmap="RdBu_r")
    plt.show()

def plot_3PC(features, label):
    features = features[:,:3]
    principal_features_df = pd.DataFrame(data = features
                , columns = ['principal component 1', 'principal component 2', 'principal component 3'])

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    ax.set_xlabel('Principal Component - 1',fontsize=20)
    ax.set_ylabel('Principal Component - 2',fontsize=20)
    ax.set_zlabel('Principal Component - 3',fontsize=20)
    plt.title("3D Principal Component Analysis of Titanic",fontsize=20)

    targets = [0, 1]
    colors = ['r', 'g']
    for target, color in zip(targets,colors):
        indicesToKeep = label == target
        ax.scatter(principal_features_df.loc[indicesToKeep, 'principal component 1']
                , principal_features_df.loc[indicesToKeep, 'principal component 2'],
                principal_features_df.loc[indicesToKeep, 'principal component 3'],
                    c = color, s = 50)

    plt.legend(('Died', 'Survived'))

    plt.show()

def plot_2PC(features, label):
    features = features[:,:2]
    principal_features_df = pd.DataFrame(data = features
                , columns = ['principal component 1', 'principal component 2'])

    fig = plt.figure()
    ax = fig.add_subplot()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    ax.set_xlabel('Principal Component - 1',fontsize=20)
    ax.set_ylabel('Principal Component - 2',fontsize=20)
    plt.title("2D Principal Component Analysis of Titanic",fontsize=20)

    targets = [0, 1]
    colors = ['r', 'g']
    for target, color in zip(targets,colors):
        indicesToKeep = label == target
        ax.scatter(principal_features_df.loc[indicesToKeep, 'principal component 1']
                , principal_features_df.loc[indicesToKeep, 'principal component 2'],
                    c = color, s = 50)

    plt.legend(('Died', 'Survived'))

    plt.show()

def plot_variance(pca):
    fig, ax1 = plt.subplots()

    fig.suptitle("Explained variance of PCs")

    ax1.xaxis.grid()
    ax1.set_xlabel('Number of Principal Components')
    plt.xticks(range(7), ["PC-"+str(i+1) for i in range(7)])

    colour = "blue"
    ax1.plot(np.cumsum(pca.explained_variance_ratio_), color=colour)
    ax1.set_ylabel('Cumulative percentage of variance', color=colour)
    ax1.tick_params(axis='y', labelcolor=colour)

    ax1.set_ylim(bottom=0)

    colour = "green"
    ax2 = ax1.twinx()
    ax2.plot(pca.explained_variance_, color=colour)
    ax2.set_ylabel('Eigenvalue', color=colour)
    ax2.tick_params(axis='y', labelcolor=colour)

    fig.tight_layout()
    plt.show()

def plot_pcacomponents(pca, features):
    plt.matshow(np.absolute(pca.components_),cmap="YlOrRd")
    plt.yticks(range(7),["PC-"+str(i+1) for i in range(7)],fontsize=10)
    plt.colorbar()
    plt.xticks(range(len(features.columns)),features.columns)
    plt.title("PCs and dependencies on original features")
    plt.tight_layout()
    plt.show()

def scatter_dimensions(df, feature1, feature2):
    ax1 = df.plot.scatter(x=feature1, y=feature2, c="red", alpha=.2)
    plt.tight_layout()
    plt.show()
# * Load the data into a pandas data framce object
# Headers are as follow: 
# PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked
training_data = pd.read_csv("csv/train.csv")

# Separate the features and the label
features = training_data.drop("Survived", 1)
label = training_data["Survived"]

# * Pre-process the data
# Remove the features we don't need
features = features.drop(["PassengerId", "Ticket", "Name", "Cabin"], 1)

# Categorise the age
features.loc[features['Age']<= 18, 'Age'] = 0
features.loc[(features['Age']> 18) & (features['Age']<= 32), 'Age'] =1
features.loc[(features['Age']> 32) & (features['Age']<=48), 'Age'] = 2
features.loc[(features['Age']> 48) & (features['Age']<=64), 'Age'] = 3
features.loc[features['Age']> 64, 'Age'] = 4

# Replace NaNs with the most frequently encountered value of the feature
features.Embarked.value_counts()
fre_embarked = features.Embarked.mode()
fre_age = features.Age.mode()
features.fillna(features.mean(), inplace=True)
features['Age']=features.Age.fillna(fre_age[0])
features['Embarked']=features.Embarked.fillna(fre_embarked[0])

features, _ = number_encode_features(features)

# Replace non existent fares with the average fare
Av_Fare = features.Fare.mean()
features['Fare']=features.Fare.fillna(Av_Fare)

# Plot 2 dimensions
#scatter_dimensions(features, "Fare", "SibSp")
# Plot distribution
# plot_distrib(features.join(label), label)

# Plot correlation
# plot_correlation(features.join(label))

# Standardise data
scaler = preprocessing.StandardScaler()
features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)

# * Apply 3D PCA to the Data
pca = PCA()
pca_features = pca.fit_transform(features)
print("Explained variance: {}".format(pca.explained_variance_))
print("Explained variance ratio: {}".format(pca.explained_variance_ratio_))
print("Explained variance ratio cumsum: {}".format(np.cumsum(pca.explained_variance_ratio_)))

# Plot PCA
# plot_variance(pca)
# plot_pcacomponents(pca, features)
# plot_2PC(pca_features, label)
# plot_3PC(pca_features, label)