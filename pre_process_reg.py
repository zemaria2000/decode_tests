"""
    Simple file that includes a pre-processing approach to four datasets currently being study in the scope of this work: Iris, Hawks, Penguins, Mushrooms
"""

import pandas as pd 
import numpy as np
from munch import Munch
import kagglehub
from datetime import datetime
from sklearn import datasets


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler


def pre_process_diamonds(seed: int, test_split: float = 0.2, scaler: str = 'stand'):
    """
        Pre-processing approach to the diamonds dataset - https://www.kaggle.com/code/burakkyildiz/diamonds-support-vector-regression
        Args:
            - seed (int): seed for random intialization
            - test_split (float): test split for the train-test datasets
            - scaler (str): string determining the scaler to be employed
        Returns:
            - trainX (np.ndarray): train features
            - testX (np.ndarray): test features
            - trainY (np.ndarray): train target(s)
            - testY (np.ndarray): test target(s)
            - trainX_scaled (np.ndarray): normalised trainX
            - testX_scaled (np.ndarray): normalised testX
            - trainY_scaled (np.ndarray): normalised trainY
            - testY_scaled (np.ndarray): normalised testY
            - scalerX: fitted scaler for features
            - scalerY: fitted scaler for target(s)
            - feature_names (list): list with all feature names
            - target_names (list): list with all target names
            - feature_types (list): list with the feature types for the variables ('C' for categoricals, 'N' for numericals)
    """

    df = pd.read_csv('./datasets/diamonds.csv', index_col = 'Unnamed: 0')

    # Checking any diamonds with no length in one of their dimensions + dropping those
    df[(df['x'] == 0) | (df['y'] == 0) | (df['z'] == 0)]

    df = df.drop(df[df['x'] == 0].index)
    df = df.drop(df[df['y'] == 0].index)
    df = df.drop(df[df['z'] == 0].index)
    
    # Dropping some outliers - based on an analysis conducted by the authors of the repository
    df = df[(df['depth'] < 75) & (df['depth'] > 45)]
    df = df[(df['table'] < 75) & (df['table'] > 40)]
    df = df[(df['z'] < 30) & (df['z'] > 2)]
    df = df[(df['y'] < 20)]

    # Selecting X and y values + columns
    X = df.drop('price', axis = 1)
    y = df['price']

    target_names = ['price']
    feature_names = list(X.columns)

    X = X.values
    y = y.values

    # Defining the feature types
    feature_types = ['N', 'C', 'C', 'C', 'N', 'N', 'N', 'N', 'N']

    # Encoding the categorical features
    from sklearn.preprocessing import LabelEncoder

    cat_features_idx = [i for i, feat in enumerate(feature_types) if feat == 'C']
    num_features_idx = [i for i, feat in enumerate(feature_types) if feat == 'N']

    # features_encoded = {}
    for i in cat_features_idx:
        le = LabelEncoder()
        X[:, i] = le.fit_transform(X[:, i])
        X[:, i] = X[:, i].astype(int)
        print(X[:, i])

    X = X.astype(float)
    y = y.astype(float)


    # Train-test split
    def train_test_split_regression(X, y, test_size=0.2, b='auto', random_state=42):
        if isinstance(b, str):
            bins = np.histogram_bin_edges(y, bins=b)
            # remove the last index (end point)
            bins = bins[:-1]
        elif isinstance(b, int):
            bins = np.linspace(min(y), max(y), num=b, endpoint=False)
        else:
            raise Exception(f'Undefined bins {b}')
            
        #print(f'Bins: {bins}')
        groups = np.digitize(y, bins)
        #print(f'Group: {groups}')
        return train_test_split(X, y, test_size=test_size, stratify=groups, random_state=random_state)

    trainX, testX, trainY, testY = train_test_split_regression(X, y, test_size = test_split, b = 10, random_state = seed)

    # Data normalisation
    scalerX, scalerY = StandardScaler(), StandardScaler()
    trainX_scaled, testX_scaled = trainX.copy(), testX.copy()
    trainX_scaled[:, num_features_idx] = scalerX.fit_transform(trainX_scaled[:, num_features_idx])
    testX_scaled[:, num_features_idx] = scalerX.transform(testX_scaled[:, num_features_idx])

    trainY_scaled, testY_scaled = trainY.copy(), testY.copy()
    trainY_scaled = scalerY.fit_transform(trainY_scaled.reshape(-1, 1))
    testY_scaled = scalerY.transform(testY_scaled.reshape(-1, 1))

    return trainX, testX, trainY, testY, trainX_scaled, testX_scaled, trainY_scaled, testY_scaled, scalerX, scalerY, feature_names, target_names, feature_types


def pre_process_housing(seed: int, test_split: float = 0.2, scaler: str = 'stand'):
    """
        Pre-processing approach to the housing dataset 
        Args:
            - seed (int): seed for random intialization
            - test_split (float): test split for the train-test datasets
            - scaler (str): string determining the scaler to be employed
        Returns:
            - trainX (np.ndarray): train features
            - testX (np.ndarray): test features
            - trainY (np.ndarray): train target(s)
            - testY (np.ndarray): test target(s)
            - trainX_scaled (np.ndarray): normalised trainX
            - testX_scaled (np.ndarray): normalised testX
            - trainY_scaled (np.ndarray): normalised trainY
            - testY_scaled (np.ndarray): normalised testY
            - scalerX: fitted scaler for features
            - scalerY: fitted scaler for target(s)
            - feature_names (list): list with all feature names
            - target_names (list): list with all target names
            - feature_types (list): list with the feature types for the variables ('C' for categoricals, 'N' for numericals)
    """

    # Loading the df
    column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    df = pd.read_csv("./datasets/housing.csv", header=None, delimiter=r"\s+", names=column_names)

    feature_types = ['N', 'N', 'N', 'C', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N']

    # X and y values
    X = df.drop('MEDV', axis = 1)
    y = df['MEDV']

    target_names = ['MEDV']
    feature_names = list(X.columns)

    X = X.values
    y = y.values

    X = X.astype(float)
    y = y.astype(float)

    # Train-test split
    def train_test_split_regression(X, y, test_size=0.2, b='auto', random_state=42):
        if isinstance(b, str):
            bins = np.histogram_bin_edges(y, bins=b)
            # remove the last index (end point)
            bins = bins[:-1]
        elif isinstance(b, int):
            bins = np.linspace(min(y), max(y), num=b, endpoint=False)
        else:
            raise Exception(f'Undefined bins {b}')
            
        #print(f'Bins: {bins}')
        groups = np.digitize(y, bins)
        #print(f'Group: {groups}')
        return train_test_split(X, y, test_size=test_size, stratify=groups, random_state=random_state)

    trainX, testX, trainY, testY = train_test_split_regression(X, y, test_size = test_split, b = 10, random_state = seed)

    # Data normalisation
    cat_features_idx = [i for i, feat in enumerate(feature_types) if feat == 'C']
    num_features_idx = [i for i, feat in enumerate(feature_types) if feat == 'N']

    scalerX, scalerY = StandardScaler(), StandardScaler()
    trainX_scaled, testX_scaled = trainX.copy(), testX.copy()
    trainX_scaled[:, num_features_idx] = scalerX.fit_transform(trainX_scaled[:, num_features_idx])
    testX_scaled[:, num_features_idx] = scalerX.transform(testX_scaled[:, num_features_idx])

    trainY_scaled, testY_scaled = trainY.copy(), testY.copy()
    trainY_scaled = scalerY.fit_transform(trainY_scaled.reshape(-1, 1))
    testY_scaled = scalerY.transform(testY_scaled.reshape(-1, 1))

    return trainX, testX, trainY, testY, trainX_scaled, testX_scaled, trainY_scaled, testY_scaled, scalerX, scalerY, feature_names, target_names, feature_types


if __name__ == '__main__':
    pre_process_housing(seed = 17)
    pre_process_diamonds(seed = 17)