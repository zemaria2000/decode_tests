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
from sklearn.preprocessing import StandardScaler, LabelEncoder


def pre_process_iris(seed: int, test_split: float = 0.2, scaler: str = 'stand'):
    """
        Pre-processing approach to the iris dataset
        Args:
            - seed (int): seed for random intialization
            - test_split (float): test split for the train-test datasets
            - scaler (str): string determining the scaler to be employed
        Returns:
            - trainX (np.ndarray): train features
            - testX (np.ndarray): test features
            - trainY (np.ndarray): train targets
            - testY (np.ndarray): test targets
            - trainX_scaled (np.ndarray): normalised trainX
            - testX_scaled (np.ndarray): normalised testX
            - scaler: fitted scaler
            - feature_names (list): list with all feature names
            - target_names (list): list with all target names
            - feature_types (list): list with the feature types for the variables ('C' for categoricals, 'N' for numericals)
    """

    iris = datasets.load_iris(as_frame = False)

    data = iris['data']
    target = iris['target']
    feature_names = iris['feature_names']
    target_names = iris['target_names']

    # Getting the X and y values
    X = data
    y = target

    # Creating a train-test split
    trainX, testX, trainY, testY = train_test_split(X, y, test_size = test_split, random_state = seed, stratify = y, shuffle = True)

    # Normalising the data 
    if scaler == 'stand':
        scaler = StandardScaler()
        trainX_scaled = scaler.fit_transform(trainX)
        testX_scaled = scaler.transform(testX)
    else:
        scaler = None
        trainX_scaled = None
        testX_scaled = None

    # Just creating a label encoder for the target variables and the class names
    labels_encoded = dict(zip(np.arange(len(target_names)), target_names))
    features_encoded = dict(zip(np.arange(len(feature_names)), feature_names))
    feature_types = ['N', 'N', 'N', 'N']

    return trainX, testX, trainY, testY, trainX_scaled, testX_scaled, scaler, feature_names, target_names, feature_types


def pre_process_mushrooms(seed: int, test_split: float = 0.2, scaler: str = 'stand'):
    """
        Pre-processing approach to the mushrooms dataset
        Args:
            - seed (int): seed for random intialization
            - test_split (float): test split for the train-test datasets
            - scaler (str): string determining the scaler to be employed
        Returns:
            - trainX (np.ndarray): train features
            - testX (np.ndarray): test features
            - trainY (np.ndarray): train targets
            - testY (np.ndarray): test targets
            - trainX_scaled (np.ndarray): normalised trainX
            - testX_scaled (np.ndarray): normalised testX
            - scaler: fitted scaler
            - feature_names (list): list with all feature names
            - target_names (list): list with all target names
            - feature_types (list): list with the feature types for the variables ('C' for categoricals, 'N' for numericals)
    """

    dataset = pd.read_csv('./datasets/mushrooms.csv')

    # Dropping rows with NaNs
    dataset.dropna(inplace = True)

    # Getting the X and y values
    X = dataset.iloc[:, 1:].values
    y = dataset.iloc[:, 0].values
    feature_types = ['C' for _ in range(X.shape[1])]
    feature_names = list(dataset.columns[1:])
    target_names = np.unique(y)

    # Handling categorical features
    cat_features_idx = [i for i, feat in enumerate(feature_types) if feat == 'C']
    num_features_idx = [i for i, feat in enumerate(feature_types) if feat == 'N']

    features_encoded = {}
    for i in cat_features_idx:
        le = LabelEncoder()
        X[:, i] = le.fit_transform(X[:, i])
        X[:, i] = X[:, i].astype(int)
        # Storing the feature encoding
        features_encoded[dataset.columns[i+1]] = dict(zip(le.transform(le.classes_), le.classes_))
    le = LabelEncoder()
    y = le.fit_transform(y)
    labels_encoded = dict(zip(le.transform(le.classes_), le.classes_))

    X = X.astype(int)
    y = y.astype(int)

    # Train test splits
    trainX, testX, trainY, testY = train_test_split(X, y, test_size=test_split, random_state=seed, stratify=y)

    # Normalising the data 
    if scaler == 'stand':
        scaler = StandardScaler()
        trainX_scaled = scaler.fit_transform(trainX)
        testX_scaled = scaler.transform(testX)
    else:
        scaler = None
        trainX_scaled = None
        testX_scaled = None

    return trainX, testX, trainY, testY, trainX_scaled, testX_scaled, scaler, feature_names, target_names, feature_types


def pre_process_hawks(seed: int, test_split: float = 0.2, scaler: str = 'stand'):
    """
        Pre-processing approach to the hawks dataset
        Args:
            - seed (int): seed for random intialization
            - test_split (float): test split for the train-test datasets
            - scaler (str): string determining the scaler to be employed
        Returns:
            - trainX (np.ndarray): train features
            - testX (np.ndarray): test features
            - trainY (np.ndarray): train targets
            - testY (np.ndarray): test targets
            - trainX_scaled (np.ndarray): normalised trainX
            - testX_scaled (np.ndarray): normalised testX
            - scaler: fitted scaler
            - feature_names (list): list with all feature names
            - target_names (list): list with all target names
            - feature_types (list): list with the feature types for the variables ('C' for categoricals, 'N' for numericals)
    """

    dataset = pd.read_csv('./datasets/Hawks.csv')

    # Analysing the last columns, given that they have a considerable amount of NaN values
    for column in dataset.columns:
        isna_list = dataset[column].isna().values
        print('Number of NaN values for column', column, '-', len(isna_list[isna_list == True]))
    columns_to_drop = ['rownames', 'StandardTail', 'Tarsus', 'WingPitFat', 'KeelFat', 'Crop', 'ReleaseTime']
    dataset.drop(columns = columns_to_drop, inplace = True)

    # Attempting to convert the Month + Day + Year + CaptureTime into a datetime object
    target_ds = dataset[['Month', 'Day', 'Year', 'CaptureTime']].values
    date_format = '%Y-%m-%d %H:%M'
    dates = []
    for m, d, y, t in target_ds:
        if len(str(t).split('.')) > 1:
            t = str(t)
            t = t.replace('.', ':')
        elif str(t) == ' ':
            t = str(t)
            t = '12:00'
        date_string = f'{str(y)}-{str(m)}-{str(d)} {t}'
        
        dates.append(datetime.strptime(date_string, date_format).timestamp())
    
    # Dropping all time columns, and putting the new one
    dataset.drop(columns = ['Month', 'Day', 'Year', 'CaptureTime'], inplace = True)
    dataset.insert(loc = 0, column = 'Time', value = dates)

    # Replacing the 'Sex' NaN values as 'Unknown'
    dataset['Sex'].fillna('U', inplace = True)

    # Dropping the BandNumber column - a specific indicator per individual sample...
    dataset.drop(columns = ['BandNumber'], inplace = True)

    # Changing feature types
    dataset['Time'] = dataset['Time'].astype(int)
    dataset['Wing'] = dataset['Wing'].astype(float)
    dataset['Weight'] = dataset['Weight'].astype(float)
    dataset['Culmen'] = dataset['Culmen'].astype(float)
    dataset['Hallux'] = dataset['Hallux'].astype(float)
    dataset['Tail'] = dataset['Tail'].astype(float)

    dataset.dropna(inplace = True)

    # X and y values
    y = dataset['Species'].values
    X_ds = dataset.drop(columns = ['Species'])
    X = X_ds.values

    feature_types = ['N', 'C', 'C', 'N', 'N', 'N', 'N', 'N']
    feature_names = list(X_ds.columns)
    target_names = np.unique(y)

    ## Preprocessing
    cat_features_idx = [i for i, feat in enumerate(feature_types) if feat == 'C']
    num_features_idx = [i for i, feat in enumerate(feature_types) if feat == 'N']

    feature_encoding = {}
    for i in cat_features_idx:
        le = LabelEncoder()
        X[:, i] = le.fit_transform(X[:, i])
        X[:, i] = X[:, i].astype(int)
        # Storing the feature encoding
        feature_encoding[dataset.columns[i+1]] = dict(zip(le.transform(le.classes_), le.classes_))
    X = X.astype(float)

    le = LabelEncoder()
    y = le.fit_transform(y)
    label_encoding = dict(zip(le.transform(le.classes_), le.classes_))


    # 2. Train test splits
    trainX, testX, trainY, testY = train_test_split(X, y, test_size=test_split, random_state=seed, stratify=y)

    # 3. Feature normalisation
    if scaler == 'stand':
        scaler = StandardScaler()
        trainX_scaled, testX_scaled = trainX.copy(), testX.copy()
        trainX_scaled[:, num_features_idx] = scaler.fit_transform(trainX_scaled[:, num_features_idx])
        testX_scaled[:, num_features_idx] = scaler.transform(testX_scaled[:, num_features_idx])
    else:
        scaler = None
        trainX_scaled = None
        testX_scaled = None


    return trainX, testX, trainY, testY, trainX_scaled, testX_scaled, scaler, feature_names, target_names, feature_types


def pre_process_penguins(seed: int, test_split: float = 0.2, scaler: str = 'stand'):
    """
        Pre-processing approach to the mushrooms dataset
        Args:
            - seed (int): seed for random intialization
            - test_split (float): test split for the train-test datasets
            - scaler (str): string determining the scaler to be employed
        Returns:
            - trainX (np.ndarray): train features
            - testX (np.ndarray): test features
            - trainY (np.ndarray): train targets
            - testY (np.ndarray): test targets
            - trainX_scaled (np.ndarray): normalised trainX
            - testX_scaled (np.ndarray): normalised testX
            - scaler: fitted scaler
            - feature_names (list): list with all feature names
            - target_names (list): list with all target names
            - feature_types (list): list with the feature types for the variables ('C' for categoricals, 'N' for numericals)
    """

    path = kagglehub.dataset_download("larsen0966/penguins")
    dataset = pd.read_csv(path + '/penguins.csv', index_col = 'Unnamed: 0')

    dataset.dropna(inplace = True)

    # X and y values
    X = dataset.iloc[:, 1:].values
    y = dataset['species'].values
    feature_types = ['C', 'N', 'N', 'N', 'N', 'C', 'C']
    feature_names = list(dataset.columns[1:])
    target_names = np.unique(y)


    ## Preprocessing
    cat_features_idx = [i for i, feat in enumerate(feature_types) if feat == 'C']
    num_features_idx = [i for i, feat in enumerate(feature_types) if feat == 'N']

    feature_encoding = {}
    for i in cat_features_idx:
        le = LabelEncoder()
        X[:, i] = le.fit_transform(X[:, i])
        X[:, i] = X[:, i].astype(int)
        # Storing the feature encoding
        feature_encoding[dataset.columns[i+1]] = dict(zip(le.transform(le.classes_), le.classes_))
    X = X.astype(float)

    le = LabelEncoder()
    y = le.fit_transform(y)
    label_encoding = dict(zip(le.transform(le.classes_), le.classes_))


    # 2. Train test splits
    trainX, testX, trainY, testY = train_test_split(X, y, test_size=test_split, random_state=seed, stratify=y)

    # 3. Feature normalisation
    if scaler == 'stand':
        scaler = StandardScaler()
        trainX_scaled, testX_scaled = trainX.copy(), testX.copy()
        trainX_scaled[:, num_features_idx] = scaler.fit_transform(trainX_scaled[:, num_features_idx])
        testX_scaled[:, num_features_idx] = scaler.transform(testX_scaled[:, num_features_idx])
    else:
        scaler = None
        trainX_scaled = None
        testX_scaled = None

    return trainX, testX, trainY, testY, trainX_scaled, testX_scaled, scaler, feature_names, target_names, feature_types


if __name__ == '__main__':

    pre_process_hawks(seed = 17)
    pre_process_penguins(seed = 17)
    pre_process_mushrooms(seed = 17)