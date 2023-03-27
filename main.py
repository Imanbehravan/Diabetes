from utils.utils import read_yaml
import pandas as pd
from parallelAPSO.APSOClustering import *
from sklearn.model_selection import train_test_split

config_file = read_yaml("Config.yaml")
DataPath = config_file["DEFAULT"]["DataPath"]
dataset = pd.read_csv(DataPath)
targetsName = config_file["DEFAULT"]["targetField"]
targets = dataset[targetsName]
dataset = dataset.drop(columns=targetsName)

X_train, X_test, y_train, y_test = train_test_split(dataset, targets, test_size=0.3, shuffle=True, random_state=1)

X_train = X_train.reset_index()
X_train = X_train.drop(columns="index")

y_train = y_train.reset_index()
y_train = y_train.drop(columns="index")

X_test = X_test.reset_index()
X_test = X_test.drop(columns="index")

y_test = y_test.reset_index()
y_test = y_test.drop(columns="index")

centroids = APSO_Clustering(config_file, X_train, [])


