import numpy as np
import sklearn.metrics
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
import csv
import os

samples_directory = "./test_data"

data_file = "dataset2.csv"
test_file = "testset.csv"

df = pd.read_csv(data_file).as_matrix()
num_features = 24

test_df = pd.read_csv(test_file).as_matrix()
classes = ["BassClarinet", "BassTrombone", "BbClarinet", "Cello", "EbClarinet", "Marimba", "TenorTrombone", "Viola", "Violin", "Xylophone"]

def write_submission(test_predictions):
    current_file = 0
    file_handler = open("results", 'w')
    csv_writer = csv.writer(file_handler)
    for file_name in os.listdir(samples_directory):
        instrument = classes[int(test_predictions[current_file])]
        csv_writer.writerow((file_name,instrument))
        current_file = current_file + 1


if __name__ == "__main__":
    target = df[:, 24]
    features = df[:, :24]
    x_train, x_test, y_train, y_test = train_test_split(
		features, target, test_size = 0.2, random_state = 0)
    print(features.shape)
    tree = tree.DecisionTreeClassifier(max_depth = 50)
    tree.fit(x_train, y_train)
    predictions = tree.predict(x_test)
    print(accuracy_score(y_test, predictions))

    test_features = test_df[:, :24]
    test_predictions = tree.predict(test_features)
    print("length of test predictions:", len(test_predictions))
    write_submission(test_predictions)

    

    