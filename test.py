# load and show an image with Pillow
import os.path

from PIL import Image
import numpy as np
import imageio
import matplotlib.pyplot as plt
import cv2
import pickle
from sklearn.model_selection import train_test_split
import pandas as pd

im = cv2.imread("output.png")

DATADIR = "./all/"
#CATEGORIES = ["ก", "ข"]
CATEGORIES = []


# for category in CATEGORIES:
#     path = os.path.join(DATADIR, category)
#     for img in os.listdir(path):
#         img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
#         plt.imshow(img_array, cmap="gray")
#         plt.show()
#         break
#     break

def create_training_data(training_data):
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        print(path)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (145, 145))
            training_data.append([new_array, class_num])

def features_labels(training_data):
    X = []
    y = []
    for features, label in training_data:
        X.append(features)
        y.append(label)

    X = np.array(X).reshape(-1, 145, 145, 1)
    y = np.array(y)

    pickle_out = open("X.pickle", "wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open("y.pickle", "wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()

    return X, y

if __name__ == '__main__':
    df = pd.read_csv("categories.csv", header=None)
    data = df.values.tolist()
    for character in data:
        print(character)
        CATEGORIES.append(character[0])  # need 0 as it is list in list
    training_data = []
    create_training_data(training_data)
    X, y = features_labels(training_data)
    #print("i am gay")
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


