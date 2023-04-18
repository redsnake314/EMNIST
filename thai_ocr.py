import cv2
import numpy as np
import tensorflow as tf
import test
from sklearn.model_selection import train_test_split
import pickle

print(tf.__version__)

# Load the dataset
# data = np.load('thai-alphabet.npz')
# images = data['image']
# labels = data['label']

# Preprocess the data
# images = [cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (32, 32)) for img in images]
# images = np.array(images).reshape(-1, 32, 32, 1) / 255.0
# labels = tf.keras.utils.to_categorical(labels)

training_data = []

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)
pickle_in.close()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Split the dataset into training and testing sets


# Define the CNN

#original from chatgpt was it??? idk
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(145, 145, 1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(44, activation='softmax')
])

# model = tf.keras.models.Sequential([
    #     tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(145, 145, 1)),
#     tf.keras.models.Conv2D(64, (3, 3), activation = 'relu'),
#     tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
#     tf.keras.models.Dropout(0.2),
#     tf.keras.models.Flatten(),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(64, activation='softmax'),
#     tf.keras.models.Dropout(0.2),
#     tf.keras.models.Dense(44, activation = 'softmax')
#
# ])


# model = tf.keras.models.Sequential()
# model.add(tf.keras.models.Conv2D(32, (3, 3), activation = 'relu', input_shape=(145, 145, 1)))
# model.add(tf.keras.models.Conv2D(64, (3, 3), activation = 'relu'))
# model.add(tf.keras.models.MaxPooling2D(pool_size = (2, 2)))
# model.add(tf.keras.models.Dropout(0.2))
# model.add(tf.keras.models.Flatten())
# model.add(tf.keras.models.Dense(128, activation = 'relu'))
# model.add(tf.keras.models.Dense(64, activation = 'relu'))
# model.add(tf.keras.models.Dropout(0.2))
# model.add(tf.keras.models.Dense(44, activation = 'softmax')) #(0-9) + (A-Z) = 36
# model.compile(loss = 'sparse_categorical_crossentropy', optimizer = tf.keras.models.optz, metrics = ['accuracy'])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
print(X_train.shape)
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)

model.save("gay_thai_32.h5")
