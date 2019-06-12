import pandas as pd
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import keras
import csv
from skimage import io
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.losses import categorical_crossentropy, binary_crossentropy
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras import metrics
import time
import codecs
from CustomTensorBoard import TrainValTensorBoard
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.utils import class_weight
import cv2


# probably have our own thing
#input_path = Path("input")
model_path = os.path.abspath('models')

poster_width = 224  
poster_height = 224  
poster_channels = 3  # RGB

epochs = 15
batch_size = 256
genres = 20


# use 224x224
# sigmoid for binary cross entropy
# categorical = ce
def create_cnn(height, width, channels, genres):
    cnn = Sequential([
        Conv2D(filters=64, strides = 2, kernel_size=(5, 5), activation="relu", input_shape=(height, width, channels)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(filters=128, strides = 2, kernel_size=(5, 5), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(filters=256, strides = 2, kernel_size=(5, 5), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(genres, activation='sigmoid')
    ])
    cnn.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=[metrics.binary_accuracy, metrics.categorical_accuracy, metrics.top_k_categorical_accuracy])
    return cnn

start_time = time.time()

r = csv.reader(codecs.open('milestone_labels.csv', 'r', encoding='utf8'))
x_train, y_train, x_dev, y_dev, x_test, y_test = [],[],[],[],[],[]
d = {}

# https://github.com/taoxinyi/multi-label-natural-scene-classification/blob/2784b452438d849c75f91c55de55ceeb8f6821b4/mlknn-rgb.py
# it has an MIT license so we gucci, maybe we can change it later so we don't have to cite.
for row in r:
    try:
        d[int(row[0].split('-')[0])] = list(map(int, row[1:]))
    except:
        pass

train_files, dev_files, test_files = os.listdir('train224x224/'), os.listdir('dev224x224/'), os.listdir('test224x224/')
# train_half = train_files[:len(train_files)//10]
# dev_half = dev_files[:len(dev_files)//10]
# test_half = test_files[:len(test_files)//10]

#image
for file in tqdm(train_files):
    img = io.imread("train224x224/" + file)
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    x_train.append(img)
    y_train.append(d[int(file.split('-')[0])])
for file in tqdm(dev_files):
    img = io.imread("dev224x224/" + file)
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    x_dev.append(img)
    y_dev.append(d[int(file.split('-')[0])])
for file in tqdm(test_files):
    img = io.imread("test224x224/" + file)
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    x_test.append(img)
    y_test.append(d[int(file.split('-')[0])])

x_train=np.asarray(x_train)
x_validation=np.asarray(x_dev)
x_test=np.asarray(x_test)

y_test=np.asarray(y_test)
y_validation=np.asarray(y_dev)
y_train=np.asarray(y_train)

model = create_cnn(poster_height, poster_width, poster_channels, genres)
model.summary()

model.load_weights('models/weights.hdf5')

# tensorboard = TrainValTensorBoard(log_dir=os.path.abspath('output'), histogram_freq=0, write_graph=True, write_images=True)
# checkpointer = ModelCheckpoint(filepath='models/weights.hdf5', verbose=1, save_best_only=True)

# # class_weights = {0: 16, 1: 9, 2: 8, 3: 24, 4: 6.4, 5: 16, 6: 1.45, 7: 4.57, 8: 4.57, 9: 3, 10: 10, 11: 5, 12: 8, 13: 3, 14: 16, 15: 4, 16: 16, 17: 1, 18: 3, 19: 14}

# history = model.fit(x_train, y_train,
#                     batch_size=batch_size,
#                     epochs=epochs,
#                     verbose=1,
#                     validation_data=(x_validation, y_validation),
#                     callbacks=[tensorboard, checkpointer])

# # history = model.fit(x_train, y_train,
# #                     batch_size=batch_size,
# #                     epochs=epochs,
# #                     verbose=1,
# #                     validation_data=(x_validation, y_validation))

# print('val below:')
# scores = model.evaluate(x_validation, y_validation, verbose=0)
# print(model.metrics_names)
# print(scores)
# print()
# preds = model.predict(x_validation)
# fin = (preds >= 0.5).astype(int)
# print(classification_report(y_validation, fin))
# print()
print('test below:')
test_scores = model.evaluate(x_test, y_test, verbose=0)
print(model.metrics_names)
print(test_scores)
print()
test_preds = model.predict(x_test)
test_fin = (test_preds >= 0.5).astype(int)
print(classification_report(y_test, test_fin))

keras.utils.plot_model(model, to_file=str(model_path + "/convolution.png"), show_shapes=True)
