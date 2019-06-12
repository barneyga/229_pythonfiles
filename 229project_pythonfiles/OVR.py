# based off of example provided on sci-kit website

from skmultilearn.adapt import MLkNN
from skmultilearn.problem_transform import BinaryRelevance
import numpy as np
import os
import csv
from skimage import io
from sklearn.model_selection import train_test_split
from sklearn.metrics import hamming_loss
import pickle
import time
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier



start_time = time.time()

r = csv.reader(open('milestone_labels.csv', 'r', encoding='utf8'))
x_train, y_train, x_dev, y_dev, x_test, y_test = [],[],[],[],[],[]
d = {}

# https://github.com/taoxinyi/multi-label-natural-scene-classification/blob/2784b452438d849c75f91c55de55ceeb8f6821b4/mlknn-rgb.py
# it has an MIT license so we gucci, maybe we can change it later so we don't have to cite.
for row in r:
    try:
        d[int(row[0].split('-')[0])] = list(map(int, row[1:]))
    except:
        pass

print(len(d))
print(d[9999])

train_files, dev_files, test_files= os.listdir('train/'), os.listdir('dev/'), os.listdir('test/')
# likely_k = int(np.ceil(np.sqrt(len(train_files)//100))) * 3
train_half = train_files[:len(train_files)//10]
dev_half = dev_files[:len(dev_files)//10]
test_half = test_files[:len(test_files)//10]

for file in tqdm(train_half):
#     x_train.append(io.imread("train/" + file).flatten())
    x_train.append(io.imread("train100x150/" + file).flatten())
    y_train.append(d[int(file.split('-')[0])])
for file in tqdm(dev_half):
#     x_dev.append(io.imread("dev/" + file).flatten())
    x_dev.append(io.imread("dev100x150/" + file).flatten())
    y_dev.append(d[int(file.split('-')[0])])
for file in tqdm(test_half):
#     x_test.append(io.imread("test/" + file).flatten())
    x_test.append(io.imread("test100x150/" + file).flatten())
    y_test.append(d[int(file.split('-')[0])])

x_train=np.asarray(x_train)
x_dev=np.asarray(x_dev)
x_test=np.asarray(x_test)

y_test=np.asarray(y_test)
y_dev=np.asarray(y_dev)
y_train=np.asarray(y_train)

# TODO: tune hyperparameters
# note that this unpickling is only for the most previously pickled (k=5 right now)
# pickle_file = open('MLkNN_milestone.pkl', 'rb')
# clf = pickle.load(pickle_file)

# 30 is currently the best tested k amount.
l = [40, 50, 100, 200, 280]
# l = [200]
# l = [likely_k]
# l = [70, 80, 90, 100, 500, 1000, 2000, 3000, 4000, 5600]
best_clf = None
lowest_hl = float('inf')
best_k = float('inf')
for k in l:
    print(25*'=')
    print('k = ' + str(k))
    clf = BinaryRelevance(KNeighborsClassifier(k))

    # train
    clf.fit(x_train, y_train)

    # predict
    predictions = clf.predict(x_dev)

    predictions = predictions.todense()
    print('all match:', np.sum(np.all(predictions == y_dev, axis=1)) / len(y_dev))
    print('at least one match:', (np.sum(np.all(predictions - y_dev <= 0, axis=1))-np.sum(np.all(predictions== 0, axis=1))) / len(y_dev))
    print('binary :', np.mean(predictions == y_dev))
    hl = hamming_loss(y_dev, predictions)
    print('Hamming Loss:', hamming_loss(y_dev, predictions))
    if hl < lowest_hl:
        lowest_hl = hl
        best_clf = clf
        best_k = k
    

# import sys
# np.set_printoptions(threshold=sys.maxsize)

print(25*'=')
predictions = best_clf.predict(x_test)

print('best k = ' + str(best_k))
predictions = predictions.todense()
# print(predictions != y_test)
print('all match:', np.sum(np.all(predictions == y_test, axis=1)) / len(y_test))
print('at least one match:', (np.sum(np.all(predictions - y_test <= 0, axis=1))-np.sum(np.all(predictions== 0, axis=1))) / len(y_test))
print('binary:', np.mean(predictions == y_test))
print('Hamming Loss:', hamming_loss(y_test, predictions))
f = open("MLkNN.pkl","wb")
pickle.dump(best_clf, f)
f.close()

print("--- %s seconds ---" % (time.time() - start_time))