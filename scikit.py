# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
import math
from sklearn import preprocessing
from sklearn.semi_supervised import label_propagation

rng = np.random.RandomState(0)


# show
def show(Mat_Label, labels, Mat_Unlabel, unlabel_data_labels):
    for i in range(Mat_Label.shape[0]):
        if int(labels[i]) == 0:
            plt.plot(Mat_Label[i, 0], Mat_Label[i, 1], 'Dr')
        elif int(labels[i]) == 1:
            plt.plot(Mat_Label[i, 0], Mat_Label[i, 1], 'Db')
        else:
            plt.plot(Mat_Label[i, 0], Mat_Label[i, 1], 'Dy')

    for i in range(Mat_Unlabel.shape[0]):
        if int(unlabel_data_labels[i]) == 0:
            plt.plot(Mat_Unlabel[i, 0], Mat_Unlabel[i, 1], 'or')
        elif int(unlabel_data_labels[i]) == 1:
            plt.plot(Mat_Unlabel[i, 0], Mat_Unlabel[i, 1], 'ob')
        else:
            plt.plot(Mat_Unlabel[i, 0], Mat_Unlabel[i, 1], 'oy')

    plt.xlabel('X1');
    plt.ylabel('X2')
    plt.xlim(0.0, 12.)
    plt.ylim(0.0, 12.)
    plt.show()


rng = np.random.RandomState(0)

f = open("nasa_train.txt", "r")
X = list()
while True:
    line = f.readline()
    if line:
        pass  # do something here
        line = line.strip()
        line.split(',')
        X.append(line.split(','))
    else:
        break
f.close()
data_setX = []
i = 0
while i < len(X):
    data_setX.append(map(float, X[i]))
    i += 1
data_setX = np.array(data_setX)

# 39 dimensions for now
y = data_setX[:, 37:]
X = data_setX[:, :37]
X = preprocessing.scale(X)
print X
# print y
target = []
for temp in y:
    # print temp
    target.append(int(temp[0]))
print target
y_30 = np.copy(target)
y_30[rng.rand(len(target)) < 0.7] = -1
label_spread = label_propagation.LabelSpreading(kernel='knn', gamma=0.25, alpha=1.0)
ls = label_spread.fit(X, y_30)
print ls.transduction_


# ls30 = (label_propagation.LabelSpreading().fit(X, target),
#         target)
# print ls30[1]
# count = 0.0
# i = 0
# while i < len(ls30[1]):
#     if target[i] == ls30[1][i]:
#         count += 1
#     i += 1
# print count / i
# (clf, y_train) = ls30
# rbf_svc = (svm.SVC(kernel='rbf').fit(X, target), target)
# print target.__len__()
# print ls30[1].__len__()
# count = 0.0
# i = 0
# while i < len(result_test):
#     if unlabel_data_labels[i] == result_test[i]:
#         count += 1
#     i += 1
#
# print count / 688
