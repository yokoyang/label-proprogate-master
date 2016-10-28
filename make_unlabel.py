# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
import math
from sklearn import preprocessing
from label_propagation import labelPropagation
from sklearn.decomposition import PCA, IncrementalPCA


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
        # print line
        line.split(',')
        X.append(line.split(','))
    else:
        break
f.close()
data_setX = []
ipca = IncrementalPCA(n_components=25)
# print X
i = 0
while i < len(X):
    data_setX.append(map(float, X[i]))
    i += 1
# print data_setX
data_setX = np.array(data_setX)

f2 = open("2.txt", "r")
X2 = list()
while True:
    line = f2.readline()
    if line:
        pass  # do something here
        line = line.strip()
        # print line
        X2.append(line)
    else:
        break
f2.close()
i = 0
count2 = 0.0
compare_file = open('compare.txt', 'w')


while i < len(data_setX):
    if int(data_setX[i][37]) == X[i]:
        count2 += 1
    compare_file.writelines('%s\n' % int(data_setX[i][37]))
    i += 1
print count2/861

compare_file.close()
# print X[1]
# 39 dimensions for now
Y = data_setX[:12, 37:]
X = data_setX[:, :37]
X_ipca = ipca.fit_transform(X)
X_scaled = preprocessing.scale(X_ipca)
# print X
# print Y

# Convert char to int in the list
# print X
target = []
for temp in Y:
    # print temp
    target.append(int(temp[0]))

print target
Mat_Label = X_scaled[:12, :]
Mat_Unlabel = X_scaled[12:, :]
for unlab_conv in Mat_Unlabel:
    unlab_conv = map(int, unlab_conv)
print Mat_Unlabel
# print len(Mat_Unlabel)
# print Mat_Label
# print len(Mat_Label)
unlabel_data_labels = labelPropagation(Mat_Label, Mat_Unlabel, target, kernel_type='knn', knn_num_neighbors=10,
                                       max_iter=400)


