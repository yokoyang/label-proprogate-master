# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
import sys, warnings, getopt, scipy, datetime, math, random;
from label_propagation import labelPropagation

# main function
if __name__ == "__main__":
    f = open("nasa_input.txt", "r")
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
    # print X
    i = 0
    while i < len(X):
        data_setX.append(map(float, X[i]))
        i += 1
    # print data_setX
    data_setX = np.array(data_setX)
    # print X[1]
    # 39 dimensions for now
    Y = data_setX[:, 37:]
    X = data_setX[:, :37]
    # print X
    # print Y

    # Convert char to int in the list
    # print X
    target = []
    for temp in Y:
        print temp
        target.append(temp[0])
    # print target

    # SVM start
    start = datetime.datetime.now()

    h = .02  # step size in the mesh

    # SVM and fit data
    # Maybe need to scale the original data

    C = 1.0  # SVM regularization parameter
    svc = svm.SVC(kernel='linear', C=C).fit(X, target)
    # rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, target)
    # poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, target)
    # lin_svc = svm.LinearSVC(C=C).fit(X, target)

    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    temp = [
        [1, 7, 0, 2, 0, 10, 4, 0.4, 4, 2.5, 1, 0.25, 11, 3, 0.67, 8, 1, 12.44, 13.22, 2175.26, 0.05, 35, 0.08, 120.85,
         164.52, 0.75, 3, 5, 9, 0.33, 14, 21, 9, 17, 12, 20, 10]]
    temp = np.array(temp)
    res = svc.predict(temp)
    print res
    # title for the plots
    # titles = ['SVC with linear kernel',
    #           'LinearSVC (linear kernel)',
    #           'SVC with RBF kernel',
    #           'SVC with polynomial (degree 3) kernel']
    end = datetime.datetime.now()
    runtime = end - start
    fout_time = open('inf.txt', 'w')
    fout_time.writelines('result: %s\n' % res)
    fout_time.writelines('running time: %s\n' % runtime)
    fout_time.close()
