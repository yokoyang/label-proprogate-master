# coding=utf-8
# reading the data set
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
import math
from label_propagation import labelPropagation

f = open("nasa_input.txt", "r")
X = list()
while True:
    line = f.readline()
    if line:
        pass  # do something here
        line = line.strip()
        # print line
        X.append(line)
    else:
        break
f.close()
print X
for i in X:
    print i.count(',')
print len(X)
