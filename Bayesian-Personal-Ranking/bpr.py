# -*- coding: utf-8 -*-
import numpy as np
import random
from sklearn.metrics import roc_auc_score

file1 = open('desktop/train.txt','r')
file2 = open('desktop/test.txt','r')

train_data = np.zeros((943,1682))
test_data = np.zeros((943,1682))
user_p = np.random.randn(943,10)*0.5
item_q = np.random.randn(1682,10)*0.5
lamda = 0.01
learning_rate = 0.01

for line in file1:
    line = line.split('\t')
    user = int(line[0])
    item = int(line[1])
    score = int(line[2])
    train_data[user-1][item-1] = score
    
for line in file2:
    line = line.split('\t')
    user = int(line[0])
    item = int(line[1])
    score = int(line[2])
    test_data[user-1][item-1] = score

def train():
    global user_p   #对全局变量进行修改
    global item_q
    for u in range(943):
        for i in range(1682):
            if train_data[u][i] > 3:
                j = random.randint(0, 1681)
                while train_data[u][j] > 3:
                    j = random.randint(0, 1681)
                xuij = np.dot(user_p[u], item_q[i].T) - np.dot(user_p[u], item_q[j].T)
                mid = 1.0 / (np.exp(xuij) + 1)
                temp = user_p[u]
                user_p[u] += -learning_rate * (-mid * (item_q[i]- item_q[j]) + lamda * user_p[u])
                item_q[i] += -learning_rate * (-mid * temp + lamda * item_q[i])
                item_q[j] += -learning_rate * (-mid * (-temp) + lamda * item_q[j])
                
def predict(user_matrix, item_matrix):
	predict_matrix = np.zeros(1586126)
	for i in range(943):
		for j in range(1682):
			predict_matrix[i*943+j] = np.dot(user_matrix[i],item_matrix[j].T)
	return predict_matrix

for i in range(200):
	print "iter %i"%i
	train()

predict_matrix = predict(user_p, item_q)
test = np.zeros(1586126)
for i in range(943):
    for j in range(1682):
        if test_data[i][j] > 3:
            test[i*943+j] = 1
        else:
            test[i*943+j] = 0
            
for i in range(1586126):
    test[i] = int(test[i])
            
a =  roc_auc_score(test, predict_matrix)
print a