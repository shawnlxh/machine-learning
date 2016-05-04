# -*- coding: utf-8 -*-

# written by shawnlxh
# use movieLens 100K
import numpy as np
import math

file1 = open('desktop/train.txt','r')
file2 = open('desktop/test.txt','r')

train_data = np.zeros((943,1682))
test_data = np.zeros((943,1682))
user_p = np.random.randn(943,10)*0.5
item_q = np.random.randn(1682,10)*0.5
lamda = 0.02
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

def avg(matrix):
    count = 0
    total = 0
    for i in range(943):
        for j in range(1682):
            if matrix[i][j] != 0:
                count += 1
                total += matrix[i][j]
    avg_total = total / count
    return avg_total
    
def user_bias(matrix, avg):
    bias = []
    for i in range(943):
        count = 0
        total = 0
        for j in range(1682):
            if matrix[i][j] != 0:
                count += 1
                total += matrix[i][j] - avg
        if count == 0:
            bias.append(count)
        else:
            bias.append(total/count)
    return bias
    
def item_bias(matrix, avg):
    bias = []
    for j in range(1682):
        count = 0
        total = 0
        for i in range(943):
            if matrix[i][j] != 0:
                count += 1
                total += matrix[i][j] - avg
        if count == 0:
            bias.append(count)
        else:
            bias.append(total/count)
    return bias

avg_train = avg(train_data)   #average of the training data
    
def train():
    global user_p   #change the global variables
    global item_q
    bias_u1 = user_bias(train_data, avg_train)
    bias_i1 = item_bias(train_data, avg_train)
    for u in range(943):
        for i in range(1682):
            if train_data[u][i] != 0:
                rui = avg_train + bias_u1[u] + bias_i1[i] + np.dot(user_p[u], item_q[i].T)
                eui = train_data[u][i] - rui
                bias_u1[u] += learning_rate * (eui - lamda * bias_u1[u])
                bias_i1[i] += learning_rate * (eui - lamda * bias_i1[i])
                temp = user_p[u]
                user_p[u] += learning_rate * (eui * item_q[i] - lamda * user_p[u])
                item_q[i] += learning_rate * (eui * temp - lamda * item_q[i])
                
def test(test_data, user_p, item_q):
    avg_test = avg(test_data)
    bias_u2 = user_bias(test_data, avg_test)
    bias_i2 = item_bias(test_data,avg_test) 
    count = 0
    total = 0
    for u in range(943):
        for i in range(1682):
            if test_data[u][i] != 0:
               rui = avg_test + bias_u2[u] + bias_i2[i] + np.dot(user_p[u], item_q[i].T)
               eui = test_data[u][i] - rui
               count += 1
               total += eui * eui
    loss = math.sqrt(total / count)
    return loss
    
for i in range(50):
    print i
    train()
    final = test(test_data, user_p, item_q)
    print final
    

                