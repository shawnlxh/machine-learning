#user_size 707
#item_size 8533

import numpy as np
import random
import math
import json

data = []
training_set = []
raw_data = open('Downloads/MachineLearning-master/Tmall/user_cart.json', 'r')
lines = raw_data.readlines()
count = 0
for line in lines:
    line1 = json.loads(line)
    n = 0
    for num in line1:
        line1[n] = int(line1[n])
        n += 1
    if n >= 10:
        data.append(line1)
        training_set.append(line1[0:int(n*0.8)])
        count += 1

u = np.random.randn(10, 10)*0.5
r = np.random.randn(10, 10)*0.5
i = np.random.randn(8533, 10)*0.5 

learning_rate = 0.01
lamda_pos = 0.001
lamda = 0.001
hidden_size = 10

def f(x):    #sigmoid
	output = 1/(1+np.exp(-x))
	return output
	
neglist = []
for n in range(707):
    line = []
    for num in training_set[n]:
        j = random.randint(0, 8532)
        while j == num:
            j = random.randint(0, 8532)
        line.append(j)
    neglist.append(line)
    
def train():
    global u
    global r
    global i
    for n in range(707):
        count = 0
        for num in training_set[n]:
            count += 1
        
        # t = 1
        h0 = np.zeros((1, 10))
        h0 = h0[0]
        i1 = i[training_set[n][0]-1]
        a1 = np.dot(i1, u) + np.dot(h0, r)
        h1 = np.zeros((1, 10))
        h1 = h1[0]
        for m in range(10):
            h1[m] = f(a1[m])
        i2 = i[training_set[n][1]-1]
        ij = i[neglist[n][1]-1]
        x = np.dot(h1, i2.T) - np.dot(h1, ij.T)
        xh1 = i2 - ij
        mid = 1/ (1 + np.exp(x))
        i[training_set[n][1]-1] += learning_rate * (mid * h1 - lamda * i2)
        i[neglist[n][1]] += learning_rate * (mid * (-h1) - lamda * ij)
        ha1 = np.zeros((1, 10))
        ha1 = ha1[0]
        xu 
        xu = (1 / (1+np.exp(x))) * xu - lamda * u
        xr = (1 / (1+np.exp(x))) * xr - lamda * r
        u += learning_rate * xu
        r += learning_rate * xr
            
        # t = 2
        i2 = i[training_set[n][1]-1]
        a2 = np.dot(i2, u) + np.dot(h1, r)
        h2 = np.zeros((1, 10))
        h2 = h2[0]
        for m in range(10):
            h2[m]= f(a2[m])
        i3 = i[training_set[n][2]-1]
        ij = i[neglist[n][2]-1]
        x = np.dot(h2, i3.T) - np.dot(h2, ij.T)
        xh2 = i3 - ij
        mid = 1/ (1 + np.exp(x))
        i[training_set[n][2]-1] += learning_rate * (mid * h2 - lamda * i3)
        i[neglist[n][2]] += learning_rate * (mid * (-h2) - lamda * ij)
        ha2 = np.zeros((1, 10))
        ha2 = ha2[0]
        xu = np.zeros((10, 10))
        xr = np.zeros((10, 10))
        for m in range(10):
            ha2[m] = h2[m] * (1 - h2[m])
        for m in range(10):
            xu[m] = i2[m] * ha2 * xh2
        for m in range(10):
            xr[m] = h1[m] * ha2 * xh2
        xu = (1 / (1+np.exp(x))) * xu - lamda * u
        xr = (1 / (1+np.exp(x))) * xr - lamda * r
        u += learning_rate * xu
        r += learning_rate * xr
        
        xh1 = np.dot(ha2 * xh2, r.T) 
        for m in range(10):
            xu[m] = i1[m] * ha1 * xh1
        for m in range(10):
            xr[m] = h0[m] * ha1 * xh1
        xu = (1 / (1+np.exp(x))) * xu - lamda * u
        xr = (1 / (1+np.exp(x))) * xr - lamda * r
        u += learning_rate * xu
        r += learning_rate * xr
        
        # t >= 3
        t = 2
        it2 = i1
        it1 = i2
        ht3 = h0
        ht2 = h1
        ht1 = h2
        hat1 = ha2
        hat2 = ha1
        ht = np.zeros((1, 10))
        ht = ht[0]
        while t < (count - 1):
            it = i[training_set[n][t]-1]
            at = np.dot(it, u) + np.dot(ht1, r)
            for m in range(10):
                ht[m]= f(at[m])
            itp = i[training_set[n][t+1]-1]
            ij = i[neglist[n][t+1]-1]
            x = np.dot(ht, itp.T) - np.dot(ht, ij.T)
            xht = itp - ij
            mid = 1/ (1 + np.exp(x))
            i[training_set[n][t+1]-1] += learning_rate * (mid * ht - lamda * itp)
            i[neglist[n][t+1]] += learning_rate * (mid * (-ht) - lamda * ij)
            hat = np.zeros((1, 10))
            hat = hat[0]
            xu = np.zeros((10, 10))
            xr = np.zeros((10, 10))
            for m in range(10):
                hat[m] = ht[m] * (1 - ht[m])
            for m in range(10):
                xu[m] = it[m] * hat * xht
            for m in range(10):
                xr[m] = ht1[m] * hat * xht
            xu = (1 / (1+np.exp(x))) * xu - lamda * u
            xr = (1 / (1+np.exp(x))) * xr - lamda * r
            u += learning_rate * xu
            r += learning_rate * xr
            
            
            xht1 = np.dot(hat * xht, r.T)
            for m in range(10):
                xu[m] = it1[m] * hat1 * xht1
            for m in range(10):
                xr[m] = ht2[m] * hat1 * xht1
            xu = (1 / (1+np.exp(x))) * xu - lamda * u
            xr = (1 / (1+np.exp(x))) * xr - lamda * r
            u += learning_rate * xu
            r += learning_rate * xr
            
            xht2 = np.dot(hat * xht1, r.T)
            for m in range(10):
                xu[m] = it2[m] * hat2 * xht2
            for m in range(10):
                xr[m] = ht3[m] * hat2 * xht2
            xu = (1 / (1+np.exp(x))) * xu - lamda * u
            xr = (1 / (1+np.exp(x))) * xr - lamda * r
            u += learning_rate * xu
            r += learning_rate * xr
            
            it2 = it1
            it1 = it
            ht3 = ht2
            ht2 = ht1
            ht1 = ht
            hat1 = hat
            hat2 = hat1
            
            t += 1
            
def predict():
    predict_count = 0
    predict_sum = 0
    n = 0
    while n < 707:
        count = 0
        for num in data[n]:
            count += 1
        ht1 = np.zeros((1, 10))
        if count > 1:
            for m in range(int(count*0.8)):
                it = i[data[n][m]-1]
                at = np.dot(ht1,r) + np.dot(it,u)
                ht = np.zeros((1, 10))
                for w in range(10):
                    ht[0][w] = f(at[0][w])
                ht1 = ht
            while m < (count - 1):
                temp = np.dot(ht1, i.T)
                sort_list = np.argsort(-temp[0])
                for k in range(10):
                    if (sort_list[k]+1) == data[n][m+1]:
                        predict_count += 1
                predict_sum += 1
                it = i[data[n][m+1]-1]
                at = np.dot(ht1,r) + np.dot(it,u)
                ht = np.zeros((1, 10))
                for w in range(10):
                    ht[0][w] = f(at[0][w])
                ht1 = ht
                m += 1
        n += 1
    return predict_count, predict_sum
    
for n in range(200):
    print n
    train()
    predict_count, predict_sum = predict()
    result = float(predict_count) / predict_sum   
    print result
 
