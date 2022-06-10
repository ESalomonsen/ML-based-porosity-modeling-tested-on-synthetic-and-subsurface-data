# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 11:08:25 2022

@author: Eier
"""
import numpy as np

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsRegressor

################## random data ##################
mean = [0, 0]
cov = [[1, 20], [20, 100]]

data = np.random.multivariate_normal(mean, cov, 50).T

data = np.sort(data, axis=- 1)

x, y = data
######################################################
################## k = 1 ##################
vari = KNeighborsRegressor(n_neighbors=1)

vari.fit(x.reshape(-1, 1), y)
vari_out = vari.predict(x.reshape(-1, 1))
################## k = 1 ##################

################## k = 50 ##################

bi = KNeighborsRegressor(n_neighbors=len(x))

bi.fit(x.reshape(-1, 1), y)
bi_out = bi.predict(x.reshape(-1, 1))
################## k = 50 ##################

################## k = 10 ##################

middle = KNeighborsRegressor(n_neighbors=10)

middle.fit(x.reshape(-1, 1), y)
middle_out = middle.predict(x.reshape(-1, 1))
################## k = 10 ##################

######################################################
# plt.plot(x, y, 'x')

f, axs = plt.subplots(1,3,figsize=(20,10))

plt.subplot(131)

plt.plot(x, y, 'x')
plt.plot(x,vari_out)
plt.xlabel('x')
plt.ylabel('y')
plt.title('k = 1')

plt.subplot(132)

plt.plot(x, y, 'x')
plt.plot(x,bi_out)
plt.xlabel('x')
plt.ylabel('y')
plt.title('k = 50')

plt.subplot(133)

plt.plot(x, y, 'x')
plt.plot(x,middle_out)
plt.xlabel('x')
plt.ylabel('y')
plt.title('k = 10')

plt.show()