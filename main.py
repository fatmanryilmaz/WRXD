# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 23:22:36 2021

@author: fatmanur
"""

import numpy as np 
from scipy.io import loadmat
from GRXD import GRXD
from WRXD import WRXD
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

data = loadmat(r"sandiego_cut1")
data = data['Sandiego_cut']
gt = loadmat(r"sandiego_cut1_gt")
gt = gt['labels']

[R,C,L] = np.shape(data)

## Weights.
w = GRXD(data)

## Dual window sizes.
w_in = 5
w_out = 7

## D: detection scores
## T: target labels.
(D,T) = WRXD(data,gt,w,w_in,w_out)

AUC = roc_auc_score(T,D)

## Analyze and visualize scores.
fpr, tpr, th = roc_curve(T,D)
## Plot the roc curve.
plt.plot(fpr, tpr, marker='.', label='AUC = {0:.3f}'.format(AUC))
## Axis labels.
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
## Show the legend.
plt.legend()
## Show the plot.
fig_roc = plt.gcf()

# print('Logistic: ROC AUC=%.3f' % (AUC))
            
## Find optimum th value.
## This is an imperfect personal method, can be improved.
## Tried to find the previous point that ROC Curve got worse. 
## False alarm rate increases as increase of err.
err = 0.003
for i in range(0,len(th)):
    err_ = abs(fpr[i+1] - fpr[i])
    if err_ >= err:
        opt_th = th[i]
        print(opt_th)
        break

vector_num = len(D)
detectionLabel = np.zeros((np.shape(T)))
for i in range (0,vector_num):
    if (D[i] >= opt_th):
        detectionLabel[i] = 1

## Reproduce anomaly gt acording to the Detection Scores.
new_gt = np.zeros((R,C))
old_gt = np.zeros((R,C))
for i in range (0,R):
    for j in range (0,C):
        new_gt[i,j] = detectionLabel[i*C+j]
        old_gt[i,j] = T[i*C+j]

plt.figure()
plt.imshow(data[:,:,50])
plt.axis('off')
plt.figure()
plt.imshow(gt[:,:])
plt.axis('off')
plt.figure()
fig_new_gt = plt.gcf()
plt.imshow(new_gt[:,:])
plt.axis('off')