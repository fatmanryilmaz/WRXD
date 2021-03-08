# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 21:30:26 2021

@author: fatmanur
"""

import numpy as np 
from numpy.linalg import inv
# import matplotlib.pyplot as plt

## Apply local dual window to the HSI data
## Window size should be entered as parameter. 
## w_in: inner window size
## w_out: outer window size
## gt is required to obtain target labels in the  same format with Detection scores.
## Returns Detection results and actual target labels to make easier AUC calculation.

def WRXD(data,gt,weights,w_in,w_out):   
    [R,C,L] = np.shape(data) 
    
    half_in = int((w_in-1)/2);
    half_out = int((w_out-1)/2);
    center = half_out;
    
    ## Number of pixels in the Outer Window Region (OWR).
    ## OWR: the area between two windows.
    n_pix_owr = w_out**2 - w_in**2
    
    # print("number of OWR pixels: ", n_pix_owr)
    
    ## Symmetric padding acording to size of window that will applied to the image.
    pad_img = np.pad(data,((half_out,half_out),(half_out,half_out),(0,0)),'symmetric')
    pad_w = np.pad(weights,((half_out,half_out),(half_out,half_out)),'symmetric')
    
    ## Shape of the padded image.
    [M,N,L] = np.shape(pad_img)
    # print("shape of padded image: ",[M,N,L])
    # print("shape of padded weights image: ",np.shape(pad_w))
    
    # plt.figure()
    # plt.imshow(pad_img[:,:,50])
    # plt.axis('off')
    # plt.figure()
    # plt.imshow(pad_w[:,:])
    # plt.axis('off')
    
    ## Total number of pixel vectors in an HSI.
    vector_num = R*C
    ## Keep local neighbors of each HSI pixel.
    HSI_pixels_local = np.zeros((vector_num,n_pix_owr,L))
    ## Keep local neighbors of weights that corresponds to above locals.
    weights_local = np.zeros((vector_num,n_pix_owr))
    ## Keep HSI data as array of pixel vectors.
    HSI_pixels = np.zeros((vector_num,L))
    ## Target label of the HSI pixels.
    ## PUT: Pixel Under Test
    ## (aka center pixel in the window.) 
    actualPUTLabels = np.zeros((vector_num,))
    
    for i in range (0,R):#(0,M - w_out + 1)
        for j in range (0,C):#(0,N - w_out + 1)
            ## Sliding window on each central pixel and get all vectors inside outer window.
            imgWindow = pad_img[i:w_out + i,j:w_out + j,:]
            ## Sliding window on weights corresponding to the local area of the image part above.
            wWindow = pad_w[i:w_out + i,j:w_out + j]
        
            ## Get OWR pixel vectors. 
            pix_owr = [[np.reshape(imgWindow[:,0:(half_out - half_in),:],(-1,L))],
                        [np.reshape(imgWindow[:,(center + half_in + 1):,:],(-1,L))],
                        [np.reshape(imgWindow[0:(half_out - half_in),(half_out - half_in):(center + half_in + 1),:],(-1,L))],
                        [np.reshape(imgWindow[(center + half_in + 1):,(half_out - half_in):(center + half_in + 1),:],(-1,L))]
                      ] 
            ## Get OWR weights. 
            w_owr = [[np.reshape(wWindow[:,0:(half_out - half_in)],(-1,))],
                      [np.reshape(wWindow[:,(center + half_in + 1):],(-1,))],
                      [np.reshape(wWindow[0:(half_out - half_in),(half_out - half_in):(center + half_in + 1)],(-1,))],
                      [np.reshape(wWindow[(center + half_in + 1):,(half_out - half_in):(center + half_in + 1)],(-1,))]
                    ] 
            HSI_pixels[i*R+j] = data[i,j,:]
            temp_HSI = np.zeros((1,n_pix_owr,L))
            temp_weights = np.zeros((1,n_pix_owr))
            count = 0
            for k in range (0,len(pix_owr)):         
                ## Keep OWR pixels for each PUT.   
                temp_HSI[:,count:count + np.shape(pix_owr[k])[1],:] = pix_owr[k]
                ## Keep OWR weights for each PUT.  
                temp_weights[:,count:count + np.shape(w_owr[k])[1]] = w_owr[k] 
                count += np.shape(pix_owr[k])[1]
        
            ## Keep all OWR pixels for all HSI data.
            HSI_pixels_local[i*R+j,:,:] = temp_HSI
            ## Keep all OWR weights for all HSI data.
            weights_local[i*R+j,:] = temp_weights
            actualPUTLabels[i*R+j] = gt[i,j]
    
    D = np.zeros((vector_num,))
    w = np.zeros((vector_num,n_pix_owr))
    ## Coeeficient c is used to avoid "singular matrix error" when matrix inverse.
    c = 0.001
    ## Reshape required because of transpose operation.
    ## 1D transpose gave incorrect result. 
    ## [L,] -> [L,1]
    HSI_pixels_local = np.reshape(HSI_pixels_local,(vector_num,n_pix_owr,L,1))
    HSI_pixels = np.reshape(HSI_pixels,(vector_num,L,1))
    for i in range (0,vector_num):
        w[i,:] = weights_local[i,:]/np.sum(weights_local[i,:])  
        mean = np.zeros((L,1))
        for j in range(0,n_pix_owr):
            mean += HSI_pixels_local[i,j,:]*w[i,j]
        c_temp = np.zeros((L,L))
        for k in range(0,n_pix_owr):
            V = HSI_pixels_local[i,k,:] - mean
            V_t = np.transpose(V)
            c_temp += w[i,k]*np.matmul(V,V_t)
        Cov = c_temp
        Cov_inv = inv(Cov + c*np.identity(L))
        VV = HSI_pixels[i,:] - mean
        VV_t = np.transpose(VV)
        D[i] = np.matmul(np.matmul(VV_t,Cov_inv),VV)
        
    return D,actualPUTLabels
    
