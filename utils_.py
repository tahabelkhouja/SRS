# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 12:54:10 2019

@author: BkTaha
"""
import os
import csv
import json
import copy
from scipy.io import arff
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import pickle as pkl
import numpy.linalg as lg
import scipy.stats as st
import pandas as pd
from scipy.stats import random_correlation

#%%Statistical functions

def m_features(s):
    """
    :s: Signal 1x1xSxC
    :return: Mean, median and mode of a signal on each channel
    """
    channel_nb = s.shape[-1]
    l = s.shape[2]
    #Stack init using channel 0
    
    #Mean
    m = tf.reduce_mean(s[0,0,:,0])
    
    #Median and Interquartile range
    cut_points = tfp.stats.quantiles(s[0,0,:,0], num_quantiles=4, 
                                     interpolation='midpoint')
    median = cut_points[2]
    Q1 = cut_points[1]
    Q3 = cut_points[3]
    quart_range = Q3-Q1 
        
    #Mode 
    nbins = 5000
    range_mod = [tf.reduce_min(s[0,0,:,0]), tf.reduce_max(s[0,0,:,0])]
    hist_mod = tf.histogram_fixed_width_bins(s[0,0,:,0], range_mod, nbins=nbins)
    a,b, count = tf.unique_with_counts(tf.reshape(hist_mod, [-1]))
    idx_mod = a[tf.argmax(count)]
    alpha = tf.subtract(tf.reduce_max(s[0,0,:,0]), tf.reduce_min(s[0,0,:,0]))
    denum = tf.multiply(alpha, tf.dtypes.cast(tf.add(tf.multiply(2, idx_mod), 1), tf.float64))
    mode_ = tf.divide(denum, tf.dtypes.cast(tf.multiply(2, nbins), tf.float64))
    mode = tf.add(mode_, tf.reduce_min(s[0,0,:,0]))
    
    #Std deviation
    std_dev = tfp.stats.stddev(s[0,0,:,0])
    
    #Interquartile range
    cut_points = tfp.stats.quantiles(s[0,0,:,0], num_quantiles=4)
    quart_range = tf.subtract(Q3, Q1)   
    
    #Skewness
    skw1 = tf.add(Q1, m)
    skw2 = tf.subtract(skw1, tf.multiply(2.0, m))
    skew = tf.divide(skw2, quart_range)
    
    #Kurtosis
    kurt_1 = tf.reduce_sum(tf.pow(tf.subtract(s[0,0,:,0], m), 4))
    kurt_2 = tf.multiply(tf.divide(1.0, l), kurt_1)
    kurt_3 = tf.divide(kurt_2, tf.pow(std_dev, 4))
    kurt = tf.subtract(kurt_3, 3)
    
    #RMS
    rms_1 = tf.reduce_sum(tf.pow(s[0,0,:,0], 2))
    rms_2 = tf.multiply(tf.divide(1.0, l), rms_1)
    rms = tf.sqrt(rms_2)
    
    #Correlation sup than a treshold for finding patterns
    corr_th = 0.8
    corr = tfp.stats.auto_correlation(s[0,0,:,0])
    corr_sup = tf.divide(tf.reduce_sum(tf.cast(tf.greater_equal(corr, corr_th), 
                                               tf.float64)), l)
    
    m_res = tf.reshape(tf.stack([m, std_dev, median, mode, quart_range, skew, kurt, rms, corr_sup]), (9,1))
    #Multiple channel case
    if channel_nb > 1:
        for c in range(1, channel_nb):
            #Mean
            m = tf.reduce_mean(s[0,0,:,c])
            
            #Median and Interquartile range
            cut_points = tfp.stats.quantiles(s[0,0,:,c], num_quantiles=4, 
                                             interpolation='midpoint')
            median = cut_points[2]
            Q1 = cut_points[1]
            Q3 = cut_points[3]
            quart_range = Q3-Q1 
                
            #Mode 
            nbins = 5000
            range_mod = [tf.reduce_min(s[0,0,:,c]), tf.reduce_max(s[0,0,:,c])]
            hist_mod = tf.histogram_fixed_width_bins(s[0,0,:,c], range_mod, nbins=nbins)
            a,b, count = tf.unique_with_counts(tf.reshape(hist_mod, [-1]))
            idx_mod = a[tf.argmax(count)]
            alpha = tf.subtract(tf.reduce_max(s[0,0,:,c]), tf.reduce_min(s[0,0,:,c]))
            denum = tf.multiply(alpha, tf.dtypes.cast(tf.add(tf.multiply(2, idx_mod), 1), tf.float64))
            mode_ = tf.divide(denum, tf.dtypes.cast(tf.multiply(2, nbins), tf.float64))
            mode = tf.add(mode_, tf.reduce_min(s[0,0,:,c]))
            
            #Std deviation
            std_dev = tfp.stats.stddev(s[0,0,:,c])
            
            #Interquartile range
            cut_points = tfp.stats.quantiles(s[0,0,:,c], num_quantiles=4)
            quart_range = tf.subtract(Q3, Q1)   
            
            #Skewness
            skw1 = tf.add(Q1, m)
            skw2 = tf.subtract(skw1, tf.multiply(2.0, m))
            skew = tf.divide(skw2, quart_range)
            
            #Kurtosis
            kurt_1 = tf.reduce_sum(tf.pow(tf.subtract(s[0,0,:,c], m), 4))
            kurt_2 = tf.multiply(tf.divide(1.0, l), kurt_1)
            kurt_3 = tf.divide(kurt_2, tf.pow(std_dev, 4))
            kurt = tf.subtract(kurt_3, 3)
            
            #RMS
            rms_1 = tf.reduce_sum(tf.pow(s[0,0,:,c], 2))
            rms_2 = tf.multiply(tf.divide(1.0, l), rms_1)
            rms = tf.sqrt(rms_2)
            
            #Correlation sup than a treshold for finding patterns
            corr_th = 0.8
            corr = tfp.stats.auto_correlation(s[0,0,:,c])
            corr_sup = tf.divide(tf.reduce_sum(tf.cast(tf.greater_equal(corr, corr_th), 
                                                       tf.float64)), l)
            
            m_res = tf.concat([m_res, tf.reshape([m, std_dev, median, mode,
                                  quart_range, skew, kurt, rms, corr_sup],
                                    (9,-1))], axis=1)
    m_res_dict={'mean':0,
                'std_dev':1,
                'median':2,
                'mode':3,
                'interquart':4,
                'skew':5,
                'kurt':6,
                'rms':7,
                'correl':8}
    
    return m_res_dict, m_res

def mode(s):
    nbins = 5000
    range_mod = [tf.reduce_min(s[:,:,:,0]), tf.reduce_max(s[:,:,:,0])]
    hist_mod = tf.histogram_fixed_width_bins(s[:,:,:,0], range_mod, nbins=nbins)
    a,b, count = tf.unique_with_counts(tf.reshape(hist_mod, [-1]))
    idx_mod = a[tf.argmax(count)]
    alpha = tf.subtract(tf.reduce_max(s[:,:,:,0]), tf.reduce_min(s[:,:,:,0]))
    denum = tf.multiply(alpha, tf.dtypes.cast(tf.add(tf.multiply(2, idx_mod), 1), tf.float64))
    mode_ = tf.divide(denum, tf.dtypes.cast(tf.multiply(2, nbins), tf.float64))
    mode = tf.add(mode_, tf.reduce_min(s[:,:,:,0]))
    return mode


def ft_stat_score(x1, x2):
    _, s1 = m_features(x1)
    _, s2 = m_features(x2)
    return np.linalg.norm(s2-s1, ord=2)

def multidim_norm(X, norm=2):
    if len(X.shape) > 2:
        raise ValueError("X shape is > 2")
    return np.sum(np.linalg.norm(X, ord=norm, axis=1))
#%% Array/Tensor functions
def nan_eliminate(l, value=0):
    for ind in np.argwhere(np.isfinite(l)==False):
        l[ind] = value
    return l
            
def label_from_one_hot_conv(l):
    ll = l.shape[0]
    labels = np.ndarray((ll)).astype(np.int64)
    for ind in range(ll):
        labels[ind] = np.int64(np.argmax(l[ind]))
    return labels
    
    
def elem_of(sess, a, b):
    """
    :return: Element of Tensor a at the index b.
    """
    R = tf.gather(a, b, name="ThisGather")
    return sess.run(R)


def normalize(x, m, s): return (x-m)/s

def feed_to_array2(sess, T, ll, c):
    """
    :return: Array of shape (ll, c) manually fed from T.
    """
    arr = np.ndarray((ll, c ))
    for ind in range(ll):
        aux = elem_of(sess, T, ind)
        arr[ind, :] = aux
    return arr

def input_standardize(temp):
    x = temp / (np.max(temp) - np.min(temp))
    return x

def voting_array(ar):
    """
    :param ar: Array of size NxL.
    :return: Array of size N of most occurent element in each sub_array of
            size L
    """
    ar = ar.astype(int)
    vote = np.inf*np.ones(ar.shape[0])
    for ind in range(ar.shape[0]):
        counts = np.bincount(ar[ind, :])
        vote[ind] = np.argmax(counts)
    return vote

def signal_dists(s1, s2, channels, norm=2):
    """
    :param sig1, sig2: Signals subjects to the distance function of 
                    size Nx1xSxC (N: Number of signals
                                  S: Segment size
                                  C: Number of channels).
    :param channels: Channels on which distance is disregarded.
    :param norm: Norm to be used (2 or np.inf)
    :return: Array of distance of signals of size NxC
    """
    if s1.shape != s2.shape:
            raise ValueError("Both inputs should be of the same size")
    dist = np.inf * np.ones((s1.shape[0], s1.shape[-1]))
    for n in range(s1.shape[0]):
        for c in range(s1.shape[-1]):
            nrm = lg.norm(s1[n,:,:,c].reshape((-1))-s2[n,:,:,c].reshape((-1)),
                            ord=norm)
            dist[n, c] = nrm if np.isfinite(nrm) else 0
    dist = np.delete(dist, np.argwhere(dist[:,1]==np.inf)[:,0], axis=0)
    return dist
    
def l_array_complem(a, idx):
    """
    :return: Array of elements of a but those in idx.
    """
    res = np.zeros(((a.shape[0]-idx.shape[0]+1)))
    res_i = 0
    for i in range(a.shape[0]):
        if not (i in idx):
            res[res_i] =  a[i]
            res_i += 1
    return res
    
def array_complem(a, idx, SEG_SIZE, CHANNEL_NB):
    """
    :return: Array of elements of a but those in idx.
    """
    res = np.zeros(((a.shape[0]-idx.shape[0]+1),1,SEG_SIZE, CHANNEL_NB))
    res_i = 0
    for i in range(a.shape[0]):
        if not (i in idx):
            res[res_i] =  a[i]
            res_i += 1
    return res 

  
def signal_alteration_rate(sig1, sig2):
    """
    :param sig1: Original signal Nx1xSxC (N: Number of signals,
                                          S: Segment size,
                                          C: Number of channels).
    :param sig2: Signal to be compared to the first one.
    :return eps: Mean difference value for each channel.
    :return tol: Total number of points disregarded for huge marginal error.
    """
    if sig1.shape != sig2.shape:
        raise ValueError("Both signals should be of the same size")
    nb_channels = sig1.shape[-1]
    eps = np.zeros((sig1.shape[0], nb_channels))
    tol = np.zeros((sig1.shape[0], nb_channels))
    tol_rate = 1e3
    for ind in range(sig1.shape[0]):
        s1 = sig1[ind].reshape(sig1.shape[2:])
        s2 = sig2[ind].reshape(sig2.shape[2:])
        for ch in range(nb_channels):
            diff_sig = np.abs((s1[:, ch]-s2[:, ch])/s1[:, ch])
            tol[ind, ch] = np.sum(diff_sig > tol_rate)
            diff_sig = [elem if elem < tol_rate else 0 for elem in diff_sig]
            eps[ind, ch] = np.mean(diff_sig)
    return eps, tol   

def sorted_dict(d):
    """
    :return: sorted dict by valye from d
    """
    d_sorted = {}
    for elem in sorted(d.items(), key=lambda x:x[1]):
        d_sorted[elem[0]]=elem[1]
    return d_sorted

def random_same_class_sig(model, x, X_set):
    """
    :return: a random signal from X_set that is the same class as x
    """
    y=model.predict(x)
    cond = True
    while cond:
        rg = np.random.randint(0, X_set.shape[0],1)[0]
        if model.predict(X_set[rg:rg+1])==y:
            cond=False
    print(rg)
    return X_set[rg:rg+1]

def random_diff_class_sig(model, x, X_set):
    y=model.predict(x)
    cond = True
    while cond:
        rg = np.random.randint(0, X_set.shape[0],1)[0]
        if model.predict(X_set[rg:rg+1])!=y:
            cond=False
    return X_set[rg:rg+1]

def random_eig_vect(n):
    '''
    Get a random vector of size n with a sum det
    '''
    vec_res = []
    for _ in range(n-1):
        vec_res.append(np.random.uniform(low=0,high=1))
    if np.sum(vec_res) >=n:
        vec_res = [v-np.random.uniform(np.min(vec_res)) for v in vec_res]
    vec_res.append(n-np.sum(vec_res))
    return vec_res
   
    
def random_MVGuaussian_params(max_mu, std, n):
    '''
    Get random params of an MVG
    '''
    mu = np.random.uniform(0,max_mu, size=n)
    semi_pos = False
    while not semi_pos:
        corr =  random_correlation.rvs(tuple(random_eig_vect(n)))
        cov = np.dot(corr, np.diag(std*np.ones(n)))
        if np.sum(np.linalg.eigvals(cov)>=0)==n:
                semi_pos = True
    return mu, cov

def noisy_input(X, y, mu_n, std_n, n_candidate): #Keep
    X_noisy = np.zeros((0,1,X.shape[-2], X.shape[-1]))
    y_noisy = np.zeros((0))
    for ind, x in enumerate(X):
        noise = np.zeros((0,1,X.shape[-2], X.shape[-1]))
        for _ in range(n_candidate):
            if X.shape[-1]==1:
                mu = mu_n * np.ones(1)
                sigma = std_n * np.ones((1,1))
            else:
                mu, sigma = random_MVGuaussian_params(mu_n, std_n, X.shape[-1])
            n = np.random.multivariate_normal(mu, sigma, X.shape[-2])
            n -= np.mean(n)
            noise = np.concatenate([noise, n[np.newaxis, np.newaxis, :, :]])
        xn = np.repeat(x[np.newaxis, :], n_candidate, axis=0)
        yn = np.repeat(y[ind], n_candidate, axis=0)
        X_n = xn + noise
        X_noisy = np.concatenate([X_noisy, X_n])
        y_noisy = np.concatenate([y_noisy, yn])
    return X_noisy, y_noisy
    

def adjust_seg_size(X, new_shape): #Keep
    assert len(new_shape)==2
    assert len(X.shape)==4
    X_ = copy.deepcopy(X)
    seg_size = X.shape[-2]
    new_seg_size = new_shape[0]
    X_adjusted = np.zeros(X.shape[:-2]+new_shape)
    channel_nb = X.shape[-1]
    new_channel_nb = new_shape[1]
    if new_seg_size < seg_size:
        if new_channel_nb < channel_nb:
            X_adjusted[:,:,:,:] = X_[:,:,:new_seg_size,:new_channel_nb]
        else:
            X_adjusted[:,:,:,:channel_nb] = X_[:,:,:new_seg_size,:]
    else:
        if new_channel_nb < channel_nb:
            X_adjusted[:,:,:seg_size,:] = X_[:,:,:,:new_channel_nb]
        else:
            X_adjusted[:,:,:seg_size,:channel_nb] = X_[:,:,:,:]
        
    return X_adjusted

def adjust_labels(y, class_nb): #Keep
    assert len(y.shape)==1
    adjusted_y = copy.deepcopy(y)
    for i in range(adjusted_y.shape[0]):
        if y[i] >= class_nb:
            adjusted_y[i] = np.random.randint(0, class_nb)
        if i==0:
            adjusted_y[i] = class_nb-1
    return adjusted_y
                                             
                                             
def data_normalize(data, max_min=None, norm_range=(-1,1)):
    print("Normalizing data to {}".format(norm_range))
    if max_min is None:
        d_max =  np.ceil(np.max(data))
        d_min = np.floor(np.min(data))
    else:
        assert len(max_min)==2, 'max-min should have two members (max, min)'
        d_max = max_min[0]
        d_min = max_min[1]
    num = (data - d_min) * (norm_range[1] - norm_range[0])
    denum = d_max - d_min
    norm_data = norm_range[0] + (num/denum) 
    return norm_data, [d_max, d_min]
#%% Plotting functions
def signal_plot(x, single_channel = False, save_fig = None):
    """
    Plots a single signal with multiple channels in different windows.
    """
    if not single_channel:
        nb_channel = x.shape[-1]
    else:
        nb_channel = 1
    if nb_channel > 1:    
        for c in range(nb_channel):    
            plt.figure(num = "Channel {}".format(c))
            plt.plot(x[:, c])
            plt.ylabel("Magnitude")
            plt.xlabel("Time")
    #        plt.show()
            print("Channel {} plotted! \n".format(c))
            if save_fig is not None:
                plt.savefig(save_fig+str(c))
    else:
        plt.figure(num = "Signal Plot")
        plt.plot(x)
        plt.ylabel("Magnitude")
        plt.xlabel("Time")
#        plt.show()
        print("Signal plotted! \n")        
        if save_fig is not None:
            plt.savefig(save_fig)
        
        
        
def superposed_plot(x1, lb1, x2, lb2, single_channel = False, save_fig = None, fig_name = ""):
    """
    Plots a two signals with multiple channels in different windows 
    for each channel.
    """
    if x1.shape != x2.shape:
            raise ValueError("Both inputs should be of the same size")
    if not single_channel:
        nb_channel = x1.shape[-1]
        
    else:
        nb_channel = 1
    if nb_channel > 1:    
        for c in range(nb_channel):
            plt.figure(num = fig_name+"Channel {}".format(c))
            plt.plot(x1[:, c], 'k', label = lb1)
            plt.plot(x2[:, c], 'r:', label = lb2)
            plt.ylabel("Magnitude")
            plt.xlabel("Time")
            plt.legend()
    #        plt.show()
            print("Channel {} plotted! \n".format(c))
            if save_fig is not None:
                plt.savefig(save_fig+str(c))
    else:
        plt.figure(num = fig_name+"Signals plot")
        plt.plot(x1, 'k', label = lb1)
        plt.plot(x2, 'r:', label = lb2)
        plt.ylabel("Magnitude")
        plt.xlabel("Time")
        plt.legend()
#        plt.show()
        print("Signals plotted! \n")
        if save_fig is not None:
            plt.savefig(save_fig)
        
 
def contour2DPlot(X, ttl=1):
    if len(X.shape) !=2 or X.shape[-1]!=2 :
        raise ValueError("X must be of shape (n, 2)")
    # Extract x and y
    x = X[:, 0]
    y = X[:, 1]
    # Define the borders
    deltaX = (max(x) - min(x))/10
    deltaY = (max(y) - min(y))/10
    xmin = min(x) - deltaX
    xmax = max(x) + deltaX
    ymin = min(y) - deltaY
    ymax = max(y) + deltaY
    # Create meshgrid
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)
    fig = plt.figure(figsize=(8,8), num=ttl)
    ax = fig.gca()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    cfset = ax.contourf(xx, yy, f, cmap='coolwarm')
    ax.imshow(np.rot90(f), cmap='coolwarm', extent=[xmin, xmax, ymin, ymax])
    cset = ax.contour(xx, yy, f, colors='k')
    ax.clabel(cset, inline=1, fontsize=10)
    ax.set_xlabel('X[0]')
    ax.set_ylabel('X[1]')
      
def bargraph_autolabel(a, ax, fsize):
    """Attach a text label above each bar in *a*, displaying its value."""
    for rect in a:
        height = rect.get_height()
        ax.annotate('{:.2f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=fsize, fontweight='bold')

        
#%% UCR dataset

def load_ucr_data(dataset_name, parent_file):
    data_train = []
    data_file = parent_file+"/"+dataset_name+"/"+dataset_name+"_TRAIN.tsv"
    with open(data_file) as f:
        content = csv.reader(f, delimiter='\t')
        next(content)
        for row in content:
            r = [float(el) if (el!='NaN' and el!='' and el!='?') else  0.0 for el in row]
            data_train.append(np.array(r))
    data_train = np.array(data_train)
    y_train_ = data_train[:,0].astype(np.int)
    
    labels = {}
    for i, y in enumerate(np.sort(np.unique(y_train_))):
        labels[y]=i
    y_train = np.array([labels[y] for y in y_train_])
    X_train = data_train[:, np.newaxis, :, np.newaxis]
    
    data_test = []
    data_file = parent_file+"/"+dataset_name+"/"+dataset_name+"_TEST.tsv"
    with open(data_file) as f:
        content = csv.reader(f, delimiter='\t')
        next(content)
        for row in content:
            r = [float(el) if (el!='NaN' and el!='' and el!='?') else  0.0 for el in row]
            data_test.append(np.array(r))
    data_test = np.array(data_test)
    y_test_ = data_test[:,0].astype(np.int)
    y_test = np.array([labels[y] for y in y_test_])
    X_test = data_test[:, np.newaxis, :, np.newaxis]
    
    rand_indices = np.arange(X_train.shape[0])
    np.random.shuffle(rand_indices)
    X_train = X_train[rand_indices]
    y_train = y_train[rand_indices]
    rand_indices = np.arange(X_test.shape[0])
    np.random.shuffle(rand_indices)
    X_test = X_test[rand_indices]
    y_test = y_test[rand_indices]
    # print("\nTraining shape: {}{} \nTesting shape : {}{}\n".format(
    #     X_train.shape, y_train.shape, X_test.shape, 
    #     y_test.shape))
    
    return X_train, y_train, X_test, y_test

def process_UCR_UV(parent_file):
    data_files = [f for f in os.listdir(parent_file) if os.path.isdir(os.path.join(parent_file, f)) ]
    json_data = {}
    discarded_data_file = open("DiscardedData.txt", "w")
    try:
        os.makedirs("UCRData_UV")
    except FileExistsError:
        pass
    for i, dataset_name in enumerate(data_files):
        print("{}: {}/{}".format(dataset_name, i, len(data_files)))
        try:
            X_train, y_train, X_test, y_test = load_ucr_data(dataset_name, parent_file)
        except KeyError:
            discarded_data_file.write("{} discarded for labels mismatch \n".format(dataset_name))
            continue
        if ((X_train.shape[2]!=X_test.shape[2]) or (X_train.shape[3]!=X_test.shape[3]) or (len(np.unique(y_train))!=len(np.unique(y_test)))):
            print(X_train.shape[2],X_test.shape[2] ,X_train.shape[3] ,X_test.shape[3], len(np.unique(y_train)), len(np.unique(y_test)))
            discarded_data_file.write("{} discarded: Seg size {}/{} | Channel {}/{} |"
                  " Class {}/{} | Instances {}/{}\n".format(dataset_name, 
                          X_train.shape[2],X_test.shape[2] ,X_train.shape[3] ,
                          X_test.shape[3], len(np.unique(y_train)), 
                          len(np.unique(y_test)), len(X_train), len(X_test)))
        else:
            json_data[dataset_name]={
                    "path": "UCRData_UV/"+dataset_name+".pkl",
                	 "SEG_SIZE": X_train.shape[2], 
                	 "CHANNEL_NB": X_train.shape[3],
                	 "CLASS_NB": len(np.unique(y_train))
                }
            pkl.dump([X_train, y_train, X_test, y_test], open("UCRData_UV/"+dataset_name+".pkl", "wb")) 
    with open('UCR__UV_parameters.json', 'w') as outfile:
        json.dump(json_data, outfile, indent=2)
    discarded_data_file.close()

def load_mv_ucr_data(dataset_name, parent_file):
    #Extract Data Dimensions
    dim_df = pd.read_csv(parent_file+"/DataDimensions.csv")
    ds_idx = dim_df[dim_df["Problem"]==dataset_name].index[0]
    ds_trn_size = dim_df.at[ds_idx, "TrainSize"]
    ds_tst_size = dim_df.at[ds_idx, "TestSize"]
    ds_channel_nb = dim_df.at[ds_idx, "NumDimensions"]
    ds_seg_size = dim_df.at[ds_idx, "SeriesLength"]
    
    
    
    #Extract TrainData
    X_train = np.zeros((ds_trn_size, 1, ds_seg_size, ds_channel_nb))
    for ch in range(ds_channel_nb):
        data_file = parent_file+"/"+dataset_name+"/"+dataset_name+"Dimension"+str(ch+1)+"_TRAIN.arff"
        data, meta = arff.loadarff(data_file)
        train_data = data[meta.names()[:-1]] #everything but the last column
        train_data = np.array(train_data.tolist())
        X_train[:,:,:,ch] = train_data.reshape((ds_trn_size, 1,ds_seg_size))
    #Extract TrainLabels
    data, meta = arff.loadarff(open(data_file, "r"))
    train_lbl = data[meta.names()[-1]] #LastColumn
    train_lbl = np.array([ss.decode('ascii') for ss in train_lbl])
    labels = {}
    for i, y in enumerate(np.sort(np.unique(train_lbl))):
        labels[y]=i
    y_train = np.array([labels[y] for y in train_lbl])
    
    #Extract TestData
    X_test = np.zeros((ds_tst_size, 1, ds_seg_size, ds_channel_nb))
    for ch in range(ds_channel_nb):
        data_file = parent_file+"/"+dataset_name+"/"+dataset_name+"Dimension"+str(ch+1)+"_TEST.arff"
        data, meta = arff.loadarff(data_file)
        test_data = data[meta.names()[:-1]] #everything but the last column
        test_data = np.array(test_data.tolist())
        X_test[:,:,:,ch] = test_data.reshape((ds_tst_size, 1,ds_seg_size))
    #Extract TestLabels
    data, meta = arff.loadarff(open(data_file, "r"))
    test_lbl = data[meta.names()[-1]] #LastColumn
    test_lbl = np.array([ss.decode('ascii') for ss in test_lbl])
    labels = {}
    for i, y in enumerate(np.sort(np.unique(test_lbl))):
        labels[y]=i
    y_test = np.array([labels[y] for y in test_lbl])
    
    rand_indices = np.arange(X_train.shape[0])
    np.random.shuffle(rand_indices)
    X_train = X_train[rand_indices]
    y_train = y_train[rand_indices]
    rand_indices = np.arange(X_test.shape[0])
    np.random.shuffle(rand_indices)
    X_test = X_test[rand_indices]
    y_test = y_test[rand_indices]
    return X_train, y_train, X_test, y_test

def process_UCR_MV(parent_file):
    data_files = [f for f in os.listdir(parent_file) if os.path.isdir(os.path.join(parent_file, f)) ]
    json_data = {}
    discarded_data_file = open("DiscardedData_MV.txt", "w")
    try:
        os.makedirs("UCRData_MV")
    except FileExistsError:
        pass
    for i, dataset_name in enumerate(data_files):
        if dataset_name in ["DuckDuckGeese", "FaceDetection", "Images", "InsectWingbeat", "PhonemeSpectra"]:
            print("Skipping "+dataset_name)
            continue
        # if os.path.isfile("UCRData_MV/"+dataset_name+".pkl"):
        #     print("Exists "+dataset_name)
        #     continue
        print("{}: {}/{}".format(dataset_name, i, len(data_files)))
        try:
            X_train, y_train, X_test, y_test = load_mv_ucr_data(dataset_name, parent_file)
        except KeyError:
            discarded_data_file.write("{} discarded for labels mismatch \n".format(dataset_name))
            continue
        if ((X_train.shape[2]!=X_test.shape[2]) or (X_train.shape[3]!=X_test.shape[3]) or (len(np.unique(y_train))!=len(np.unique(y_test)))):
            print(X_train.shape[2],X_test.shape[2] ,X_train.shape[3] ,X_test.shape[3], len(np.unique(y_train)), len(np.unique(y_test)))
            discarded_data_file.write("{} discarded: Seg size {}/{} | Channel {}/{} |"
                  " Class {}/{} | Instances {}/{}\n".format(dataset_name, 
                          X_train.shape[2],X_test.shape[2] ,X_train.shape[3] ,
                          X_test.shape[3], len(np.unique(y_train)), 
                          len(np.unique(y_test)), len(X_train), len(X_test)))
        else:
            json_data[dataset_name]={
                    "path": "UCRData_MV/"+dataset_name+".pkl",
                	 "SEG_SIZE": X_train.shape[2], 
                	 "CHANNEL_NB": X_train.shape[3],
                	 "CLASS_NB": len(np.unique(y_train))
                }
            pkl.dump([X_train, y_train, X_test, y_test], open("UCRData_MV/"+dataset_name+".pkl", "wb")) 
    with open('UCR__MV_parameters.json', 'w') as outfile:
        json.dump(json_data, outfile, indent=2)
    discarded_data_file.close()
# process_UCR_MV("../UCR_dataset_MV")   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    