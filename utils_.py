import copy
import numpy as np


def adjust_seg_size(X, new_shape):
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

def adjust_labels(y, class_nb):
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







