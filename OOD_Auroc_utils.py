import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import auc


def APR(recall, precision):#average_precision_score
    assert len(precision)==len(recall)
    AP = 0
    for i in range(1, len(precision)):
        AP += (recall[i] - recall[i-1]) * precision[i]
    return AP

def best_f1(recall, precision):# highest f1 score
    # f1 metric measures the balance between precision and recall.
    assert len(precision)==len(recall)
    best_f1 = -np.inf
    for R, P in zip(recall, precision):
        if not np.any(np.isnan([P, R])):
            f1 = (2*P*R)/(P+R)
            if f1 > best_f1:
                best_f1 = f1
    if best_f1==-np.inf:
        return 1
    else:
        return best_f1


def ratio_accuracy(ratio_ref, ratio, mode='out_data', method='var', var_factor=2):
    if not np.all(np.isfinite(ratio)):
        return -1
    accuracy = 0
    if method=='var':
        mu = np.mean(ratio_ref)
        var = np.sqrt(np.var(ratio_ref))
        r_min = mu - var_factor *var
        r_max = mu + var_factor *var
    elif method=='minmax':
        r_min = np.min(ratio_ref)
        r_max = np.max(ratio_ref)
    else:
        raise ValueError("Wrong method entry")
    if mode=='in_data':
        for r in ratio:
            if r <= r_max and r >= r_min:
                accuracy += 1
    elif mode=='out_data':
        for r in ratio:
            if r > r_max or r < r_min:
                accuracy += 1
    else:
        raise ValueError("Wrong mode entry")
    return 100*accuracy/len(ratio)


def is_OOD(ratio_ref, ratio,  method='var', var_factor=0.25):
    if method=='var':
        mu = np.mean(ratio_ref)
        var = np.sqrt(np.var(ratio_ref))
        r_min = mu - var_factor *var
        r_max = mu + var_factor *var
    elif method=='minmax':
        r_min = np.min(ratio_ref)
        r_max = np.max(ratio_ref)
    else:
        raise ValueError("Wrong method entry")
    labels = []
    for r in ratio:
        try:
            if r > r_max or r < r_min:
                labels.append(1)
            else:
                labels.append(-1)
        except RuntimeWarning:
                labels.append(1)
    return np.array(labels)

def label_ratio(ratio_dict):
    for k in ratio_dict.keys():
        assert k in['in', 'ood']
    ratio_elements = []
    ratio_labels = []
    for k, v in ratio_dict.items():
        for el in v:
            ratio_elements.append(el)
            if k=='in':
                ratio_labels.append(-1)
            else:
                ratio_labels.append(1)
    return ratio_elements, ratio_labels
    
def tpr_fpr(ratio_ref, ratio_elements, ratio_labels, th):
    predicted_ood = is_OOD(ratio_ref, ratio_elements,  method='var', var_factor=th)
    tn, fp, fn, tp = cm(ratio_labels, predicted_ood, [-1,1]).ravel()
    tpr = tp / (tp+fn)
    fpr = fp / (fp+tn)
    return fpr, tpr

def precision_recall(ratio_ref, ratio_elements, ratio_labels, th):
    predicted_ood = is_OOD(ratio_ref, ratio_elements,  method='var', var_factor=th)
    tn, fp, fn, tp = cm(ratio_labels, predicted_ood, [-1,1]).ravel()
    # if tp==0:
    #     return 0, 0
    P = tp / (tp+fp)
    R = tp / (tp+fn)
    return P, R

def get_best_th(ratio_ref, ratio_elements, ratio_labels, th_range=[0,15], th_step = 0.5):
    best_score = 0
    best_th = -np.inf
    for th in np.arange(th_range[0], th_range[1], th_step):
        predicted_ood = is_OOD(ratio_ref, ratio_elements,  method='var', var_factor=th)
        tn, fp, fn, tp = cm(ratio_labels, predicted_ood, [-1,1]).ravel()
        tpr = tp / (tp+fn)
        fpr = fp / (fp+tn)
        score = tpr-fpr
        if score > best_score:
            best_score = score
            best_th = th
    return best_th

def fpr_at_tpr(ratio_ref, ratio_elements, ratio_labels, th_range=[0,100], 
               th_step = 1, at_tpr=0.8,
          save_roc_curve=False, roc_curve_path="ROC_Curve.png", roc_curve_title="ROC Curve"):
    tpr = []
    fpr = []
    th_range = np.arange(th_range[0], th_range[1], th_step)
    for th in th_range:
        v_fpr, v_tpr = tpr_fpr(ratio_ref, ratio_elements, ratio_labels, th)
        tpr.append(v_tpr)
        fpr.append(v_fpr)
    
    if save_roc_curve:
        plt.figure(num=roc_curve_path)
        plt.plot(fpr, tpr)    
        plt.plot(np.arange(0,1.1,0.1), np.arange(0,1.1,0.1), ':')
        plt.ylabel("TPR")
        plt.xlabel("FPR")
        plt.title(roc_curve_title, position=(0.5, 0.1), fontsize=40)
        plt.grid(color='k', linestyle=':')
        plt.xlim([-0.1, 1.1])
        plt.ylim([-0.1, 1.1])
        plt.savefig(roc_curve_path, bbox_inches = 'tight', pad_inches = .1,
                    dpi=100)
        
    fpr_th_step = th_step/10
    if tpr[0]>tpr[-1]:
        tpr = tpr[::-1]
        fpr = fpr[::-1]
        th_range = th_range[::-1] 
        fpr_th_step *= -1
    if tpr[0]>=at_tpr:
        return 0
    elif tpr[-1]>=at_tpr:
        for i, v_tpr in enumerate(tpr):
            if v_tpr > at_tpr:
                break
        smallest_gap = np.inf
        fpr_at_tpr = np.inf
        for th in np.arange(th_range[i-1], th_range[i], fpr_th_step):
            v_fpr, v_tpr = tpr_fpr(ratio_ref, ratio_elements, ratio_labels, th)
            gap = np.abs(v_tpr-at_tpr)
            if gap < smallest_gap:
                smallest_gap = gap
                fpr_at_tpr = v_fpr
        return fpr_at_tpr
    else:
        return -1

def auroc(ratio_ref, ratio_elements, ratio_labels, th_range=[0,15], th_step = 0.5):
    tpr = []
    fpr = []
    for th in np.arange(th_range[0], th_range[1], th_step):
        v_fpr, v_tpr = tpr_fpr(ratio_ref, ratio_elements, ratio_labels, th)
        tpr.append(v_tpr)
        fpr.append(v_fpr)
    auc_value = auc(fpr, tpr)
    if auc_value >=0.5:
        return auc_value
    else:
        return 1-auc_value
    

def f1(ratio_ref, ratio_elements, ratio_labels, th_range=[0,15], th_step = 0.5):
    precision = []
    recall = []
    for th in np.arange(th_range[0], th_range[1], th_step):
        P, R = precision_recall(ratio_ref, ratio_elements, ratio_labels, th)
        precision.append(P)
        recall.append(R)
    f1_value = best_f1(recall, precision)
    return f1_value
    

#################### Likelihood Regret #######################################
def LR_is_OOD(LR_in, LR, tau):
    labels = []
    for lr in LR:
        try:
            if lr > tau:
                labels.append(1)
            else:
                labels.append(-1)
        except RuntimeWarning:
                labels.append(1)
    return np.array(labels)    

def label_LR(LR_dict):
    for k in LR_dict.keys():
        assert k in['in', 'ood']
    LR_elements = []
    LR_labels = []
    for k, v in LR_dict.items():
        for el in v:
            LR_elements.append(el)
            if k=='in':
                LR_labels.append(-1)
            else:
                LR_labels.append(1)
    return LR_elements, LR_labels
    
    
def LR_tpr_fpr(LR_ref, LR_elements, LR_labels, th):
    predicted_ood = LR_is_OOD(LR_ref, LR_elements, th)
    tn, fp, fn, tp = cm(LR_labels, predicted_ood, labels=[-1,1]).ravel()
    tpr = tp / (tp+fn)
    fpr = fp / (fp+tn)
    return fpr, tpr

def LR_precision_recall(LR_ref, LR_elements, LR_labels, th):
    predicted_ood = LR_is_OOD(LR_ref, LR_elements, th)
    tn, fp, fn, tp = cm(LR_labels, predicted_ood, labels=[-1,1]).ravel()
    P = tp / (tp+fp)
    R = tp / (tp+fn)
    return P, R

def LR_auroc(LR_ref, LR_elements, LR_labels, th_range=[-10,10], th_step = 1):
    tpr = []
    fpr = []
    for th in np.arange(th_range[0], th_range[1], th_step):
        v_fpr, v_tpr = LR_tpr_fpr(LR_ref, LR_elements, LR_labels, th)
        tpr.append(v_tpr)
        fpr.append(v_fpr)
    auc_value = auc(fpr, tpr)
    if auc_value >=0.5:
        return auc_value
    else:
        return 1-auc_value
    
def LR_f1(LR_ref, LR_elements, LR_labels, th_range=[-10,10], th_step = 1):
    precision = []
    recall = []
    for th in np.arange(th_range[0], th_range[1], th_step):
        P, R = LR_precision_recall(LR_ref, LR_elements, LR_labels, th)
        precision.append(P)
        recall.append(R)
    f1_value = best_f1(recall, precision)
    return f1_value

def LR_fpr_at_tpr(LR_ref, LR_elements, LR_labels, th_range=[-100,100], 
                 th_step = 5, at_tpr=0.8,
          save_roc_curve=False, roc_curve_path="ROC_Curve.png", roc_curve_title="ROC Curve"):
    tpr = []
    fpr = []
    th_range = np.arange(th_range[0], th_range[1], th_step)
    for th in th_range:
        v_fpr, v_tpr = LR_tpr_fpr(LR_ref, LR_elements, LR_labels, th)
        tpr.append(v_tpr)
        fpr.append(v_fpr)
        
    if save_roc_curve:
        plt.figure(num=roc_curve_path)
        plt.plot(fpr, tpr)    
        plt.plot(np.arange(0,1.1,0.1), np.arange(0,1.1,0.1), ':')
        plt.ylabel("TPR")
        plt.xlabel("FPR")
        plt.title(roc_curve_title, position=(0.5, 0.1), fontsize=40)
        plt.grid(color='k', linestyle=':')
        plt.xlim([-0.1, 1.1])
        plt.ylim([-0.1, 1.1])
        plt.savefig(roc_curve_path, bbox_inches = 'tight', pad_inches = .1,
                    dpi=100)
        
    fpr_th_step = th_step/10
    if tpr[0]>tpr[-1]:
        tpr = tpr[::-1]
        fpr = fpr[::-1]
        th_range = th_range[::-1] 
        fpr_th_step *= -1
    if tpr[0]>=at_tpr:
        return 0
    elif tpr[-1]>=at_tpr:
        for i, v_tpr in enumerate(tpr):
            if v_tpr > at_tpr:
                break
        smallest_gap = np.inf
        fpr_at_tpr = np.inf
        for th in np.arange(th_range[i-1], th_range[i], fpr_th_step):
            v_fpr, v_tpr = LR_tpr_fpr(LR_ref, LR_elements, LR_labels, th)
            gap = np.abs(v_tpr-at_tpr)
            if gap < smallest_gap:
                smallest_gap = gap
                fpr_at_tpr = v_fpr
        return fpr_at_tpr
    else:
        return -1
    
    
    
    
    
    