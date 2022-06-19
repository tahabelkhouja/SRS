import numpy as np
import copy
from dtw_master import dtw as dtwM

def signal_step_extend(x, timestep, start, end):
    '''
    Enlarge x at timestep with length start:end
    '''
    assert len(x.shape)==2
    end += 1
    span = end-start
    x_res = copy.deepcopy(x)
    x_res[timestep:timestep+span,:] = x[timestep, :] * np.ones(x_res[timestep:timestep+span,:].shape)
    if -span+1 ==0:
        x_res[timestep+span:, :] = x[timestep+1:, :]
    else:
        x_res[timestep+span:, :] = x[timestep+1:-span+1, :]
    return x_res
    
def signal_step_reduce(x, timestep, start, end):
    '''
    Reduce x at timestep with length start:end
    '''
    assert len(x.shape)==2
    end += 1
    span = end-start
    x_res = copy.deepcopy(x)
    x_res[start,:] = x[timestep, :] 
    x_res[start+1:, :] = np.concatenate([x[end:, :], x[-1, :] * np.ones((span-1, x.shape[-1]))], axis=0)
    return x_res

def signal_step_align(x, s1, s2):
    '''
    Align x according to s1 e1
    '''
    assert len(x.shape)==2
    x_res = copy.deepcopy(x)
    if s1==s2:
        return x_res
    x_res = np.roll(x_res, s1-s2, axis=0)
    if s1-s2 > 0:
        x_res[:s1-s2, :] = x[s1-s2, :] * np.ones(x_res[:s1-s2, :].shape)
    else:
        x_res[s1-s2:, :] = x[s1-s2, :] * np.ones(x_res[s1-s2:, :].shape)
    return x_res
    
def scan_path(path):
    '''
    Find longest one-to-many alignments or sequential one-to-one alignemnts
    '''
    one_to_many = [0, 0] # [start, end]
    many_to_one = [0, 0]
    diag_seq = [0, 0]
    scan = {
        'one_to_many': [0, 0], # [start, end]
        'many_to_one': [0, 0],
        'diag_seq': [0, 0]
        }
    len_seq = lambda x: x[1]-x[0]
    #look for one_to_many
    i = 0
    j = 1
    while i < len(path[0]):
        while (i+j < len(path[0]) and path[0][i]==path[0][i+j]):
            one_to_many = [i, i+j]
            j += 1
        if len_seq(one_to_many)!=0:
            if len_seq(scan['one_to_many']) < len_seq(one_to_many):
                scan['one_to_many'] = one_to_many
            one_to_many = [0, 0]
            i = i+j
            j = 1
        else:
            i += 1   
    #look for many_to_one
    i = 0
    j = 1
    while i < len(path[1]):
        while (i+j < len(path[1]) and path[1][i]==path[1][i+j]):
            many_to_one = [i, i+j]
            j += 1
        if len_seq(many_to_one)!=0:
            if len_seq(scan['many_to_one']) < len_seq(many_to_one):
                scan['many_to_one'] = many_to_one
            many_to_one = [0, 0]
            i = i+j
            j = 1
        else:
            i += 1
    #look for diag_seq
    i = 0
    j = 0
    while i < len(path[1]):
        while (j+1 < len(path[1]) and path[0][j]==path[0][j+1]-1 and path[1][j]==path[1][j+1]-1):
            diag_seq = [i, j+1]
            j += 1
        if len_seq(diag_seq)!=0:
            if len_seq(scan['diag_seq']) < len_seq(diag_seq):
                scan['diag_seq'] = diag_seq
            diag_seq = [0, 0]
            i = j
        else:
            i += 1
            j = i
    return scan, [x for x in scan.keys()], [len_seq(scan[x]) for x in scan]
        
        
        
        
def signals_align(x_ref, x):
    '''
    return x aligned the best with x_ref
    '''
    assert len(x_ref.shape)==len(x.shape)
    x_out = copy.deepcopy(x.reshape(x.shape[-2:]))
    x_in = x_ref.reshape(x_ref.shape[-2:])
    d, _, _, dtw_path = dtwM.dtw(x_in, x_out, dist=lambda x, y: np.linalg.norm(x-y, ord=2))
    scan_res, scan_keys, scan_lengths = scan_path(dtw_path)
    
    ##Prioritize Diagonal Method 1
    scan_lengths[2] += 1 
    best_transform_key = scan_keys[np.argmax(scan_lengths)]
    ##Prioritize Diagonal Method 2
    # if(scan_lengths[2] > 1.15*x.shape[-2]):
    #     best_transform_key = 'diag_seq'
    # else:
    #     best_transform_key = scan_keys[np.argmax(scan_lengths)]
    
    if best_transform_key=='one_to_many':
        x_res = signal_step_reduce(x_out, dtw_path[0][scan_res[best_transform_key][0]], 
                                   dtw_path[1][scan_res[best_transform_key][0]], dtw_path[1][scan_res[best_transform_key][1]])
    elif best_transform_key=='many_to_one':
        x_res = signal_step_extend(x_out, dtw_path[1][scan_res[best_transform_key][0]], 
                                   dtw_path[0][scan_res[best_transform_key][0]], dtw_path[0][scan_res[best_transform_key][1]])
    elif best_transform_key=='diag_seq':
        x_res = signal_step_align(x_out, dtw_path[0][scan_res[best_transform_key][0]], dtw_path[1][scan_res[best_transform_key][0]])
    
    return x_res.reshape(x.shape)


def optimize_alignment(X_in, iterations=5):
    X_set = copy.deepcopy(X_in)
    indices = np.arange(X_set.shape[0])
    best_score = alignment_score(X_set)
    print("Initial Average alignment score: {:.1f}%".format(best_score))
    for _ in range(iterations):
        np.random.shuffle(indices)
        X_set_opt = copy.deepcopy(X_set[:1])
        for i1, i2 in zip(indices[:-1], indices[1:]):
            X_opt = signals_align(X_set[i1:i1+1], X_set[i2:i2+1])
            X_set_opt = np.concatenate([X_set_opt, X_opt], axis=0)
        if alignment_score(X_set_opt) > best_score:
            X_set = X_set_opt
            best_score = alignment_score(X_set_opt)
            print("Average alignment score: {:.1f}%".format(best_score))
        else:
            print("Final Average alignment score: {:.1f}%".format(alignment_score(X_set)))
            return X_set
    print("Final Average alignment score: {:.1f}%".format(alignment_score(X_set_opt)))
    return X_set_opt
        
def optimize_alignment_to_pattern(pattern, X_in, iterations=5):
    X_set = copy.deepcopy(X_in)
    best_score = alignment_score(X_set)
    print("Initial Average alignment score: {:.1f}%".format(best_score))
    for _ in range(iterations):
        X_set_opt = np.zeros((0,)+X_set.shape[1:])
        for i1 in range(X_set.shape[0]):
            X_opt = signals_align(pattern.reshape(X_set[i1:i1+1].shape), X_set[i1:i1+1])
            X_set_opt = np.concatenate([X_set_opt, X_opt], axis=0)
        if alignment_score(X_set_opt) > best_score:
            X_set = X_set_opt
            best_score = alignment_score(X_set_opt)
            print("Average alignment score: {:.1f}%".format(best_score))
        else:
            print("Final Average alignment score: {:.1f}%".format(alignment_score(X_set)))
            return X_set
    print("Final Average alignment score: {:.1f}%".format(alignment_score(X_set_opt)))
    return X_set_opt


def alignment_score(X_set):
    '''
    return the score describing how well-aligned are the inputs in X_set
    '''
    align_score = 0
    for x1, x2 in zip(X_set[:-1], X_set[1:]):
        _, _, _, dtw_path = dtwM.dtw(x1.reshape((x1.shape[-2:])), x2.reshape((x2.shape[-2:])), dist=lambda x, y: np.linalg.norm(x-y, ord=2))
        align_score += 100*np.sum(dtw_path[0]==dtw_path[1])/x1.shape[-2]
    return align_score / (X_set.shape[0]-1)
    

def alignment_analysis(X_set):
    #1st search
    align_mx = {}
    for i in range(X_set.shape[0]):
        print("Progress {}/{}".format(i+1, X_set.shape[0]))
        for j in range(i+1, X_set.shape[0]):
            x1 = X_set[i]
            x2 = X_set[j]
            _, _, _, dtw_path = dtwM.dtw(x1.reshape((x1.shape[-2:])), x2.reshape((x2.shape[-2:])), dist=lambda x, y: np.linalg.norm(x-y, ord=2))
            align_mx[(i,j)] = 100*np.sum(dtw_path[0]==dtw_path[1])/x1.shape[-2]
    sorted_mx = {k:v for k,v in sorted(align_mx.items(), key=lambda x:x[1], reverse=True)}
    good_align_set = []
    for k,v in sorted_mx.items():
        if v > 25:
           good_align_set.append(k[0])
           good_align_set.append(k[1]) 
    good_align_set = set(good_align_set)
    #2nd search
    good_align_set2 = set()
    for i in range(X_set.shape[0]):
        if i in good_align_set2:
            continue
        print("Progress {}/{}".format(i+1, X_set.shape[0]))
        for j in range(i+1, X_set.shape[0]):
            x1 = X_set[i]
            x2 = X_set[j]
            _, _, _, dtw_path = dtwM.dtw(x1.reshape((x1.shape[-2:])), x2.reshape((x2.shape[-2:])), dist=lambda x, y: np.linalg.norm(x-y, ord=2))
            align_score =  100*np.sum(dtw_path[0]==dtw_path[1])/x1.shape[-2]
            if align_score > 25:
                good_align_set2.add(i)
                good_align_set2.add(j)
                continue
