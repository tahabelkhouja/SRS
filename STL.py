import numpy as np
import copy 

from stldecompose import decompose

class STL_decomp():
    def __init__(self, seg_size, channel_nb, data, labels):
        assert len(data.shape)==3, "len(data.shape)!=3"
        self.pattern = None
        self.class_nb = len(np.unique(labels))
        self.seg_size = seg_size
        self.channel_nb = channel_nb
        self.data_array = copy.deepcopy(data)
        self.labels = copy.deepcopy(labels)
        self.data = {}
        for cl in range(self.class_nb): #divide per class
            x0 = np.zeros((0, self.seg_size, self.channel_nb))
            for i, y in enumerate(labels):
                if y==cl:
                    x = np.zeros_like(data[i:i+1])
                    for ch in range(self.channel_nb):
                        x[0,  :, ch] = data[i,:,ch]
                    x0 = np.concatenate([x0, x], axis=0)
            self.data[cl] = x0
        self.STL_extract_pattern()
            
    def STL_extract_pattern(self):
        """
        STL Decomposition of the data
        """
        decomp = None
        stl_decomp = {}
        stl_decomp['S'] = np.zeros((self.class_nb, self.seg_size))
        stl_decomp['T'] = np.zeros((self.class_nb, self.seg_size))
        self.pattern = {}
        self.seas_pattern = {}
        self.decomp_ = {}
        for cl in range(self.class_nb):
            self.seas_pattern[cl] = np.zeros((self.seg_size, self.channel_nb))
            self.pattern[cl] = np.zeros((self.seg_size, self.channel_nb))
            for ch in range(self.channel_nb):
                decomp = decompose(self.data[cl][:,:,ch].reshape(-1), period=self.seg_size)
                self.decomp_[ch] = decomp
                seasonal = decomp.seasonal
                trend = decomp.trend
                stl_decomp['S'][cl] = seasonal[:self.seg_size]
                stl_decomp['T'][cl] = np.mean(trend)
                self.seas_pattern[cl][:, ch] = seasonal[:self.seg_size]
                self.pattern[cl][:, ch] = stl_decomp['T'][cl]+stl_decomp['S'][cl]
            
    def residuals_of(self, data, labels):
        res_labels = copy.deepcopy(labels)
        residuals = copy.deepcopy(data)
        for i, res in enumerate(residuals):
            cl = labels[i]
            residuals[i] = res - self.pattern[cl]
        return residuals, res_labels
    



