# module for drug sensitivity
import numpy as np
from scipy.stats import ttest_ind
import time

class aece(object):
    def __init__(self, *, num_bootstrap = 50, limit = 2):
        # initialize model with parameters
        self.limit = limit
        self.num_bootstrap = num_bootstrap

    def __pvalue(X, y):
        # compute p-values via t-test
        ttest = [None] * X.shape[1]
        for i in range(X.shape[1]):
            sample1 = y[ X[:,i] == 1 ]
            sample0 = y[ X[:,i] == 0 ]
            if len(sample1) > 1:
                ttest[i] = ttest_ind(sample1, sample0, equal_var = False)
            else:
                ttest[i] = [np.nan]*2
        ttest = np.array(ttest)
        return ttest[:,1]
        
    def fit(self, X, y):
        # fit training data
        self.X = X
        self.y = y
        self.p = aece.__pvalue(X, y)
   
    def __group_data(X, y):
        # group samples with the same feature together
        # in case there is only one sample
        if len(X.shape) == 1:
            X = X[np.newaxis,:]  
        # group all data
        idx = list(range(X.shape[0]))
        group = []
        while len(idx)>0:
            standard = X[idx[0],:]
            group.append(np.array([idx[0],y[idx[0]]])[np.newaxis,:])
            del idx[0]
            if len(idx)>0:
                panel = idx.copy()
                for element in panel:
                    if sum(abs(X[element,:]-standard))==0:
                        group[-1] = np.r_[group[-1],
                              np.array([element,y[element]])[np.newaxis,:]]
                        idx.remove(element)
        group_pos = [int(element[0,0]) for element in group]
        group_feature = X[group_pos,:]
        return group, group_feature
    
    def __check(sample, group, group_feature):
        # find samples with the same given feature from group
        num_group = len(group)
        for i in range(num_group):     
            if sum(abs(sample - group_feature[i])) == 0:
                element = group[i]
                num_same = element.shape[0]
                est_mean = element[:,1].mean()
                est_var = element[:,1].var()
                return num_same, est_mean, est_var
        return 0, np.nan, np.nan
    
    def predict_naive(self, test_feature):
        # Empirical Conditional Expectation
        t0 = time.time()
        # Data setting
        test_IC50s = np.zeros(test_feature.shape[0])
        test_group, group_feature = aece.__group_data(test_feature,test_IC50s)
        group_relation = [list(map(int,element[:,0])) for element in test_group]
        # Prediction
        p_thresholding = np.sort(np.unique(self.p[~np.isnan(self.p)]))[::-1]
        pred = []
        for sample in group_feature:
            for i in range(len(p_thresholding)):
                print('sample %d/%d, thresholding %d/%d.' % 
                      (len(pred), len(group_feature)-1, i, len(p_thresholding)-1))         
                p_loc = self.p <= p_thresholding[i]
                temp_group, temp_group_feature = aece.__group_data(self.X[:,p_loc],
                                                                   self.y)
                temp_sample = sample[p_loc]
                num_same, temp_mean, temp_var = aece.__check(
                        temp_sample, temp_group, temp_group_feature)
                if num_same >= self.limit:
                    pred.append(temp_mean)
                    break
            print('prediction: %f.' % (pred[-1]))
        # Inverse projection to test_IC50s
        for i in range(len(pred)):
            test_IC50s[group_relation[i]] = pred[i]
        # Time running
        t1 = time.time()
        print('Time running: %f.' % (t1-t0))
        return test_IC50s
    
    def predict(self, test_feature):
        # Adaptive Empirical Conditional Expectation with early stopping
        t0 = time.time()
        # Data setting
        test_IC50s = np.zeros(test_feature.shape[0])
        test_group, group_feature = aece.__group_data(test_feature,test_IC50s)
        group_relation = [list(map(int,element[:,0])) for element in test_group]
        # Prediction
        p_thresholding = np.sort(np.unique(self.p[~np.isnan(self.p)]))[::-1]
        pred = []
        for sample in group_feature:
            est_mean = np.inf
            est_var = np.inf
            mean_std = []
            for i in range(len(p_thresholding)):
                print('sample %d/%d, thresholding %d/%d.' % 
                      (len(pred), len(group_feature)-1, i, len(p_thresholding)-1))         
                p_loc = self.p <= p_thresholding[i]
                temp_group, temp_group_feature = aece.__group_data(self.X[:,p_loc],
                                                                   self.y)
                temp_sample = sample[p_loc]
                num_same, temp_mean, temp_var = aece.__check(
                        temp_sample, temp_group, temp_group_feature)
                if num_same >= self.limit:
                    mean_std = mean_std or temp_mean
                    temp_var = (temp_mean-mean_std)**2 + temp_var/(num_same-1)
                    if temp_var <= est_var:
                        est_mean = temp_mean
                        est_var = temp_var
                        if i == len(p_thresholding)-1:
                            pred.append(est_mean)
                    else:
                        pred.append(est_mean)
                        break
            print('prediction: %f.' % (pred[-1]))
        # Inverse projection to test_IC50s
        for i in range(len(pred)):
            test_IC50s[group_relation[i]] = pred[i]
        # Time running
        t1 = time.time()
        print('Time running: %f.' % (t1-t0))
        return test_IC50s
    
    def __bootstrap_mean(self, sample, sample_counter):
        # Initialization
        mean_std = 0
        # Bootstrap
        for j in range(self.num_bootstrap):
            print('bootstrap: %d/%d, sample: %d/%d.' % (j, self.num_bootstrap-1,
                  sample_counter[0], sample_counter[1]))
            index = list(map(int,len(self.X)*np.random.rand(len(self.X))))
            X_new = self.X[index,:]
            y_new = self.y[index]
            p_new = aece.__pvalue(X_new, y_new)
            p_thresholding = np.sort(np.unique(p_new[~np.isnan(p_new)]))[::-1]
            for i in range(len(p_thresholding)):
                p_loc = p_new <= p_thresholding[i]
                group, group_feature = aece.__group_data(X_new[:,p_loc], y_new)
                temp_sample = sample[p_loc]
                num_same, temp_mean, temp_var = aece.__check(
                        temp_sample, group, group_feature)
                if num_same >= 1:
                    print('%d iteration in this phase.' % (i))
                    mean_std += temp_mean
                    break
        mean_std /= self.num_bootstrap
        print('bootstrap mean: %f.' % (mean_std))
        return mean_std
    
    def predict_bootstrap(self, test_feature):
        # Adaptive Empirical Conditional Expectation with early stopping and bootstrap mean
        t0 = time.time()
        # Data setting
        test_IC50s = np.zeros(test_feature.shape[0])
        test_group, group_feature = aece.__group_data(test_feature, test_IC50s)
        group_relation = [list(map(int,element[:,0])) for element in test_group]
        # Prediction
        p_thresholding = np.sort(np.unique(self.p[~np.isnan(self.p)]))[::-1]
        pred = []
        for sample in group_feature:
            est_mean = np.inf
            est_var = np.inf
            mean_std = self.__bootstrap_mean(sample,
                                             (len(pred), len(test_group)-1))
            for i in range(len(p_thresholding)):
                print('sample %d/%d, thresholding %d/%d.' % 
                      (len(pred), len(group_feature)-1, i, len(p_thresholding)-1))         
                p_loc = self.p <= p_thresholding[i]
                temp_group, temp_group_feature = aece.__group_data(self.X[:,p_loc],
                                                                   self.y)
                temp_sample = sample[p_loc]
                num_same, temp_mean, temp_var = aece.__check(
                        temp_sample, temp_group, temp_group_feature)
                if num_same >= self.limit:
                    temp_var = (temp_mean-mean_std)**2 + temp_var/(num_same-1)
                    if temp_var <= est_var:
                        est_mean = temp_mean
                        est_var = temp_var
                        if i == len(p_thresholding)-1:
                            pred.append(est_mean)
                    else:
                        pred.append(est_mean)
                        break
            print('prediction: %f.' % (pred[-1]))
        # Inverse projection to test_IC50s
        for i in range(len(pred)):
            test_IC50s[group_relation[i]] = pred[i]
        # Time running
        t1 = time.time()
        print('Time running: %f.' % (t1-t0))
        return test_IC50s