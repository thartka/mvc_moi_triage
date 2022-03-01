import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn import metrics

def var_perf(data, predictors, response, weight):
    '''
    This function determines the accuracy of different variables for predicting an
    outcome of interest.
    
    Parameters:
        data - data to analyze
        predictors - list of columns for predictors
        response - outcome column
        weight - column of weights 
    Returns:
        dataframe with results (n, confusion matrix, accuracy, sensitivity, and specificity)
    '''
    
    if weight == None:
        weight='wgt'
        data[weight]=1
        prnt_wgt=0
    else:
        prnt_wgt=1
    
    # create flag for any positive predictor
    data['pred'] = (data[predictors]==1).any(axis=1).astype(int)
    data['pred_inv'] = (~data.pred.astype(bool)).astype(int)
    data['resp'] = data[response]
    data['resp_inv'] = (~data[response].astype(bool)).astype(int)
    
    # create flag for confusion matrix
    data['tp'] = data.pred * data.resp
    data['tn'] = data.pred_inv * data.resp_inv
    data['fp'] = data.pred * data.resp_inv
    data['fn'] = data.pred_inv * data.resp
    data['correct'] = (data.tp + data.tn).astype(bool).astype(int)
        
    # calculate unweighted results
    acc = sum(data['correct']) / len(data)
    sens = sum(data['tp']) / sum(data.resp) 
    spec = sum(data['tn']) / sum(data.resp_inv)
    
    # calculate weighted results
    acc_wgt = sum(data['correct']*data[weight]) / sum(data[weight])
    sens_wgt = sum(data['tp']*data[weight]) / sum(data.resp*data[weight]) 
    spec_wgt = sum(data['tn']*data[weight]) / sum(data.resp_inv*data[weight]) 
    
    # store results   
    res = pd.DataFrame({'weighted':[prnt_wgt]})
    res['n'] = sum(data[weight])
    res['outcome'] = sum(data['tp']*data[weight]) + sum(data['fn']*data[weight])
    res['positivity'] = (sum(data['tp']*data[weight])+sum(data['fn']*data[weight]))/sum(data[weight])
    res['tp'] = sum(data['tp']*data[weight])
    res['tn'] = sum(data['tn']*data[weight])
    res['fp'] = sum(data['fp']*data[weight])
    res['fn'] = sum(data['fn']*data[weight])
    res['accuracy'] = acc_wgt
    res['sensitivity'] = sens_wgt
    res['specificity'] = spec_wgt
       
    return res


def vars_auc(data, predictors, response, weight, verbose=False):
    '''
    This function determines the cumulative sens and spec for a list of
    predictors.  The AUROC is determined and returns
    
    Parameters:
        data - data to analyze
        predictors - list of columns for predictors
        response - outcome column
        weight - column of weights 
        ages - list of ages
    Returns:
        auc, cum_perf, tpr, fpr - AUC for variable, dataframe with the cumlative performance
    '''
    
    # list of predictors currently used in model
    variables = []
    
    # empty df for sens/spec results
    results = pd.DataFrame()
    
    # loop through all predictors
    for i,var in enumerate(predictors):
        
        # add to list of current predictors
        variables.append(var)
        
        if verbose:
            print(variables)
    
        # get results for current iteration
        itr_result = var_perf(data, variables, response, weight)
        
        # record the number of variables
        itr_result['variables'] = i+1
        
        # store results
        results = results.append(itr_result)
        
    # calculate AUC-ROC
    tpr = results.sensitivity.values
    tpr = np.append(tpr, 1.0)
    tpr = np.insert(tpr, 0, 0.0)
    
    fpr = 1-results.specificity
    fpr = np.append(fpr, 1.0)
    fpr = np.insert(fpr, 0, 0.0)
    
    auc_res = metrics.auc(fpr, tpr)
    
    return auc_res, results.reset_index(drop=True), tpr, fpr


def auc_bootstrap(dat, predictors, response, sample_size, bs_num, verbose=False):
    '''
    This function runs bootstrapped AUC calculation and return a list of
    results.
    
    Parameters:
        dat - data to analyze
        predictors - list of columns for predictors
        response - outcome column
        sample_size - size of bootstrapped sample
        bs_num - number of bootstrap iterations
    Returns:
        list of AUC results for 
    '''
    
    # list for results
    auc_bs = []

    for i in range(0,bs_num):
        # sample with replacement
        sample = dat.sample(sample_size, replace=True)

        # calculate AUC
        auc, _, _, _ = vars_auc(sample, predictors, response, 'casewgt')

        if verbose:
            if (i%100==0):
                print("Sample: ", i, " of ", bs_num)
        
        # store results to list
        auc_bs += [auc]
        
    return auc_bs


def med_ci(vals, median=True, sig_dig=-1):
    '''
    This function calculates the median or mean and the 
    95% confidence interval for a list of results.  The 95% CI
    is determined by the interval that contains 95% of the results.
    This is calculated by removing the 2.5% low and high values.
    
    Parameters:
        vals - values to analyze
        median - True if median is calculated, False if mean is used
        sig_dig - significant digits (-1 if no rounding)
    Returns: 
        median/mean, list containing lower and upper bound of (95% CI)
    '''
    
    # calculate median or mean
    if median:
        med = np.median(vals)
    else:
        med = np.mean(vals)
    
    # get 95% CI
    ci = list(np.percentile(vals,[2.5,97.5]))
        
    if sig_dig != -1:
        med = round(med, sig_dig)
        ci[0] = round(ci[0], sig_dig)
        ci[1] = round(ci[1], sig_dig)
        
    return med,ci