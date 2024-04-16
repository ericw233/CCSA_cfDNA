from sklearn.metrics import roc_curve
import numpy as np
import pandas as pd

# calculate the threshold at a given specificity

def find_threshold(response, score, target_specificity):
    fpr, tpr, thresholds = roc_curve(response, score)

    specificity = 1 - fpr
    
    if target_specificity > max(specificity):
        raise ValueError("Target specificity is not achievable with the given data.")

    thres_df = pd.DataFrame({'thres':thresholds, 'spec':specificity}).sort_values(by='spec',ascending=True, ignore_index=True)
    target_thres = thres_df.loc[thres_df['spec'] >= target_specificity,"thres"].values[0]
    return target_thres

def find_sensitivity(response,score,threshold):
    
    score_pos = score[response == 1]
    sensitivity = np.mean(score_pos >= threshold)
    
    return sensitivity

def find_specificity(response,score,threshold):
    
    score_neg = score[response == 0]
    specificity = np.mean(score_neg < threshold)
    
    return specificity

def find_sensitivity(response,score,threshold):
    
    score_pos = score[response == 1]
    sensitivity = np.mean(score_pos >= threshold)
    
    return sensitivity

def find_sensandspec(response,score,threshold):
    
    score_neg = score[response == 0]
    specificity = np.mean(score_neg < threshold)
    
    score_pos = score[response == 1]
    sensitivity = np.mean(score_pos >= threshold)
    
    return sensitivity,specificity