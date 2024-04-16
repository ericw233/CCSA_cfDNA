import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from data.load_data import load_data_1D_impute

# create pairs
class TrainSet(Dataset):
    def __init__(self, data_dir, input_size, feature_type):

        data, X_train_tensor, y_train_tensor, _, X_valid_tensor, y_valid_tensor, _, _, _, _, _, _, _ = load_data_1D_impute(data_dir, input_size, feature_type) 

        self.X_source=X_train_tensor
        self.y_source=y_train_tensor
        self.X_target=X_valid_tensor
        self.y_target=y_valid_tensor
        
        self.sampleid_source=data.loc[data["train"]=="training","SampleID"]
        self.sampleid_target=data.loc[data["train"]=="validation","SampleID"]
        
        print("Source X : ", X_train_tensor.size(0), " Y : ", y_train_tensor.size(0))
        print("Target X : ", X_valid_tensor.size(0), " Y : ", y_valid_tensor.size(0))
                
        Training_P=[]
        Training_N=[]
        for trs in range(self.y_source.size(0)):
            for trt in range(self.y_target.size(0)):
                if self.y_source[trs] == self.y_target[trt]:
                    Training_P.append([trs,trt, 1]) # each element is a list of 3, two indexes and one eq number
                else:
                    Training_N.append([trs,trt, 0])
        print("Class P : ", len(Training_P), " N : ", len(Training_N))
        
        random.shuffle(Training_N)
        self.imgs = Training_P+Training_N[:len(Training_P)]
        random.shuffle(self.imgs)

    def __getitem__(self, idx):
        src_idx, tgt_idx, eq = self.imgs[idx]

        x_src, y_src = self.X_source[src_idx], self.y_source[src_idx]
        x_tgt, y_tgt = self.X_target[tgt_idx], self.y_target[tgt_idx]
        return x_src, y_src, x_tgt, y_tgt

    def __len__(self):
        return len(self.imgs)


class TestSet(Dataset):
    def __init__(self, data_dir, input_size, feature_type):
        
        ### TestSet is different from TrainSet, does not return pairwise images
        
        data, _, _, _, _, _, _, X_all_tensor, y_all_tensor, _, _, _, _ = load_data_1D_impute(data_dir, input_size, feature_type) 
        self.X_test = X_all_tensor
        self.y_test = y_all_tensor        
        self.data_idonly=data[["SampleID","Train_Group","train","Project","Domain","R01B_label"]]
        
    def __getitem__(self, idx):
        X, y, Project = self.X_test[idx], self.y_test[idx], self.data_idonly['Project'].values[idx]
        return X, y, Project

    def __len__(self):
        return self.X_test.size(0)
    

class MyDataset(Dataset):
    def __init__(self, X_source, y_source, X_target, y_target):

        self.X_source=X_source
        self.y_source=y_source
        self.X_target=X_target
        self.y_target=y_target

        print("Source X : ", X_source.size(0), " Y : ", y_source.size(0))
        print("Target X : ", X_target.size(0), " Y : ", y_target.size(0))

        Training_P=[]
        Training_N=[]
        for trs in range(self.y_source.size(0)):
            for trt in range(self.y_target.size(0)):
                if self.y_source[trs] == self.y_target[trt]:
                    Training_P.append([trs,trt, 1]) # each element is a list of 3, two indexes and one eq number
                else:
                    Training_N.append([trs,trt, 0])
        print("Class P : ", len(Training_P), " N : ", len(Training_N))

        random.shuffle(Training_N)
        self.imgs = Training_P+Training_N[:len(Training_P)]
        random.shuffle(self.imgs)

    def __getitem__(self, idx):
        src_idx, tgt_idx, eq = self.imgs[idx]

        x_src, y_src = self.X_source[src_idx], self.y_source[src_idx]
        x_tgt, y_tgt = self.X_target[tgt_idx], self.y_target[tgt_idx]
        return x_src, y_src, x_tgt, y_tgt

    def __len__(self):
        return len(self.imgs)
        


    
