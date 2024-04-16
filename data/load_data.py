import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

from utils.pad_and_reshape import pad_and_reshape, pad_and_reshape_1D

def load_data_1D_impute(data_dir="/mnt/binf/eric/Mercury_Dec2023/Feature_all_Dec2023_revised2.pkl", input_size=900, feature_type = "Arm"):
    # Read data from CSV file
    
    if(data_dir.endswith('.csv')):
        data = pd.read_csv(data_dir)
    elif(data_dir.endswith('.pkl')):
        data = pd.read_pickle(data_dir)

    # keep a full dataset without shuffling
    mapping = {'Healthy':0,'Cancer':1}
    
    # Split the data into train and test sets
    # X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    X_train = data.loc[data["train"] == "training"].filter(regex = feature_type, axis=1)
    y_train = (data.loc[data["train"] == "training","Train_Group"] == "Cancer").astype(int)
    d_train = data.loc[data["train"] == "training","Domain"]
    
    X_test = data.loc[data["train"] == "validation"].filter(regex = feature_type, axis=1)
    y_test = (data.loc[data["train"] == "validation","Train_Group"] == "Cancer").astype(int)
    d_test = data.loc[data["train"] == "validation","Domain"]
    
    X_all = data.filter(regex = feature_type, axis=1)
    y_all = (data.loc[:,'Train_Group']== "Cancer").astype(int)
    d_all = data.loc[:,'Domain']
    
    X_r01b = data.loc[data["R01B_label"] == "R01B_match"].filter(regex = feature_type, axis=1)
    y_r01b = (data.loc[data["R01B_label"] == "R01B_match","Train_Group"] == "Cancer").astype(int)

    #### drop constant NA columns based on X_train
    na_columns = X_train.columns[X_train.isna().all()]
    X_train_drop = X_train.drop(columns = na_columns)
    X_test_drop = X_test.drop(columns = na_columns)
    X_all_drop = X_all.drop(columns = na_columns)
    X_r01b_drop = X_r01b.drop(columns = na_columns)
    
    #### impute variables based on X_train_drop
    mean_imputer = SimpleImputer(strategy = 'mean')
    X_train_drop_imputed = mean_imputer.fit_transform(X_train_drop)
    X_test_drop_imputed = mean_imputer.transform(X_test_drop)
    X_all_drop_imputed = mean_imputer.transform(X_all_drop)
    X_r01b_drop_imputed = mean_imputer.transform(X_r01b_drop)
    
    # Scale the features to a suitable range (e.g., [0, 1])
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_drop_imputed)
    X_test_scaled = scaler.transform(X_test_drop_imputed)
    X_all_scaled = scaler.transform(X_all_drop_imputed)
    X_r01b_scaled = scaler.transform(X_r01b_drop_imputed)

    # Convert the data to PyTorch tensors
    input_size = input_size
    X_train_tensor = pad_and_reshape_1D(X_train_scaled, input_size).type(torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    d_train_tensor = torch.tensor(d_train.values, dtype=torch.float32)
    
    X_test_tensor = pad_and_reshape_1D(X_test_scaled, input_size).type(torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)
    d_test_tensor = torch.tensor(d_test.values, dtype=torch.float32)

    X_all_tensor = pad_and_reshape_1D(X_all_scaled, input_size).type(torch.float32)
    y_all_tensor = torch.tensor(y_all.values, dtype=torch.float32)
    d_all_tensor = torch.tensor(d_all.values, dtype=torch.float32)

    X_r01b_tensor = pad_and_reshape_1D(X_r01b_scaled, input_size).type(torch.float32)
    y_r01b_tensor = torch.tensor(y_r01b.values, dtype=torch.float32)
    
    ### keep unshuffled X_train
    # X_train_tensor_unshuffled = pad_and_reshape_1D(X_train_scaled, input_size).type(torch.float32)
    # y_train_tensor_unshuffled = torch.tensor(y_train.values, dtype=torch.float32)
    train_sampleid = data.loc[data["train"] == "training","SampleID"].values

    return data, X_train_tensor, y_train_tensor, d_train_tensor, X_test_tensor, y_test_tensor, d_test_tensor, X_all_tensor, y_all_tensor, d_all_tensor, X_r01b_tensor, y_r01b_tensor, train_sampleid

