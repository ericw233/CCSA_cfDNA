import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from ..utils.pad_and_reshape import pad_and_reshape_1D

class DropAllNA(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.columns_to_drop = X.columns[X.isna().all()].tolist()
        return self
    def transform(self, X, y=None):
        return X.drop(columns=self.columns_to_drop)       

def create_feature_transformer():
    transformer = Pipeline(steps=[('drop_na',DropAllNA()),('imputer',SimpleImputer(strategy='mean')),('scaler',MinMaxScaler())])
    return transformer

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

    # Create the feature transformer pipeline
    feature_transformer = create_feature_transformer()

    # Fit and transform the training data, and transform the other datasets
    X_train_transformed = feature_transformer.fit_transform(X_train)
    X_test_transformed = feature_transformer.transform(X_test)
    X_all_transformed = feature_transformer.transform(X_all)
    X_r01b_transformed = feature_transformer.transform(X_r01b)

    # Convert the data to PyTorch tensors
    X_train_tensor = pad_and_reshape_1D(X_train_transformed, input_size).type(torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    d_train_tensor = torch.tensor(d_train.values, dtype=torch.float32)
    
    X_test_tensor = pad_and_reshape_1D(X_test_transformed, input_size).type(torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)
    d_test_tensor = torch.tensor(d_test.values, dtype=torch.float32)

    X_all_tensor = pad_and_reshape_1D(X_all_transformed, input_size).type(torch.float32)
    y_all_tensor = torch.tensor(y_all.values, dtype=torch.float32)
    d_all_tensor = torch.tensor(d_all.values, dtype=torch.float32)

    X_r01b_tensor = pad_and_reshape_1D(X_r01b_transformed, input_size).type(torch.float32)
    y_r01b_tensor = torch.tensor(y_r01b.values, dtype=torch.float32)
    
    ### keep unshuffled X_train
    # X_train_tensor_unshuffled = pad_and_reshape_1D(X_train_scaled, input_size).type(torch.float32)
    # y_train_tensor_unshuffled = torch.tensor(y_train.values, dtype=torch.float32)
    train_sampleid = data.loc[data["train"] == "training","SampleID"].values
    data_idonly = data.loc[:,["SampleID","train","Train_Group","Project"]]
    data_idonly_train = data.loc[data["train"] == "training",["SampleID","train","Train_Group","Project","Domain","R01B_label"]].reset_index()
 
    return (data, data_idonly_train, X_train_tensor, y_train_tensor, d_train_tensor, X_test_tensor, y_test_tensor, d_test_tensor, 
            X_all_tensor, y_all_tensor, d_all_tensor, X_r01b_tensor, y_r01b_tensor, train_sampleid, 
            feature_transformer)

