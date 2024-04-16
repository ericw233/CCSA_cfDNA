import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.metrics import roc_auc_score
from copy import deepcopy

from torch.utils.data import DataLoader
from model import CCSA
from utils.find_threshold import find_threshold, find_sensandspec, find_sensitivity
from data.load_dataset import TrainSet, TestSet, MyDataset
from fit_model import fit_CCSA
from cv_model import cv_CCSA


batch_size = 256
epoch_num = 5
alpha = 0.5

feature_type = "Frag"
input_size = 1100
data_dir="/mnt/binf/eric/Mercury_Dec2023/Feature_all_Apr2024_frozenassource.pkl"

batch_patience = 2000
output_path = "/mnt/binf/eric/CCSA_test_0415/"

if os.path.exists(output_path):
    print("Output directory already exists")
else:
    os.makedirs(output_path)
    print("Output directory created")

##### define model
# best_config={'input_size':input_size,
#              'out1': 32, 'out2': 128, 'conv1': 3, 'pool1': 2, 'drop1': 0.0, 
#              'conv2': 4, 'pool2': 1, 'drop2': 0.4, 
#              'feature_fc1': 256, 'feature_fc2': 128,
#              'fc1': 64, 'fc2': 16, 'drop3': 0.2}

best_config={'feature_type':feature_type,'input_size':input_size,
             'out1': 32, 'out2': 128, 'conv1': 3, 'pool1': 2, 'drop1': 0.2, 
             'conv2': 4, 'pool2': 1, 'drop2': 0.4, 
             'feature_fc1': 256, 'feature_fc2': 128,
             'fc1': 64, 'fc2': 16, 'drop3': 0.2}

device = torch.device("cuda")
model = CCSA(**best_config)
model = model.to(device)

####### prepare dataset
train_set = TrainSet(data_dir=data_dir, input_size=input_size, feature_type=feature_type)
train_set_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)

test_set = TestSet(data_dir=data_dir, input_size=input_size, feature_type=feature_type)
# test_set_loader = DataLoader(test_set, batch_size=batch, shuffle=True, drop_last=True)
# tensors in the test_set class are actually X_all_tensor and y_all_tensor
testing_index = test_set.data_idonly.loc[test_set.data_idonly["train"] == "testing"].index
X_test_tensor = test_set.X_test[testing_index]
y_test_tensor = test_set.y_test[testing_index]

print("Dataset Length Train : ", len(train_set), " Test : ", len(test_set))

################ fitting model
fit_CCSA(model, train_set_loader,X_test_tensor,y_test_tensor,device,feature_type,epoch_num,batch_patience,alpha,output_path)
print("======================   complete fitting the {feature_type} model!  =======================")

### GPU outage solution
# model.to("cpu")
# pred_all, feature_all = model(test_set.X_test)

### GPU 
with torch.no_grad():
    model.eval()
    pred_all, feature_all = model(test_set.X_test.to(device))

data_idonly = test_set.data_idonly
data_idonly["CCSA_score"] = pred_all.detach().cpu().numpy()
data_idonly["Response"] = (data_idonly["Train_Group"] == "Cancer").astype(int)
data_idonly.to_csv(f"{output_path}/{feature_type}_CCSA_score.csv")

colnames=['CCSA_'+feature_type+'_'+str(i) for i in range(0,best_config["feature_fc2"])]
feature_all_df = pd.DataFrame(feature_all.detach().cpu().numpy(),columns=colnames)
feature_all_df_bind = pd.concat([data_idonly.loc[:,["SampleID","Train_Group","train"]],feature_all_df],axis=1)
feature_all_df_bind.to_csv(f"{output_path}/{feature_type}_CCSA_feature.csv")

#### get R01B sens
thres95 = find_threshold(data_idonly.loc[data_idonly['train']=="training",'Response'],data_idonly.loc[data_idonly['train']=="training",'CCSA_score'],0.95)
thres98 = find_threshold(data_idonly.loc[data_idonly['train']=="training",'Response'],data_idonly.loc[data_idonly['train']=="training",'CCSA_score'],0.98)

data_idonly_test = data_idonly.loc[data_idonly["train"] == "testing"]

train_auc = roc_auc_score(data_idonly.loc[data_idonly['train']=="training",'Response'],data_idonly.loc[data_idonly['train']=="training",'CCSA_score'])
valid_auc = roc_auc_score(data_idonly.loc[data_idonly['train']=="testing",'Response'],data_idonly.loc[data_idonly['train']=="testing",'CCSA_score'])
tgt_auc = roc_auc_score(data_idonly.loc[data_idonly['train']=="validation",'Response'],data_idonly.loc[data_idonly['train']=="validation",'CCSA_score'])

valid_sens95, valid_spec95 = find_sensandspec(data_idonly.loc[data_idonly['train']=="testing",'Response'],data_idonly.loc[data_idonly['train']=="testing",'CCSA_score'],thres95)
valid_sens98, valid_spec98 = find_sensandspec(data_idonly.loc[data_idonly['train']=="testing",'Response'],data_idonly.loc[data_idonly['train']=="testing",'CCSA_score'],thres98)

tgt_sens95, tgt_spec95 = find_sensandspec(data_idonly.loc[data_idonly['train']=="validation",'Response'],data_idonly.loc[data_idonly['train']=="validation",'CCSA_score'],thres95)
tgt_sens98, tgt_spec98 = find_sensandspec(data_idonly.loc[data_idonly['train']=="validation",'Response'],data_idonly.loc[data_idonly['train']=="validation",'CCSA_score'],thres98)

print(f"Validation AUC: {valid_auc:.4f}, sens95: {valid_sens95:.4f}, spec95: {valid_spec95:.4f}, sens98: {valid_sens98:.4f}, spec98: {valid_spec98:.4f}")
print(f"Target AUC: {tgt_auc:.4f}, sens95: {tgt_sens95:.4f}, spec95: {tgt_spec95:.4f}, sens98: {tgt_sens98:.4f}, spec98: {tgt_spec98:.4f}")

#### R01B detection
R01B_sens95 = find_sensitivity(data_idonly.loc[data_idonly['Project'].str.contains("R01B"),'Response'],data_idonly.loc[data_idonly['Project'].str.contains("R01B"),'CCSA_score'],thres95)
R01B_sens98 = find_sensitivity(data_idonly.loc[data_idonly['Project'].str.contains("R01B"),'Response'],data_idonly.loc[data_idonly['Project'].str.contains("R01B"),'CCSA_score'],thres98)

R01B_sens95_test = find_sensitivity(data_idonly_test.loc[data_idonly_test['Project'].str.contains("R01B"),'Response'],data_idonly_test.loc[data_idonly_test['Project'].str.contains("R01B"),'CCSA_score'],thres95)
R01B_sens98_test = find_sensitivity(data_idonly_test.loc[data_idonly_test['Project'].str.contains("R01B"),'Response'],data_idonly_test.loc[data_idonly_test['Project'].str.contains("R01B"),'CCSA_score'],thres98)

print(f"Threshold - spec95: {thres95:.4f}, spec98: {thres98:.4f} ")
print(f"R01B sens 95 - all: {R01B_sens95*161:.1f}, test: {R01B_sens95_test*111:.1f}, train: {(R01B_sens95*161 - R01B_sens95_test*111):.1f}")
print(f"R01B sens 98 - all: {R01B_sens98*161:.1f}, test: {R01B_sens98_test*111:.1f}, train: {(R01B_sens98*161 - R01B_sens98_test*111):.1f}")

############## Generate CV scores and features 
output_path_cv = output_path+"/cv/"
if os.path.exists(output_path_cv):
    print("CV output directory already exists")
else:
    os.makedirs(output_path_cv)
    print("CV output directory created")

cv_CCSA(model, train_set_loader, device, feature_type, batch_size, epoch_num, batch_patience, alpha, output_path_cv)

