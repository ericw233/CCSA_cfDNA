import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from copy import deepcopy
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from data.load_dataset import MyDataset
from utils.loss_function import csa_loss
from utils.find_threshold import find_threshold, find_sensandspec
from fit_model import fit_CCSA

def cv_CCSA(model, loader, device, feature_type, batch_size, epoches, batch_patience, alpha, output_path):
        
        init_state = deepcopy(model.state_dict())
        
        ### split source and target datasets
        fold_src = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
        fold_tgt = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
        
        X_src = loader.dataset.X_source
        y_src = loader.dataset.y_source
        X_tgt = loader.dataset.X_target
        y_tgt = loader.dataset.y_target
        
        sampleid_src = loader.dataset.sampleid_source.reset_index(drop=True)
        sampleid_tgt = loader.dataset.sampleid_target.reset_index(drop=True)
        
        iteration_src = fold_src.split(np.zeros(X_src.size(0)), y_src)
        iteration_tgt = fold_tgt.split(np.zeros(X_tgt.size(0)), y_tgt)   
        
        score_full_df = pd.DataFrame()
        feature_full_df = pd.DataFrame()
        
        for (foldi_src, (index_trainfold_src, index_validfold_src)),(foldi_tgt,(index_trainfold_tgt, index_validfold_tgt)) in zip(enumerate(iteration_src), enumerate(iteration_tgt)):
            
            trainfold_dataset = MyDataset(X_source=X_src[index_trainfold_src],y_source=y_src[index_trainfold_src],X_target=X_tgt[index_trainfold_tgt],y_target=y_tgt[index_trainfold_tgt])
            trainfold_loader = DataLoader(trainfold_dataset,batch_size=batch_size, shuffle=True, drop_last=True)
            # X_validfold = torch.cat((X_src[index_validfold_src],X_tgt[tgt_index_validfold]),dim=0)
            # y_validfold = torch.cat((y_src[index_validfold_src],y_tgt[tgt_index_validfold]),dim=0)
                        
            sampleid_validfold = pd.concat([sampleid_src[index_validfold_src],sampleid_tgt[index_validfold_tgt]]).tolist()
            
            print(f"------------- Start fitting source fold: {foldi_src} and target fold: {foldi_tgt} -------------")
            model.reset_parameters()
            model.load_state_dict(init_state)
            
            # output_path_i = output_path+"/"+str(foldi_src)
            
            fit_CCSA(model, trainfold_loader,X_src[index_validfold_src],y_src[index_validfold_src],device,feature_type,epoches,batch_patience,alpha,output_path)

            with torch.no_grad():
                model.eval()
                pred_validfold_src, feature_validfold_src = model(X_src[index_validfold_src].to(device))
                pred_validfold_tgt, feature_validfold_tgt = model(X_tgt[index_validfold_tgt].to(device))
                pred_trainfold_src, _ = model(X_src[index_trainfold_src].to(device))
                
                ### export prediction scores and feature dataframes
                pred_validfold_src = pred_validfold_src.detach().cpu().numpy()
                pred_validfold_tgt = pred_validfold_tgt.detach().cpu().numpy()
                pred_trainfold_src = pred_trainfold_src.detach().cpu().numpy()
                
                colnames=['CCSA_'+feature_type+'_'+str(i) for i in range(0,feature_validfold_src.size(1))]
                feature_df_validfold_src = pd.DataFrame(feature_validfold_src.detach().cpu().numpy(),columns=colnames)
                feature_df_validfold_tgt = pd.DataFrame(feature_validfold_tgt.detach().cpu().numpy(),columns=colnames)
                feature_df_validfold = pd.concat([feature_df_validfold_src, feature_df_validfold_tgt],axis=0)
                feature_df_validfold["SampleID"] = sampleid_validfold
                feature_df_validfold["Fold"] = foldi_src
                feature_df_validfold.to_csv(f"{output_path}/{feature_type}_CCSA_feature_fold{foldi_src}.csv",index=False)
                
                score_foldi = pd.DataFrame()
                score_foldi["SampleID"] = sampleid_validfold
                score_foldi["Fold"] = foldi_src
                score_foldi["CCSA_score"] = np.concatenate((pred_validfold_src, pred_validfold_tgt), axis=0)
                score_foldi.to_csv(f"{output_path}/{feature_type}_CCSA_score_fold{foldi_src}.csv",index=False)
                
                feature_full_df = pd.concat([feature_full_df, feature_df_validfold], axis=0)
                score_full_df = pd.concat([score_full_df, score_foldi],axis=0) 
                
                ### calculate valid fold metrics
                auc_src = roc_auc_score(y_src[index_validfold_src], pred_validfold_src)
                auc_tgt = roc_auc_score(y_tgt[index_validfold_tgt], pred_validfold_tgt)
                auc_src_trainfold = roc_auc_score(y_src[index_trainfold_src], pred_trainfold_src)
                
                thres98 = find_threshold(y_src[index_trainfold_src],pred_trainfold_src,0.98)
                sens_src, spec_src = find_sensandspec(y_src[index_validfold_src], pred_validfold_src,thres98)
                sens_tgt, spec_tgt = find_sensandspec(y_tgt[index_validfold_tgt], pred_validfold_tgt,thres98)
                
                print(f"------------ Fold: {foldi_src} metrics ---------------")
                print(f"Trainfold source AUC: {auc_src_trainfold:.4f}, threshold98: {thres98:.4f}")
                print(f"Source AUC: {auc_src:.4f}, sensitivity: {sens_src:.4f}, specificity: {spec_src:.4f}")
                print(f"Target AUC: {auc_tgt:.4f}, sensitivity: {sens_tgt:.4f}, specificity: {spec_tgt:.4f}")
                print("--------------------------------------------------")
                
        print("------- Complete CV fitting --------")
        feature_full_df.to_csv(f"{output_path}/{feature_type}_CCSA_feature_CV.csv",index=False)        
        score_full_df.to_csv(f"{output_path}/{feature_type}_CCSA_score_CV.csv",index=False)

        
                
        
        
        
                
                
                
                
                
                
                
                
                

                
                
                