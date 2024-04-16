import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from copy import deepcopy
from utils.loss_function import CSALoss, csa_loss
from utils.find_threshold import find_threshold, find_sensitivity, find_specificity
from sklearn.metrics import roc_auc_score

class CCSA(nn.Module):
    def __init__(self, feature_type, input_size, out1, out2, conv1, pool1, drop1, conv2, pool2, drop2, feature_fc1, feature_fc2, fc1, fc2, drop3):
        super(CCSA,self).__init__()        
       
        # Feature extractor p1
        self.feature_extractor_p1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=out1, kernel_size=conv1, stride=2, bias=None),
            nn.ReLU(),
            nn.BatchNorm1d(out1),
            nn.Dropout(drop1),
            nn.MaxPool1d(kernel_size=pool1, stride=2),
            
            nn.Conv1d(in_channels=out1, out_channels=out2, kernel_size=conv2, stride=2, bias=None),
            nn.ReLU(),
            nn.BatchNorm1d(out2),
            nn.Dropout(drop2),
            nn.MaxPool1d(kernel_size=pool2, stride=2)
        )
        
        self.fc_input_size = self._get_fc_input_size(input_size)
        self.feature_type = feature_type
        
        # Feature extractor p2
        self.feature_extractor_p2 = nn.Sequential(
            nn.Linear(self.fc_input_size, feature_fc1),
            nn.ReLU(True),
            nn.Linear(feature_fc1, feature_fc2)
        )        
        
        # Task classifier
        self.task_classifier = nn.Sequential(
            nn.Linear(feature_fc2, fc1),
            nn.ReLU(),
            nn.Dropout(drop3),
            nn.Linear(fc1, fc2),
            nn.Linear(fc2, 1),
            nn.Sigmoid()
        )
    
    def _get_fc_input_size(self, input_size):
        dummy_input = torch.randn(1, 1, input_size)
        x = self.feature_extractor_p1(dummy_input)
        flattened_size = x.size(1) * x.size(2)
        return flattened_size
    
    def forward(self, x):
        feature = self.feature_extractor_p1(x)
        feature = feature.view(feature.size(0),-1)
        feature = self.feature_extractor_p2(feature)
        
        pred = self.task_classifier(feature)       
        return pred.squeeze(1), feature
    
    def fit(self, loader, X_test_tensor, y_test_tensor, device, epoches, batch_patience, alpha, output_path):
                
        ### define loss function
        ce_loss = nn.BCELoss()
        self.optim = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay= 1e-6)
        # self.csa_loss = CSALoss()
        
        self.epoches = epoches
        self.alpha = alpha
        self.batch_patience = batch_patience
        # device = device
        patience = 10
        break_flag = False
        
        max_auc = 0.0
        min_loss = float('inf')
        best_model = None
        
        batch_without_improvement = 0
        epoches_without_improvement = 0
        
        loss_ce_list = []
        loss_csa_list = []
        loss_list = []
        
        # loss = model.train_on_batch([X1, X2], [y1, yc])
        for epoch in range(epoches):
            self.train()
            for i, (src_img, src_label, tgt_img, tgt_label) in enumerate(loader):
                
                src_img, tgt_img = (x.to(device, dtype=torch.float) for x in [src_img, tgt_img])
                src_label, tgt_label = (x.to(device, dtype=torch.float) for x in [src_label, tgt_label])
                
                src_pred, src_feature = self(src_img)
                _, tgt_feature = self(tgt_img)
                
                loss_ce  = ce_loss(src_pred, src_label)
                loss_csa = csa_loss(src_feature, tgt_feature,
                            (src_label == tgt_label).float())
                                                            
                loss = (1 - alpha) * loss_ce + alpha * loss_csa
                
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                if i % 100 == 0:
                    print(f"------------   Batch {i} loss: {loss.item():.4f}   ------------")

                loss_ce_list.append(loss_ce.item())
                loss_csa_list.append(loss_csa.item())
                loss_list.append(loss.item())
                
                if loss.item() <= min_loss:
                    min_loss = loss.item()
                    best_model = deepcopy(self.state_dict())
                    batch_without_improvement = 0
                else:
                    batch_without_improvement += 1
                    if batch_without_improvement >= batch_patience:
                        print(f"Early stopping triggered in Epoch {epoch}! No improvement in {batch_patience} batches.")
                        break_flag = True
                        break
                
            if break_flag:
                break
                
            with torch.no_grad():
                self.eval()
                src_img, src_label, tgt_img, tgt_label in loader
                
                # src_img, tgt_img = (x.to(device, dtype=torch.float) for x in [src_img, tgt_img])
                # src_label, tgt_label = (x.to(device, dtype=torch.float) for x in [src_label, tgt_label])
                src_img = src_img.to(device)
                tgt_img = tgt_img.to(device)
                src_label = src_label.to(device)
                tgt_label = tgt_label.to(device)
                
                X_test_tensor = X_test_tensor.to(device)
                y_test_tensor = y_test_tensor.to(device)
                
                src_pred, _ = self(src_img)
                tgt_pred, _ = self(tgt_img)
                test_pred, _ = self(X_test_tensor)
                
                src_auc = roc_auc_score(src_label.to("cpu"), src_pred.to("cpu"))
                tgt_auc = roc_auc_score(tgt_label.to("cpu"), tgt_pred.to("cpu"))
                test_auc = roc_auc_score(y_test_tensor.to("cpu"),test_pred.to("cpu"))
                
                thres95 = find_threshold(src_label.detach().cpu().numpy(), src_pred.detach().cpu().numpy(), 0.95)          
                tgt_sens = find_sensitivity(tgt_label.detach().cpu().numpy(),tgt_pred.detach().cpu().numpy(),thres95)
                tgt_spec = find_specificity(tgt_label.detach().cpu().numpy(),tgt_pred.detach().cpu().numpy(),thres95)
                
                print(f"==================== Epoch {epoch} =====================")
                print(f"Source AUC: {src_auc:.4f}, Target AUC: {tgt_auc:.4f}, Test AUC: {test_auc:.4f}")
                print(f"threshold 95: {thres95:.4f}, R01B sens: {tgt_sens:.4f}, target spec: {tgt_spec:.4f}")
                print("=========================================================")
                
                if src_auc >= max_auc:
                    max_auc = src_auc
                    best_model = deepcopy(self.state_dict())
                    epoches_without_improvement = 0
                else:
                    epoches_without_improvement += 1
                    if epoches_without_improvement >= patience:
                        print(f"Early stopping triggered in Epoch {epoch}! No improvement in {patience} epoches.")
                        break
        
        self.load_state_dict(best_model)    
        torch.save(self, f"{output_path}/{self.feature_type}_CCSA.pt")
        
        # Plotting the loss values
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(loss_ce_list) + 1), loss_ce_list, label='Loss CE')
        plt.plot(range(1, len(loss_csa_list) + 1), loss_csa_list, label='Loss CSA')
        plt.plot(range(1, len(loss_list) + 1), loss_list, label='Loss TOTAL')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Losses over epoches')
        plt.legend()
        plt.grid(True)
        plt.show()
            
        return loss.item()
    
    def reset_parameters(self):
        for module in self.children():
            ### some layers do not have parameters to reset
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
        
    

    


