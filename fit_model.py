import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from copy import deepcopy
import matplotlib.pyplot as plt

from utils.loss_function import csa_loss
from utils.find_threshold import find_threshold, find_sensandspec

def fit_CCSA(model, loader, X_test_tensor, y_test_tensor, device, feature_type, epoches, batch_patience, alpha, output_path):
                
        ### define loss function
        ce_loss = nn.BCELoss()
        optim = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay= 1e-6)
        # model.csa_loss = CSALoss()
        # model.epoches = epoches
        # model.alpha = alpha
        # model.batch_patience = batch_patience
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
            model.train()
            for i, (src_img, src_label, tgt_img, tgt_label) in enumerate(loader):
                
                src_img, tgt_img = (x.to(device, dtype=torch.float) for x in [src_img, tgt_img])
                src_label, tgt_label = (x.to(device, dtype=torch.float) for x in [src_label, tgt_label])
                
                src_pred, src_feature = model(src_img)
                _, tgt_feature = model(tgt_img)
                
                loss_ce  = ce_loss(src_pred, src_label)
                loss_csa = csa_loss(src_feature, tgt_feature,
                            (src_label == tgt_label).float())
                                                            
                loss = (1 - alpha) * loss_ce + alpha * loss_csa
                
                optim.zero_grad()
                loss.backward()
                optim.step()
                if i % 100 == 0:
                    print(f"------------   Batch {i} loss: {loss.item():.4f}   ------------")

                loss_ce_list.append(loss_ce.item())
                loss_csa_list.append(loss_csa.item())
                loss_list.append(loss.item())
                
                if loss.item() <= min_loss:
                    min_loss = loss.item()
                    best_model = deepcopy(model.state_dict())
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
                model.eval()
                src_img, src_label, tgt_img, tgt_label in loader
                
                src_img = loader.dataset.X_source
                src_label = loader.dataset.y_source
                tgt_img = loader.dataset.X_target
                tgt_label = loader.dataset.y_target
                
                # src_img, tgt_img = (x.to(device, dtype=torch.float) for x in [src_img, tgt_img])
                # src_label, tgt_label = (x.to(device, dtype=torch.float) for x in [src_label, tgt_label])
                src_img = src_img.to(device)
                tgt_img = tgt_img.to(device)
                src_label = src_label.to(device)
                tgt_label = tgt_label.to(device)
                
                X_test_tensor = X_test_tensor.to(device)
                y_test_tensor = y_test_tensor.to(device)
                
                src_pred, _ = model(src_img)
                tgt_pred, _ = model(tgt_img)
                test_pred, _ = model(X_test_tensor)
                
                src_auc = roc_auc_score(src_label.to("cpu"), src_pred.to("cpu"))
                tgt_auc = roc_auc_score(tgt_label.to("cpu"), tgt_pred.to("cpu"))
                test_auc = roc_auc_score(y_test_tensor.to("cpu"),test_pred.to("cpu"))
                
                thres95 = find_threshold(src_label.detach().cpu().numpy(), src_pred.detach().cpu().numpy(), 0.95)          
                tgt_sens, tgt_spec = find_sensandspec(tgt_label.detach().cpu().numpy(),tgt_pred.detach().cpu().numpy(),thres95)
                
                print(f"==================== Epoch {epoch} =====================")
                print(f"Source AUC: {src_auc:.4f}, Target AUC: {tgt_auc:.4f}, Test AUC: {test_auc:.4f}")
                print(f"threshold 95: {thres95:.4f}, target sens: {tgt_sens:.4f}, target spec: {tgt_spec:.4f}")
                print("=========================================================")
                
                if src_auc >= max_auc:
                    max_auc = src_auc
                    best_model = deepcopy(model.state_dict())
                    epoches_without_improvement = 0
                else:
                    epoches_without_improvement += 1
                    if epoches_without_improvement >= patience:
                        print(f"Early stopping triggered in Epoch {epoch}! No improvement in {patience} epoches.")
                        break
        
        model.load_state_dict(best_model)    
        torch.save(model, f"{output_path}/{feature_type}_CCSA.pt")
        
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