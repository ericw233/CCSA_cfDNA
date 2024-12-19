import os
import sys
import json
import torch
import pandas as pd
from copy import deepcopy
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from CCSA_cfDNA.model.model import CCSA
from CCSA_cfDNA.utils.find_threshold import find_threshold, find_sensandspec, find_sensitivity
from CCSA_cfDNA.data.load_dataset import TrainSet, TestSet
from CCSA_cfDNA.training.fit_model import fit_CCSA
from CCSA_cfDNA.training.cv_model import cv_CCSA

def parse_arguments():
    # get arguments from command lines or use default values
    defaults = {
        "feature_type": "Frag",
        "input_size": 1100,
        "data_dir": "/mnt/binf/eric/Mercury_June2024_MGI_new/Feature_June2024_T7MGI2000_CCSA_v2.csv",
        "output_path": "/mnt/binf/eric/CCSA_June2024Results/CCSA_0627/",
        "batch_size": 256,
        "epoch_num": 3,
        "batch_patience": 500,
        "alpha": 0.5,
    }

    if len(sys.argv) >= 9:
        keys = ["feature_type", "input_size", "data_dir", "output_path", "batch_size", "epoch_num", "batch_patience", "alpha"]
        args = dict(zip(keys, sys.argv[1:]))
        args["input_size"] = int(args["input_size"])
        args["batch_size"] = int(args["batch_size"])
        args["epoch_num"] = int(args["epoch_num"])
        args["batch_patience"] = int(args["batch_patience"])
        args["alpha"] = float(args["alpha"])
    else:
        args = defaults
        print("Using default arguments:", defaults)

    os.makedirs(args["output_path"], exist_ok=True)
    return args

def load_params(output_path, feature_type, default_params):
    
    # provide json config files for hyperparameter settings; otherwise use default values.
    param_file = f"{output_path}/{feature_type}_parameters.json"
    if os.path.exists(param_file):
        with open(param_file, "r") as file:
            params = json.load(file)
            params.update({"feature_type": feature_type})
            params = {k: params[k] for k in default_params if k in params} # only update keys existing in default_params.
    else:
        params = default_params

    # with open(config_file, "w") as file:
    #     json.dump(config, file)

    return params

def prepare_datasets(data_dir, input_size, feature_type, batch_size):
    # Prepare training and testing datasets. In TrainSet, 'train' subset is the src_domain, 'valid' subset is the tgt_domain.
    train_set = TrainSet(data_dir=data_dir, input_size=input_size, feature_type=feature_type)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)

    # Testset is the independent one for validation.
    test_set = TestSet(data_dir=data_dir, input_size=input_size, feature_type=feature_type)
    testing_index = test_set.data_idonly.loc[test_set.data_idonly["train"] == "testing"].index

    return train_loader, test_set, testing_index

def evaluate_model(model, test_set, testing_index, output_path, feature_type):
    # Evaluate the model and save results.
    device = next(model.parameters()).device

    with torch.no_grad():
        model.eval()
        pred_all, feature_all = model(test_set.X_test.to(device))

    data_idonly = test_set.data_idonly
    data_idonly["CCSA_score"] = pred_all.detach().cpu().numpy()
    data_idonly["Response"] = (data_idonly["Train_Group"] == "Cancer").astype(int)
    data_idonly.to_csv(f"{output_path}/{feature_type}_CCSA_score.csv")

    colnames = [f"CCSA_{feature_type}_{i}" for i in range(feature_all.size(1))]
    feature_df = pd.DataFrame(feature_all.detach().cpu().numpy(), columns=colnames)
    feature_df_bind = pd.concat([data_idonly[["SampleID", "Train_Group", "train"]], feature_df], axis=1)
    feature_df_bind.to_csv(f"{output_path}/{feature_type}_CCSA_feature.csv")

    return data_idonly

def compute_metrics(data_idonly):
    
    train_data = data_idonly[data_idonly["train"] == "training"]
    test_data = data_idonly[data_idonly["train"] == "testing"]
    valid_data = data_idonly[data_idonly["train"] == "validation"]

    thres95 = find_threshold(train_data["Response"], train_data["CCSA_score"], 0.95)
    thres98 = find_threshold(train_data["Response"], train_data["CCSA_score"], 0.98)

    metrics = {
        "valid": find_sensandspec(test_data["Response"], test_data["CCSA_score"], thres95),
        "target": find_sensandspec(valid_data["Response"], valid_data["CCSA_score"], thres95),
    }

    print("Validation Metrics:", metrics["valid"])
    print("Target Metrics:", metrics["target"])

def main():
    args = parse_arguments()

    default_params = {
        "feature_type": args["feature_type"],
        "input_size": args["input_size"],
        "out1": 32,
        "out2": 128,
        "conv1": 3,
        "pool1": 2,
        "drop1": 0.2,
        "conv2": 3,
        "pool2": 1,
        "drop2": 0.4,
        "fc1": 64,
        "fc2": 16,
        "drop3": 0.2,
        "feature_fc1": 256,
        "feature_fc2": 64,
    }

    params = load_params(args["output_path"], args["feature_type"], default_params)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CCSA(**params).to(device)
    init_state = deepcopy(model.state_dict())

    train_loader, test_set, testing_index = prepare_datasets(
        args["data_dir"], args["input_size"], args["feature_type"], args["batch_size"]
    )

    model.load_state_dict(init_state)
    fit_CCSA(
        model,
        train_loader,
        test_set.X_test[testing_index],
        test_set.y_test[testing_index],
        device,
        args["feature_type"],
        args["epoch_num"],
        args["batch_patience"],
        args["alpha"],
        args["output_path"],
    )

    data_idonly = evaluate_model(model, test_set, testing_index, args["output_path"], args["feature_type"])
    compute_metrics(data_idonly)

    cv_output_path = os.path.join(args["output_path"], "cv")
    os.makedirs(cv_output_path, exist_ok=True)
    cv_CCSA(
        model,
        train_loader,
        device,
        args["feature_type"],
        args["batch_size"],
        args["epoch_num"],
        args["batch_patience"],
        args["alpha"],
        cv_output_path,
    )

if __name__ == "__main__":
    main()
