import torch
import pandas as pd
import re
import os
from CCSA_cfDNA.data.load_data import load_data_1D_impute
from CCSA_cfDNA.utils.pad_and_reshape import pad_and_reshape_1D

def load_new_data(data_path):
    # Load data from a CSV or pickle file.
    if data_path.endswith('.csv'):
        return pd.read_csv(data_path)
    elif data_path.endswith('.pkl'):
        return pd.read_pickle(data_path)
    else:
        raise ValueError("Unsupported file format: " + data_path)

def get_model_files(target_path, feature_inputsize_dict):
    # Get model files in the target path.
    target_files = os.listdir(target_path)
    return [
        tmp for tmp in target_files 
        if (".pt" in tmp) and (re.sub("_CCSA.*$", "", tmp) in feature_inputsize_dict.keys())
    ]

def load_and_transform_data(data_dir, feature_type, input_size, transformer):
    # Load and transform feature data into tensors.
    X_tmp = data_dir.filter(regex=feature_type, axis=1)
    X_transformed_tmp = transformer.transform(X_tmp)
    X_tmp_tensor = pad_and_reshape_1D(X_transformed_tmp, input_size).type(torch.float32)
    return X_tmp_tensor

def predict_scores(model, input_tensor, device):
    # Generate prediction scores using the model.
    model = model.to(device)
    input_tensor = input_tensor.to(device)
    score_tmp, _ = model(input_tensor)
    return score_tmp.detach().cpu().numpy()

def main():
    # Define paths and parameters
    feature_inputsize_dict = {"Arm": 950, "Mut": 180}
    target_path = "/mnt/binf/eric/test/June_v3_test2/"
    data_dir = "/mnt/binf/eric/test/new_feature_test.pkl"
    output_file = target_path + "Prediction_score_0703test.csv"

    # Load data
    new_data = load_new_data(data_dir)
    new_data_idonly = new_data.loc[:, ["SampleID", "Train_Group", "train"]]

    # Get target files
    target_files = get_model_files(target_path, feature_inputsize_dict)

    device = "cpu"

    for i, target_file in enumerate(target_files):
        # Load model
        feature_type_tmp = re.sub("[_].*$", "", target_file)
        input_size_tmp = feature_inputsize_dict[feature_type_tmp]
        model_tmp = torch.load(os.path.join(target_path, target_file))

        # Load data and transformer
        _, _, _, _, _, _, _, _, _, _, _, _, _, _, feature_transformer = load_data_1D_impute(
            data_dir, input_size_tmp, feature_type_tmp
        )

        # Prepare input tensor
        X_tmp_tensor = load_and_transform_data(new_data, feature_type_tmp, input_size_tmp, feature_transformer)

        # Predict scores
        scores = predict_scores(model_tmp, X_tmp_tensor, device)
        new_data_idonly[feature_type_tmp + "_score"] = scores

        print(feature_type_tmp)
        print(i)

    # Export scores
    score_columns = [cname for cname in new_data_idonly.columns if "_score" in cname]
    new_data_idonly["final_mean"] = new_data_idonly.loc[:, score_columns].mean(axis=1)
    new_data_idonly.to_csv(output_file, index=False)

if __name__ == "__main__":
    main()
