import numpy as np
import torch

### pad and input array to a new_dim*new_dim output tensor 

def pad_and_reshape(input_array, new_dim):
    
    zeros_tensor = torch.tensor(np.zeros((input_array.shape[0],new_dim * new_dim - input_array.shape[1])),dtype=torch.float32)
    padded_tensor = torch.cat((torch.tensor(input_array,dtype=torch.float32), zeros_tensor), dim = 1)
    reshaped_tensor = padded_tensor.resize_(input_array.shape[0],1,new_dim,new_dim)
    return(reshaped_tensor)

def pad_and_reshape_1D(input_array, new_dim):
    
    zeros_tensor = torch.tensor(np.zeros((input_array.shape[0],new_dim - input_array.shape[1])),dtype=torch.float32)
    padded_tensor = torch.cat((torch.tensor(input_array,dtype=torch.float32), zeros_tensor), dim = 1)
    reshaped_tensor = padded_tensor.resize_(input_array.shape[0],1,new_dim)
    return(reshaped_tensor)