import os, sys, re
import argparse
from torch import nn, tensor

from .utils_ML import LayerNorm
from model.XMolNet import XMolNet




def FindModel(directory):
    # Regular expression to match model files
    model_pattern = re.compile(r'model_(\d+)\.chk')

    # List files in the directory and filter those that match the pattern
    model_files = [f for f in os.listdir(directory) if model_pattern.match(f)]
    
    # Debugging: print the list of matching model files
    print("Matching model files:", model_files)

    # Extract numbers and find the maximum
    if not model_files:
        return None

    largest_model = max(model_files, key=lambda f: int(model_pattern.match(f).group(1)))

    return directory+'/'+largest_model


def GetModel(device):
    
    diffu_params = \
    {'y_c': tensor([0., -0., -0.]),
     'y_hw': tensor([23.59669, 19.97351, 15.76245]),
     'n_diffu': 8,
     'P_mean': -1.2,
     'P_std': 1.2,
     'sigma_data': 0.25}  

#    diffu_params = \
#        {'y_c': tensor([0., 0., -0.]),
#         'y_hw': tensor([12.28458,  9.14532,  6.19654]),
#         'n_diffu': 8,
#         'P_mean': -1.2,
#         'P_std': 1.2,
#         'sigma_data': 0.26}

    natts_diffu = 2

    num_steps =  15
    
    z_max = 20
    z_emb_dim = 64
    q_max = 20
    q_emb_dim = 64
    pos_out_dim = 64
    
    natts = 6
    scale = 4
    att_dim = int(128*scale)
    nheads = int(8*scale)
    
    model = XMolNet(z_max, z_emb_dim, q_max, q_emb_dim, pos_out_dim, att_dim=att_dim, diffu_params = diffu_params, natts=natts, nheads=nheads, 
                    dot_product=True, res=True, act1=nn.ReLU, act2=nn.ReLU, norm=LayerNorm, attention_type='full', lstm=True, sumup=False, num_steps=num_steps, device=device, natts_diffu=natts_diffu)    
    return model, diffu_params


def GetInds():
    parser = argparse.ArgumentParser(description='model indexes')
    parser.add_argument('--train_type',type=str)
    parser.add_argument('--save_dir',type=str)
    parser.add_argument('--save_ind1',type=str)
    parser.add_argument('--save_ind2',type=str)
    args = parser.parse_args()  
    return args.train_type, args.save_dir, args.save_ind1, args.save_ind2
