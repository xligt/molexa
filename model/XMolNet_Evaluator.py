import torch, math
from torch import nn
import torch.nn.functional as F
from torch_geometric.utils import scatter, softmax
from functools import partial
from einops import rearrange
import numpy as np

from utils.utils_ML import initialization, ResBlock, ResLayer, LayerNorm

from .Attention_Block import Attention_Block, UFO
from .XMolNet_Attention import NodeAttentionLayer


class Evaluator(nn.Module):
    def __init__(self, num_channels=256, natts=2, att_dim=512, nheads=8, dot_product=True, res=True, act1=nn.ReLU, act2=nn.ReLU, dropout_prob=None, err_bin_center=None):
        super().__init__()
        self.rga = 'm n (h d) -> m h n d' 
        self.rgb = 'm h n d -> m n (h d)' 
        self.num_channels = num_channels
        self.natts = natts
        self.att_dim = att_dim
        self.err_bin_center = err_bin_center.view(1, 1, 1, -1)
        self.num_bins = self.err_bin_center.size(3)
        self.Linear_x = nn.Linear(3, self.num_channels)
 
        self.Prj_preAtt = ResLayer(self.num_channels*2 + self.att_dim, self.att_dim, res=res, act1=act1, act2=act2, norm=LayerNorm, norm_dim=-1)  
        self.UFO = UFO(self.att_dim)
        self.Att_Blocks = nn.ModuleList([Attention_Block(att_dim=self.att_dim, nheads=nheads, dot_product=dot_product, res=res, act1=act1, act2=act2, norm=LayerNorm, norm_dim=-1,  
                                                         rga=self.rga, rgb=self.rgb, dropout_prob=dropout_prob) for i in range(self.natts)]) #to be verified       

        self.Prj_out = ResLayer(self.att_dim, 3*self.num_bins, res=res, act1=act1, act2=None, norm=None) 
        
        self.LN1 = LayerNorm()
        
        self.layers = [self.Linear_x]
        self.mods = [self.Prj_preAtt, self.Prj_out]
        self.blocks = [self.Att_Blocks]  

        self.reset_parameters()

    def reset_parameters(self):
        
        map(partial(initialization, kaiming=False, mode='fan_in', nonlinearity='relu',glorot_scale=2.0), self.layers)
        for mod in self.mods:
            mod.reset_parameters()
            
        for block in self.blocks:
            for mod in block:
                mod.reset_parameters()

        self.UFO.reset_parameters()

    def forward(self, x, edge_ij, i_node, j_node, idx_i_edge, idx_j_edge):
        x = self.Linear_x(x)

        edge_ij = torch.cat((x[:,i_node], x[:,j_node], edge_ij), dim=-1)  
        edge_ij = self.Prj_preAtt(edge_ij)
        edge_ij_mem = edge_ij.clone()
        for i in range(self.natts):
            edge_ij = self.Att_Blocks[i](edge_ij, idx_i_edge, idx_j_edge)
            edge_ij, edge_ij_mem = self.UFO(edge_ij, edge_ij_mem)

        x = scatter(edge_ij, i_node, dim=1, reduce='add')
        x = self.LN1(x)
        prob = self.Prj_out(x)           
        prob = rearrange(prob, 'n m (d h) -> n m d h', h=self.num_bins)
        prob = F.softmax(prob, dim=-1)
        err_pred = (prob*self.err_bin_center).sum(dim=-1)

        return prob, err_pred


    
