import torch
from torch import nn
import torch.nn.functional as F
from functools import partial

from utils.utils_ML import initialization, ResLayer, LayerNorm
from utils.utils_Geom import edge_inds

from .XMolNet_Attention import NodeAttentionLayer, EdgeAttentionLayer

class Attention_Block(nn.Module):
    def __init__(self, att_dim, nheads=1, dot_product=True, res=True, act1=nn.ReLU, act2=None, norm=LayerNorm, norm_dim=-1, rga='n (h d) -> h n d', rgb='h n d -> n (h d)', dropout_prob=None):
        super().__init__()      

        self.att_dim = att_dim      
        self.rga = rga
        self.rgb = rgb        
        self.Attention_Edge = EdgeAttentionLayer(tf_in_dim = self.att_dim, 
                                                         tf_out_dim = self.att_dim, 
                                                         nheads=nheads,
                                                         dot_product=dot_product,
                                                rga=self.rga,
                                                rgb=self.rgb,
                                                dropout_prob=dropout_prob)
        self.LayerNorm_edge = LayerNorm(dim=norm_dim)


        self.Prj_postAtt = ResLayer(self.att_dim, self.att_dim, res=res, act1=act1, act2=act2, norm=norm, norm_dim=norm_dim, dropout_prob=dropout_prob)        

        self.modules = [self.Attention_Edge,
                       self.Prj_postAtt]

    def reset_parameters(self):
        for mod in self.modules:
            mod.reset_parameters()

    def forward(self, x, idx_i_edge, idx_j_edge):
        
        x1 = self.Attention_Edge(x, idx_i_edge, idx_j_edge)
        x = x + x1
        x = self.LayerNorm_edge(x)

        x = self.Prj_postAtt(x)        
        
        return x

class UFO(nn.Module):
    def __init__(self, att_dim):
        super().__init__()
        self.att_dim = att_dim
        self.prj_t = nn.Linear(self.att_dim, self.att_dim)
        self.prj_u = nn.Linear(self.att_dim, self.att_dim)
        self.prj_f = nn.Linear(self.att_dim, self.att_dim)
        self.prj_o = nn.Linear(self.att_dim, self.att_dim)

        self.layers = [self.prj_t, self.prj_u, self.prj_f, self.prj_o]    

    def reset_parameters(self):
        map(partial(initialization, kaiming=False, mode='fan_in', nonlinearity='relu',glorot_scale=2.0), self.layers)

    def forward(self, h, c):
        c_tilde = torch.tanh(self.prj_t(h))
        g_u = torch.sigmoid(self.prj_u(h))
        g_f = torch.sigmoid(self.prj_f(h))
        g_o = torch.sigmoid(self.prj_o(h))
        c = g_u*c_tilde + g_f*c
        h = g_o*torch.tanh(c)
        return h, c




        

