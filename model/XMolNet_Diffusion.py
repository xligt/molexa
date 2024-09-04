import torch, math
from torch import nn
from torch_geometric.utils import scatter, softmax
from functools import partial
from einops import rearrange
import numpy as np

from utils.utils_ML import initialization, ResBlock, ResLayer, LayerNorm

from .Attention_Block import Attention_Block, UFO
from .XMolNet_Attention import NodeAttentionLayer

class PositionalEmbedding(nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False, device='cuda'):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint
        freqs = torch.arange(0, self.num_channels//2, dtype=torch.float32, device=device)
        freqs = freqs/(self.num_channels//2 - (1 if self.endpoint else 0))
        self.freqs = (1/self.max_positions)**freqs        

    def forward(self, x):
        x = torch.einsum('ij,k->ijk', x, self.freqs) #i: n_diffu, j: batch*natoms, k: nchannels
        x = torch.cat([x.cos(), x.sin()], dim=-1)
        return x

class FourierEmbedding(nn.Module):
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer('freqs', torch.randn(num_channels//2)*scale*2*np.pi)
        
    def forward(self, x):
        x = torch.einsum('ij,k->ijk', x, self.freqs)
        x = torch.cat([x.cos(), x.sin()], dim=-1)
        return x


class NoiseIdentifier(nn.Module):
    def __init__(self, EMB=FourierEmbedding, num_channels=256, natts=2, att_dim=512, nheads=8, dot_product=True, res=True, act1=nn.ReLU, act2=nn.ReLU, dropout_prob=None):
        super().__init__()
        self.rga = 'm n (h d) -> m h n d' 
        self.rgb = 'm h n d -> m n (h d)' 
        self.num_channels = num_channels
        self.natts = natts
        self.att_dim = att_dim
        self.EMB_sigma = EMB(num_channels=self.num_channels)
        self.Linear_emb = nn.Linear(self.num_channels, self.num_channels)
        self.Linear_x = nn.Linear(3, self.num_channels)
        #self.Prj_preEnc = ResLayer(2*self.att_dim, self.att_dim, res=res, act1=act1, act2=act2, norm=LayerNorm, norm_dim=-1) 

        #self.Attention_Encoder = NodeAttentionLayer(tf_in_dim=self.att_dim,tf_out_dim=self.att_dim, nheads=nheads, dot_product=dot_product, rga=self.rga, rgb=self.rgb) 
        
        self.Prj_preAtt = ResLayer(self.num_channels*4 + self.att_dim, self.att_dim, res=res, act1=act1, act2=act2, norm=LayerNorm, norm_dim=-1)  
        self.UFO = UFO(self.att_dim)
        #self.UFO = mLSTM(tf_in_dim=self.att_dim, tf_out_dim=self.att_dim, nheads=nheads)
        self.Att_Blocks = nn.ModuleList([Attention_Block(att_dim=self.att_dim, nheads=nheads, dot_product=dot_product, res=res, act1=act1, act2=act2, norm=LayerNorm, norm_dim=-1,  
                                                         rga=self.rga, rgb=self.rgb, dropout_prob=dropout_prob) for i in range(natts)]) #to be verified       

        self.Attention_Decoder = NodeAttentionLayer(tf_in_dim=self.att_dim,tf_out_dim=self.att_dim, nheads=nheads, dot_product=dot_product, rga=self.rga, rgb=self.rgb, dropout_prob=dropout_prob)        

        self.Prj_out = ResLayer(self.att_dim, 3, res=res, act1=act1, act2=None, norm=None) 
        
        #self.LNa = LayerNorm()
        #self.LN0 = LayerNorm()
        self.LN1 = LayerNorm()
        self.LN2 = LayerNorm()
        # self.LN3 = LayerNorm()
        
        self.layers = [self.Linear_emb, self.Linear_x]
        self.mods = [self.Prj_preAtt, self.Prj_out, self.Attention_Decoder]
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

    def forward(self, x, sigma, edge_ij, i_node, j_node, idx_i_edge, idx_j_edge):
        emb = self.EMB_sigma(sigma)
        emb = self.Linear_emb(emb)
        x = self.Linear_x(x)
        x = torch.cat((emb, x), dim=-1)

        #x1 = self.LNa(x1)
        # x = self.Attention_Encoder(x1, i_node, j_node)
        # x = x + x1
        #x = self.LN0(x)
        
        edge_ij = torch.cat((x[:,i_node], x[:,j_node], edge_ij), dim=-1)  
        edge_ij = self.Prj_preAtt(edge_ij)
        edge_ij_mem = edge_ij.clone()
        for i in range(self.natts):
            edge_ij = self.Att_Blocks[i](edge_ij, idx_i_edge, idx_j_edge)
            edge_ij, edge_ij_mem = self.UFO(edge_ij, edge_ij_mem)

        x = scatter(edge_ij, i_node, dim=1, reduce='add')
        x1 = self.LN1(x)
        x = self.Attention_Decoder(x1, i_node, j_node)
        x = x + x1
        x = self.LN2(x)
        out = self.Prj_out(x)           
        
        return out
        
class Denoiser(nn.Module):
    def __init__(self, sigma_data, model=NoiseIdentifier, EMB=FourierEmbedding, num_channels=256, natts=2, att_dim=512, nheads=8, dropout_prob=None):
        super().__init__()
        
        self.model = model(EMB=EMB, num_channels=num_channels, natts=natts, att_dim=att_dim, nheads=nheads, dropout_prob=dropout_prob)
        self.sigma_data = sigma_data
        self.reset_parameters()

    def reset_parameters(self):
        self.model.reset_parameters()    
        
    def forward(self, x, sigma, edge_ij, i_node, j_node, idx_i_edge, idx_j_edge):
        c_skip = self.sigma_data**2/(sigma**2 + self.sigma_data**2)
        c_out = sigma*self.sigma_data/(sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1/(self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log()/4
        c_noise = c_noise.flatten(start_dim=1) 
        F_x = self.model(c_in*x, c_noise, edge_ij, i_node, j_node, idx_i_edge, idx_j_edge)
        D_x = c_skip*x + c_out*F_x
        return D_x

def Sampler(model, y_start, edge_ij, i_node, j_node, idx_i_edge, idx_j_edge, num_steps=15, sigma_min=0.002, sigma_max=80, rho=7, S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, heun=False, step_scale=1, device='cuda'):
    
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (sigma_max**(1/rho) + step_indices/(num_steps-1)*(sigma_min**(1/rho)-sigma_max**(1/rho)))**rho
    t_steps = torch.cat((t_steps, torch.zeros(1, dtype=torch.float64, device=device)))
    
    y_next = y_start.to(torch.float64)*t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        y_cur = y_next

        #if S_churn=0, t_hat=t_cur and y_hat=y_cur
        gamma = S_churn if S_min<=t_cur<=S_max else 0 #min(S_churn/num_steps, np.sqrt(2)-1) if S_min<=t_cur<=S_max else 0
        t_hat = t_cur + gamma*t_cur
        y_hat = y_cur + (t_hat**2 - t_cur**2).sqrt()*S_noise*torch.randn_like(y_cur)
        
        denoised = model(y_hat.to(torch.float32), torch.full((y_hat.size(0),y_hat.size(1),1), t_hat, dtype=torch.float32, device=device), edge_ij, i_node, j_node, idx_i_edge, idx_j_edge).to(torch.float64)
        d_cur = (y_hat - denoised)/t_hat
        y_next = y_hat + (t_next - t_hat)*d_cur*step_scale
        
        if heun and i < num_steps-1:
            denoised = model(y_next.to(torch.float32), torch.full((y_next.size(0),y_next.size(1),1), t_next, dtype=torch.float32, device=device), edge_ij, i_node, j_node, idx_i_edge, idx_j_edge).to(torch.float64)
            d_prime = (y_next - denoised)/t_next
            y_next = y_hat + (t_next - t_hat)*(0.5*d_cur + 0.5*d_prime)

    return y_next.to(torch.float32)


        





















    
