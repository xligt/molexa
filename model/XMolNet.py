import torch
from torch import nn
from torch_geometric.utils import scatter
from functools import partial

from utils.utils_ML import initialization, ResBlock, ResLayer, LayerNorm
from utils.utils_Geom import edge_inds

from .Attention_Block import Attention_Block, UFO
from .XMolNet_Diffusion import Denoiser, NoiseIdentifier, PositionalEmbedding, FourierEmbedding, Sampler


class XMolNet(nn.Module):
    def __init__(self, z_max, z_emb_dim, q_max, q_emb_dim, pos_out_dim, att_dim, diffu_params, natts=6, nheads=1, dot_product=True, res=True, act1=nn.ReLU, act2=nn.ReLU, 
                 norm=LayerNorm, attention_type='full', lstm=False, sumup=True, num_steps=5, sigma_min=0.002, sigma_max=80, rho=7, S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, heun=False, step_scale=1, device='cuda', natts_diffu=2):
        super().__init__()

        self.y_c, self.y_hw, self.sigma_data = diffu_params['y_c'], diffu_params['y_hw'], diffu_params['sigma_data']
        self.n_diffu, self.P_mean, self.P_std = diffu_params['n_diffu'], diffu_params['P_mean'], diffu_params['P_std']
        self.y_c = self.y_c.view(1,3).to(device)
        self.y_hw = self.y_hw.view(1,3).to(device)

        self.natts_diffu = natts_diffu
        
        self.in_dim = (z_emb_dim+q_emb_dim+pos_out_dim)  
        self.att_dim = att_dim
        self.natts = natts
        self.nheads = nheads
        self.lstm = lstm
        self.sumup = sumup
        self.num_steps = num_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.S_churn = S_churn
        self.S_min = S_min 
        self.S_max = S_max
        self.S_noise = S_noise
        self.heun = heun
        self.step_scale = step_scale

        if attention_type == 'full':
            self.full = True
            self.triangular_out = False
            self.triangular_in = False
        elif attention_type == 'triangular_out':
            self.full = False
            self.triangular_out = True
            self.triangular_in = False
        elif attention_type == 'triangular_in':
            self.full = False
            self.triangular_out = False
            self.triangular_in = True             
        
        self.EMB_z = nn.Embedding(z_max, z_emb_dim)
        self.EMB_q = nn.Embedding(q_max, q_emb_dim)
        self.Linear_pos = nn.Linear(3, pos_out_dim)

        self.Prj_edge = ResLayer(int(self.in_dim*2), self.att_dim, res=res, act1=act1, act2=act2, norm=LayerNorm, norm_dim=-1) 

        if self.lstm:      
            self.UFO = UFO(self.att_dim)

        self.Att_Blocks = nn.ModuleList([Attention_Block(att_dim=self.att_dim, nheads=self.nheads, dot_product=dot_product, res=res, act1=act1, act2=act2, norm=norm, norm_dim=-1) for i in range(natts)])


        # if self.sumup:
        #     self.Prj_Blocks = nn.ModuleList([ResLayer(self.att_dim, 3, res=res, act1=act1, act2=None, norm=None) for i in range(natts)])
        # else:
        #     self.Prj_out = ResLayer(self.att_dim, 3, res=res, act1=act1, act2=None, norm=None) 

        self.Denoiser = Denoiser(sigma_data=self.sigma_data, model=NoiseIdentifier, EMB=FourierEmbedding, num_channels=256, natts=self.natts_diffu, att_dim=self.att_dim, nheads=self.nheads)

        self.layers = [self.EMB_z, self.EMB_q, self.Linear_pos]
        self.mods = [self.Prj_edge, self.Denoiser]
        self.blocks = [self.Att_Blocks]

        self.reset_parameters()

    def reset_parameters(self):
        
        map(partial(initialization, kaiming=False, mode='fan_in', nonlinearity='relu',glorot_scale=2.0), self.layers)
        for mod in self.mods:
            mod.reset_parameters()
            
        for block in self.blocks:
            for mod in block:
                mod.reset_parameters()
                
        if self.lstm: 
            self.UFO.reset_parameters()

        
    def forward(self, batch):
        z, q, pos, inds, self.y, natoms = batch.z, batch.q, batch.pos, batch.batch, batch.y, batch.natoms
        
        #i_node_loop, j_node_loop = edge_inds(pos,inds)
        i_node, j_node, i_edge, idx_i_edge, idx_j_edge = edge_inds(pos,inds,loop=True,edge_index_only=False,full=self.full,triangular_out=self.triangular_out,triangular_in=self.triangular_in)
        
        z = self.EMB_z(z)
        q = self.EMB_q(q)
        pos = self.Linear_pos(pos)
        x = torch.cat((z, q, pos), dim=-1)
        
        x_i = x[i_node]
        x_j = x[j_node]
        edge_ij = torch.cat([x_i, x_j], dim=-1)        

        edge_ij = self.Prj_edge(edge_ij)
        edge_ij_mem = edge_ij.clone()
        
        for i in range(self.natts):
            edge_ij = self.Att_Blocks[i](edge_ij, idx_i_edge, idx_j_edge)
            if self.lstm: edge_ij, edge_ij_mem = self.UFO(edge_ij, edge_ij_mem)
            

        #     if self.sumup:
        #         x = scatter(edge_ij, i_node, dim=0, reduce='add')
        #         out1 = self.Prj_Blocks[i](x)
                
        #         if i==0:
        #             out = out1
        #         else:
        #             out += out1
                    
        # if not self.sumup: 
        #     x = scatter(edge_ij, i_node, dim=0, reduce='add')
        #     out = self.Prj_out(x)

            
###
        if self.training:
            self.y = (self.y - self.y_c)/self.y_hw
            self.y = self.y.repeat(self.n_diffu, 1, 1)
    
            rnd_normal = torch.randn([self.n_diffu, natoms.size(0), 1], device=pos.device).repeat_interleave(natoms, dim=1)        
            sigma = (rnd_normal*self.P_std + self.P_mean).exp()
            noise = torch.randn_like(self.y)*sigma
            #self.sigma_train = sigma[0,0,:] #to be deleted
            #self.noise_train = noise[0,0,:] #to be deleted
    
            self.D_yn = self.Denoiser(self.y+noise, sigma, edge_ij.repeat(self.n_diffu, 1, 1), i_node, j_node, idx_i_edge, idx_j_edge)
            
            return (self.D_yn, self.y, sigma)
        else:
            self.y = self.y.repeat(1, 1, 1)
            y_diffu = torch.randn_like(self.y)
            #self.y_diffu_valid = y_diffu[0,0,:] #to be deleted
            
            self.D_yn = Sampler(self.Denoiser, y_diffu, edge_ij.repeat(1, 1, 1), i_node, j_node, idx_i_edge, idx_j_edge, num_steps=self.num_steps,
                               sigma_min=self.sigma_min, sigma_max=self.sigma_max, rho=self.rho, S_churn=self.S_churn, S_min=self.S_min, S_max=self.S_max, 
                                S_noise=self.S_noise, heun=self.heun, step_scale=self.step_scale, device=pos.device)
            self.D_yn = self.D_yn*self.y_hw.view(1, 1, 3) + self.y_c.view(1, 1, 3)
            
            return self.D_yn
        
            
        

###        
        
        














