import torch, math
from torch import nn
from torch_geometric.utils import scatter, softmax
from functools import partial
from einops import rearrange

from utils.utils_ML import initialization, ResBlock, LayerNorm, ResLayer

class mLSTM(nn.Module):
    def __init__(self, tf_in_dim, tf_out_dim, nheads=1, rga='n (h d) -> h n d', rgb='h n d -> n (h d)', gate='exp'):
        super().__init__()

        self.tf_in_dim = tf_in_dim
        self.tf_out_dim = tf_out_dim
        self.nheads = nheads     
        self.rga = rga
        self.rgb = rgb
        self.last_dim = 2 if self.rga=='n (h d) -> h n d' else 3
        self.scale = math.sqrt(self.tf_out_dim/self.nheads)   
        self.gate = gate

        self.QKVO = nn.Linear(self.tf_in_dim, 4*self.tf_out_dim, bias=True)
        self.UF = nn.Linear(self.tf_in_dim, 2*self.nheads, bias=True)
        self.LayerNorm = LayerNorm(dim=-1)
        
        self.layers = [self.QKVO, self.UF]
        

    def reset_parameters(self):
        
        map(partial(initialization, kaiming=False, mode='fan_in', nonlinearity='relu',glorot_scale=2.0), self.layers)
        

    def forward(self, c, x):
        
        qkvo = self.QKVO(x)
        qkvo = rearrange(qkvo, self.rga, h=self.nheads)
        q, k, v, o = torch.chunk(qkvo, 4, dim=-1)
        k = k/self.scale
    
        ct = torch.einsum('ijk,ijl->ijkl', v, k) if self.rga=='n (h d) -> h n d' else torch.einsum('hijk,hijl->hijkl', v, k) #note that the h here is diffu index, rather than head. h is only head in rga or rgb

        uf = self.UF(x)
        uf = rearrange(uf, self.rga, h=self.nheads)
        u, f = torch.chunk(uf, 2, dim=-1)


        if c is None:
            c = ct
            if self.gate != 'sigmoid':
                self.m = torch.max(torch.cat((f, u), dim=self.last_dim), dim=self.last_dim, keepdim=True).values
                self.n = torch.exp(u)*k
        else:
            if self.gate == 'sigmoid':
                f = torch.sigmoid(f)
                u = torch.sigmoid(u)
            else:
                mprev = self.m
                self.m = torch.max(torch.cat((f+mprev, u), dim=self.last_dim), dim=self.last_dim, keepdim=True).values
                u = torch.exp(u - self.m)
                f = torch.exp(f + mprev - self.m)

            c = f.unsqueeze(3)*c + u.unsqueeze(3)*ct if self.rga=='n (h d) -> h n d' else f.unsqueeze(4)*c + u.unsqueeze(4)*ct 
            self.n = f*self.n + u*k

        h= torch.einsum('ijkl,ijl->ijk', ct, q) if self.rga=='n (h d) -> h n d' else torch.einsum('hijkl,hijl->hijk', ct, q)
        nm = torch.abs(torch.einsum('ijk,ijk->ij', self.n, q)) if self.rga=='n (h d) -> h n d' else torch.abs(torch.einsum('hijk,hijk->hij', self.n, q))
        h = h/torch.max(torch.cat((nm, torch.ones_like(nm)), dim=self.last_dim-1), dim=self.last_dim-1, keepdim=True).values.unsqueeze(self.last_dim)

        h = h*torch.sigmoid(o)
        
        h = rearrange(h, self.rgb, h=self.nheads)
        
        h = x + h
        h = self.LayerNorm(h)
        
        return c, h


class NodeAttentionLayer(nn.Module):
    def __init__(self, tf_in_dim,tf_out_dim, nheads=1, dot_product=True, rga='n (h d) -> h n d', rgb='h n d -> n (h d)'):
        super().__init__()
        self.tf_in_dim = tf_in_dim
        self.tf_out_dim = tf_out_dim
        self.nheads = nheads
        self.dot_product = dot_product
        self.rga = rga
        self.rgb = rgb
        self.reduce_dim = 1 if self.rga=='n (h d) -> h n d' else 2
        self.last_dim   = 2 if self.rga=='n (h d) -> h n d' else 3      
        if self.dot_product: self.scale = math.sqrt(self.tf_out_dim/self.nheads) 

        self.node_tf_qkv = nn.Linear(self.tf_in_dim, 3*self.tf_out_dim, bias=False)
        self.prj = nn.Linear(self.tf_out_dim, self.tf_out_dim, bias=False)

        if not self.dot_product:
            self.tf_att = nn.Linear(int(2*self.tf_out_dim/self.nheads), 1, bias=True)

        self.layers = [self.node_tf_qkv, self.prj]


    def reset_parameters(self):
        map(partial(initialization, kaiming=False, mode='fan_in', nonlinearity='relu',glorot_scale=2.0), self.layers)
        
        if not self.dot_product: 
            initialization(self.tf_att, kaiming=False, mode='fan_in', nonlinearity='relu',glorot_scale=2.0)

    def forward(self, x, i_node, j_node):

        x = self.node_tf_qkv(x)
        x = rearrange(x, self.rga, h=self.nheads)
        
        x_i_q, x_j_k, x_j_v = torch.chunk(x, 3, dim=-1)

        x_i_q = x_i_q[:,i_node,:] if self.rga=='n (h d) -> h n d' else x_i_q[:,:,i_node,:]
        x_j_k = x_j_k[:,j_node,:] if self.rga=='n (h d) -> h n d' else x_j_k[:,:,j_node,:]
        x_j_v = x_j_v[:,j_node,:] if self.rga=='n (h d) -> h n d' else x_j_v[:,:,j_node,:]

        if self.dot_product:
            alpha = (x_i_q*x_j_k).sum(dim=-1)/self.scale
        else:
            x_all = torch.cat([x_i_q, x_j_k], dim=-1)
            alpha = self.tf_att(x_all).squeeze(self.last_dim) 
        
        alpha = softmax(alpha, i_node, num_nodes=x.size(self.reduce_dim), dim=self.reduce_dim)
    
        x = scatter(x_j_v * alpha.unsqueeze(self.last_dim), i_node, dim=self.reduce_dim, reduce='add')

        x = rearrange(x, self.rgb, h=self.nheads)

        x = self.prj(x)
        
        return x




class EdgeAttentionLayer(nn.Module):
    def __init__(self, tf_in_dim, tf_out_dim, nheads=1, dot_product=True, rga='n (h d) -> h n d', rgb='h n d -> n (h d)'):
        super().__init__()

        self.tf_in_dim = tf_in_dim
        self.tf_out_dim = tf_out_dim
        self.nheads = nheads     
        self.dot_product = dot_product
        self.rga = rga
        self.rgb = rgb
        self.reduce_dim = 1 if self.rga=='n (h d) -> h n d' else 2
        self.last_dim   = 2 if self.rga=='n (h d) -> h n d' else 3
        if self.dot_product: self.scale = math.sqrt(self.tf_out_dim/self.nheads)   

        self.edge_tf_qkv = nn.Linear(self.tf_in_dim, 3*self.tf_out_dim, bias=False)
        self.prj = nn.Linear(self.tf_out_dim, self.tf_out_dim, bias=False)

        if not self.dot_product:
            self.tf_att = nn.Linear(int(2*self.tf_out_dim/self.nheads), 1, bias=True)

        self.layers = [self.edge_tf_qkv, self.prj]
        

    def reset_parameters(self):
        
        map(partial(initialization, kaiming=False, mode='fan_in', nonlinearity='relu',glorot_scale=2.0), self.layers)
        
        if not self.dot_product:
            initialization(self.tf_att, kaiming=False, mode='fan_in', nonlinearity='relu',glorot_scale=2.0)

    def forward(self, edge_ij, idx_i_edge, idx_j_edge):

        edge_ij = self.edge_tf_qkv(edge_ij)
        edge_ij = rearrange(edge_ij, self.rga, h=self.nheads)
        
        edge_a_q, edge_b_k, edge_b_v = torch.chunk(edge_ij, 3, dim=-1)
      
        edge_a_q = edge_a_q[:,idx_i_edge,:] if self.rga=='n (h d) -> h n d' else edge_a_q[:,:,idx_i_edge,:]
        edge_b_k = edge_b_k[:,idx_j_edge,:] if self.rga=='n (h d) -> h n d' else edge_b_k[:,:,idx_j_edge,:]
        edge_b_v = edge_b_v[:,idx_j_edge,:] if self.rga=='n (h d) -> h n d' else edge_b_v[:,:,idx_j_edge,:]

        if self.dot_product:
            alpha = (edge_a_q*edge_b_k).sum(dim=-1)/self.scale
        else:
            edge_all = torch.cat([edge_a_q, edge_b_k], dim=-1)
            alpha = self.tf_att(edge_all).squeeze(self.last_dim) 

        alpha = softmax(alpha, idx_i_edge, num_nodes=edge_ij.size(self.reduce_dim), dim=self.reduce_dim) # note that num_nodes here is actually number of edges

        x = scatter(edge_b_v * alpha.unsqueeze(self.last_dim), idx_i_edge, dim=self.reduce_dim, reduce='add')

        x = rearrange(x, self.rgb, h=self.nheads)

        edge_ij = self.prj(x)        
        
        return edge_ij
