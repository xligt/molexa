import torch, random
import sys, gc, traceback
from torch import nn
import fastcore.all as fc
import torch.nn.init as init
import numpy as np
from torch_geometric.nn.inits import glorot_orthogonal
from functools import partial
import torch.nn.functional as F

class EDM_MSELoss(nn.Module):
    def __init__(self, diffu_params):
        super(EDM_MSELoss, self).__init__()
        self.y_c, self.y_hw, self.sigma_data = diffu_params['y_c'], diffu_params['y_hw'], diffu_params['sigma_data']
        self.y_c = self.y_c.mean()
        self.y_hw = self.y_hw.mean()

    def forward(self, out):
        D_yn, y, sigma = out
        weight = (sigma**2 + self.sigma_data**2)/(sigma*self.sigma_data)**2
        loss = weight*((D_yn - y)**2)
        return loss.mean()*self.y_hw**2
        #note that the loss is scaled due to both the scaling factor and dim expansion
        
class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, preds, targets):
        loss = (preds - targets)**2        
        return loss.mean()
        

class MeanAbsoluteError:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.sum_abs_error = 0.0
        self.count = 0
    
    def update(self, preds, targets):
        abs_error = torch.abs(preds - targets)
        self.sum_abs_error += torch.sum(abs_error).item()
        self.count += preds.numel()
    
    def compute(self):
        return self.sum_abs_error / self.count if self.count != 0 else 0.0
    
    def __call__(self, preds, targets):
        self.reset()
        self.update(preds, targets)
        return self.compute()

def clean_ipython_hist():
    # Code in this function mainly copied from IPython source
    if not 'get_ipython' in globals(): return
    ip = get_ipython()
    user_ns = ip.user_ns
    ip.displayhook.flush()
    pc = ip.displayhook.prompt_count + 1
    for n in range(1, pc): user_ns.pop('_i'+repr(n),None)
    user_ns.update(dict(_i='',_ii='',_iii=''))
    hm = ip.history_manager
    hm.input_hist_parsed[:] = [''] * pc
    hm.input_hist_raw[:] = [''] * pc
    hm._i = hm._ii = hm._iii = hm._i00 =  ''

def clean_tb():
    # h/t Piotr Czapla
    if hasattr(sys, 'last_traceback'):
        traceback.clear_frames(sys.last_traceback)
        delattr(sys, 'last_traceback')
    if hasattr(sys, 'last_type'): delattr(sys, 'last_type')
    if hasattr(sys, 'last_value'): delattr(sys, 'last_value')

def clean_mem():
    clean_tb()
    clean_ipython_hist()
    gc.collect()
    torch.cuda.empty_cache()


def set_seed(seed, deterministic=False):
    torch.use_deterministic_algorithms(deterministic)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def initialization(m, kaiming=True, mode='fan_in', nonlinearity='relu',glorot_scale=2.0):
    if isinstance(m, nn.Linear):
        if kaiming:
            init.kaiming_normal_(m.weight, mode=mode, nonlinearity=nonlinearity)
        else:
            glorot_orthogonal(m.weight, scale=glorot_scale)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.Embedding):
        m.weight.data.uniform_(-sqrt(3), sqrt(3))


class LayerNorm(nn.Module):
    def __init__(self, dim=-1, eps=1e-5):
        super().__init__()
        self.dim=dim
        self.eps = eps
        self.mult = nn.Parameter(torch.tensor(1.))
        self.add  = nn.Parameter(torch.tensor(0.))

    def forward(self, x):
        m = x.mean(dim=self.dim, keepdim=True)
        v = x.var (dim=self.dim, keepdim=True)
        x = (x-m) / ((v+self.eps).sqrt())
        return x*self.mult + self.add
        
class SGD:
    def __init__(self, params, lr, wd=0.):
        params = list(params)
        fc.store_attr()
        self.i = 0

    def step(self):
        with torch.no_grad():
            for p in self.params:
                self.reg_step(p)
                self.opt_step(p)
        self.i +=1

    def opt_step(self, p): p -= p.grad * self.lr
    def reg_step(self, p):
        if self.wd != 0: p *= 1 - self.lr*self.wd

    def zero_grad(self):
        for p in self.params: p.grad.data.zero_()


class ResLayer(nn.Module):
    def __init__(self, ni, no, res=True, act1=nn.ReLU, act2 = nn.ReLU, norm=LayerNorm, norm_dim=-1):
        super().__init__()

        self.res = res
        self.act1 = act1() if act1 else fc.noop
        self.act2 = act2() if act2 else fc.noop
        self.norm = norm(dim=norm_dim) if norm else fc.noop

        self.lin1 = nn.Linear(ni, ni)
        self.lin2 = nn.Linear(ni, no)

        # self.lin1 = nn.Linear(ni, no)
        # self.lin2 = nn.Linear(no, no)
        
        if self.res:
            self.idconv = fc.noop if ni==no else nn.Linear(ni, no)

    def reset_parameters(self):
        if self.res:
            map(partial(initialization, kaiming=False, mode='fan_in', nonlinearity='relu',glorot_scale=2.0), [self.lin1, self.lin2, self.idconv])
        else:
            map(partial(initialization, kaiming=False, mode='fan_in', nonlinearity='relu',glorot_scale=2.0), [self.lin1, self.lin2])
        
    def forward(self, x):
        if self.res:
            return self.norm(self.idconv(x) + self.act2(self.lin2(self.act1(self.lin1(x)))))          
        else:
            return self.act2(self.lin2(self.act1(self.lin1(x))))


def _fc_block(ni, no, norm=None):
    return nn.Sequential(nn.Linear(ni, no),norm,
                         nn.Linear(no, no),norm)

class ResBlock(nn.Module):
    def __init__(self, ni, no,act=nn.ReLU,norm=None):
        super().__init__()
        self.fcs = _fc_block(ni, no, norm=norm)
        self.idconv = fc.noop if ni==no else nn.Linear(ni, no)
        self.act = act()

    def reset_parameters(self):
        initialization(self.idconv, kaiming=False, mode='fan_in', nonlinearity='relu',glorot_scale=2.0)
        self.fcs.apply(partial(initialization, kaiming=False, mode='fan_in', nonlinearity='relu',glorot_scale=2.0)) 

    def forward(self, x): return self.act(self.fcs(x) + self.idconv(x))
        



class Adam(SGD):
    def __init__(self, params, lr, wd=0., beta1=0.9, beta2=0.99, eps=1e-5):
        super().__init__(params, lr=lr, wd=wd)
        self.beta1,self.beta2,self.eps = beta1,beta2,eps

    def opt_step(self, p):
        if not hasattr(p, 'avg'): p.avg = torch.zeros_like(p.grad.data)
        if not hasattr(p, 'sqr_avg'): p.sqr_avg = torch.zeros_like(p.grad.data)
        p.avg = self.beta1*p.avg + (1-self.beta1)*p.grad
        unbias_avg = p.avg / (1 - (self.beta1**(self.i+1)))
        p.sqr_avg = self.beta2*p.sqr_avg + (1-self.beta2)*(p.grad**2)
        unbias_sqr_avg = p.sqr_avg / (1 - (self.beta2**(self.i+1)))
        p -= self.lr * unbias_avg / (unbias_sqr_avg + self.eps).sqrt()