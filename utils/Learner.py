import torch
import torch.nn.functional as F
import fastcore.all as fc
from operator import attrgetter
from functools import partial

class with_cbs:
    def __init__(self, nm): self.nm = nm
    def __call__(self, f):
        def _f(o, *args, **kwargs):
            try:
                o.callback(f'before_{self.nm}')
                f(o, *args, **kwargs)
                o.callback(f'after_{self.nm}')
            except globals()[f'Cancel{self.nm.title()}Exception']: pass
            finally: o.callback(f'cleanup_{self.nm}')
        return _f

class Learner():
    def __init__(self, model, dls=(0,), loss_func=F.mse_loss, val_loss_func=F.mse_loss, cbs=None, load_path=None, opt_func=torch.optim.Adam, 
                 lr=0.001, beta1=0.9, beta2=0.99, parallel=False, device_id='cuda', rank=0, sampler=None, freeze=False, load_opt=True, exclude_loading=None):
        cbs = fc.L(cbs)
        fc.store_attr()

    @with_cbs('batch')
    def _one_batch(self):
        self.predict()
        self.callback('after_predict')
        self.get_loss()
        self.callback('after_loss')
        if self.training:
            self.backward()
            self.callback('after_backward')
            self.step()
            self.callback('after_step')
            self.zero_grad()

    @with_cbs('epoch')
    def _one_epoch(self):
        for self.iter,self.batch in enumerate(self.dl): self._one_batch()
    
    def one_epoch(self, training):
        self.model.train(training)
        
        if training:
            if self.sampler is not None: self.sampler.set_epoch(self.epoch)
            self.dl = self.dls.train 
        else:
            self.dl = self.dls.valid
            
        self._one_epoch()
    
    @with_cbs('fit')
    def _fit(self, train, valid):              
        for self.epoch in self.epochs:
            if train: self.one_epoch(True)
            if valid: torch.no_grad()(self.one_epoch)(False)

    def fit(self, n_epochs=1, train=True, valid=True, cbs=None, lr=None, beta1=None, beta2=None):
        cbs = fc.L(cbs)
        
        for cb in cbs: self.cbs.append(cb)
        try:
            self.n_epochs = n_epochs
            self.epochs = range(n_epochs)
            
            if lr is not None: self.lr = lr
            if beta1 is not None: self.beta1 = beta1
            if beta2 is not None: self.beta2 = beta2
            if self.opt_func: 
                if self.freeze:
                    self.opt = self.opt_func(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr, betas=(self.beta1,self.beta2)) 
                else:
                    self.opt = self.opt_func(self.model.parameters(), lr=self.lr, betas=(self.beta1,self.beta2)) 

            if self.load_path:
                self.load_checkpoint(load_opt=self.load_opt)
                self.load_path = None            
                self.lr = self.opt.param_groups[0]['lr']
                self.beta1 = self.opt.param_groups[0]['betas'][0]
                self.beta2 = self.opt.param_groups[0]['betas'][1]

            
            self._fit(train, valid)
        finally:
            for cb in cbs: self.cbs.remove(cb)

    def predict_batch(self, batch):
        if self.model.training: self.model.train(False)
        if self.load_path:
                self.load_checkpoint(load_opt=False)
                self.load_path = None                   
        return self.model(batch)
        
    def __getattr__(self, name):
        if name in ('predict','get_loss','backward','step','zero_grad'): return partial(self.callback, name)
        raise AttributeError(name)
    
    def callback(self, method_nm): run_cbs(self.cbs, method_nm, self)


    def load_checkpoint(self, save_id=0, load_opt=True):
        if self.parallel:
            checkpoint = torch.load(self.load_path, map_location = {'cuda:%d' % save_id: 'cuda:%d' % self.device_id})
            if self.exclude_loading is not None:
                model_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if not k.startswith(self.exclude_loading)}
            else:
                model_dict = checkpoint['model_state_dict']

            self.model.module.load_state_dict(model_dict, strict=False)
        else:
            checkpoint = torch.load(self.load_path, map_location='cuda:0')
            if self.exclude_loading is not None:
                model_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if not k.startswith(self.exclude_loading)}
            else:
                model_dict = checkpoint['model_state_dict']
 
            self.model.load_state_dict(model_dict, strict=False)
        
        if load_opt and not self.freeze: 
            try:
                self.opt.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as err:
                pass
  
    
    @property
    def training(self): return self.model.training

class CancelFitException(Exception): pass
class CancelBatchException(Exception): pass
class CancelEpochException(Exception): pass


def run_cbs(cbs, method_nm, learn=None):
    for cb in sorted(cbs, key=attrgetter('order')):
        method = getattr(cb, method_nm, None)
        if method is not None: method(learn)
