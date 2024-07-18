import torch, torch_geometric
from typing import Mapping
from copy import copy
from torcheval.metrics import MeanSquaredError, Mean
import fastcore.all as fc
from fastprogress import progress_bar,master_bar
import matplotlib.pyplot as plt
import os, sys
from datetime import datetime


class Callback(): order = 0

class CompletionCB(Callback):
    def before_fit(self, learn): self.count = 0
    def after_batch(self, learn): self.count += 1
    def after_fit(self, learn): print(f'Completed {self.count} batches')

def to_device(x, device=None):
    if device is None:
          device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    if isinstance(x, torch_geometric.data.batch.DataBatch): return x.to(device)
    if isinstance(x, torch.Tensor): return x.to(device)
    if isinstance(x, Mapping): return {k:v.to(device) for k,v in x.items()}
    return type(x)(to_device(o, device) for o in x)
    
def to_cpu(x):
    if isinstance(x, torch_geometric.data.batch.DataBatch): return to_device(x, device='cpu')
    if isinstance(x, Mapping): return {k:to_cpu(v) for k,v in x.items()}
    if isinstance(x, list): return [to_cpu(o) for o in x]
    if isinstance(x, tuple): return tuple(to_cpu(list(x)))
    res = x.detach().cpu()
    return res.float() if res.dtype==torch.float16 else res

class OptResetCB(Callback):
    def __init__(self, step_size, gamma):
        self.step_size = step_size
        self.gamma = gamma

    def after_epoch(self, learn):
        if learn.model.training and learn.epoch%self.step_size==0:
            learn.lr *= self.gamma
            learn.opt = learn.opt_func(learn.model.parameters(), lr=learn.lr, betas=(learn.beta1,learn.beta2))

class MetricsCB(Callback):
    def __init__(self, *ms, **metrics):
        for o in ms: metrics[type(o).__name__] = o
        self.metrics = metrics
        self.all_metrics = copy(metrics)
        self.all_metrics['loss'] = self.loss = Mean()
        self.vars = ['epoch', 'loss_t', 'mae_t', 'loss_v', 'mae_v', 'lr', 'beta1', 'beta2', 'time', 'rank']
        self.vars_valid = ['epoch', 'loss_v', 'mae_v', 'lr', 'beta1', 'beta2', 'time', 'rank']
        self.vars_space = {'epoch':15, 'loss_t':15, 'mae_t':15, 'loss_v':15, 'mae_v':15, 'lr':15, 'beta1':15, 'beta2':15, 'time':15, 'rank':15}
        self.vars_digit = {'loss_t':5, 'mae_t':9, 'loss_v':5, 'mae_v':9, 'lr':9, 'beta1':3, 'beta2':3}
        self.report_dict = {} 
    
    def _log(self, d): print(d)
        
    def before_fit(self, learn):
        learn.metrics = self
        if learn.rank==0:
            header = f" ".join(map(lambda var: f"{f'{var}':<{self.vars_space[var]}}", self.vars))
            print(header)
        sys.stdout.flush()
        
    def before_epoch(self, learn): [o.reset() for o in self.all_metrics.values()]
    
    def after_epoch(self, learn):
        log = {k: v.compute() for k,v in self.all_metrics.items()}
        
        tp = '_t' if learn.model.training else "_v"
        
        for k in log.keys():
            self.report_dict[k+tp] = log[k] 
            
        if not learn.training:
            self.report_dict['epoch'] = learn.epoch
            self.report_dict['lr'] = learn.opt.param_groups[0]['lr']
            self.report_dict['beta1'] = learn.opt.param_groups[0]['betas'][0]
            self.report_dict['beta2'] = learn.opt.param_groups[0]['betas'][1]      
            self.report_dict['time'] = datetime.now().strftime('%H:%M:%S,%m-%d')
            self.report_dict['rank'] = learn.rank
            try:
                output = f" ".join(map(lambda var: f"{self.report_dict[var]:<{self.vars_space[var]}}" if var in ['time', 'epoch', 'rank'] else f"{self.report_dict[var]:<{self.vars_space[var]}.{self.vars_digit[var]}f}", self.vars))
                print(output)

                #for i in range(torch.cuda.device_count()):
                #    print(f"rank: {learn.rank},GPU {i}: {torch.cuda.get_device_name(i)}")
                #    print(f"  Allocated: {torch.cuda.memory_allocated(i) / 1024 ** 3:.2f} GB")
                #    print(f"  Cached:    {torch.cuda.memory_reserved(i) / 1024 ** 3:.2f} GB")
                #print('rank:', learn.rank, 'valid, batch.y[0]:', learn.batch.y[0])
                #print('rank:', learn.rank, 'valid, learn.y_diffu_valid[0]:', learn.model.module.y_diffu_valid[0])
                sys.stdout.flush()
            except:
                
                header = f" ".join(map(lambda var: f"{f'{var}':<{self.vars_space[var]}}", self.vars_valid))
                print(header)                
                output = f" ".join(map(lambda var: f"{self.report_dict[var]:<{self.vars_space[var]}}" if var in ['time', 'epoch', 'rank'] else f"{self.report_dict[var]:<{self.vars_space[var]}.{self.vars_digit[var]}f}", self.vars_valid))
                print(output)
                sys.stdout.flush()
                #pass
        # else: 
        #     try:
        #         print('rank:', learn.rank, 'train, batch.y[0]:', learn.batch.y[0])
        #         print('rank:', learn.rank, 'train,  learn.sigma_train, learn.noise_train[0]:', learn.model.module.sigma_train, learn.model.module.noise_train[0])
        #         sys.stdout.flush()
        #     except:
        #         pass                

    def after_batch(self, learn):
        #x,*_,y = to_cpu(learn.batch) # was x,y,*_ = to_cpu(learn.batch)
        ###
        if isinstance(learn.model, torch.nn.parallel.DistributedDataParallel):
            list_cpu = to_cpu([learn.model.module.D_yn, learn.model.module.y])
        else:
            list_cpu = to_cpu([learn.model.D_yn, learn.model.y])

        ###
        for m in self.metrics.values(): m.update(list_cpu[0], list_cpu[1])
        self.loss.update(to_cpu(learn.loss), weight=list_cpu[1].size(1))

class SaveCB(Callback):
    def __init__(self, epochs, step_size, save_dir, prefix='model_epch', save_chk=True): 
        self.epochs = epochs
        self.step_size = step_size
        self.prefix = prefix
        self.save_chk = save_chk
        self.folder_name = save_dir
        # Check if the folder exists
        if not os.path.exists(self.folder_name):
            # Create the folder if it doesn't exist
            os.makedirs(self.folder_name)  

    def save_checkpoint(self, model, optimizer, save_path, epoch):
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            torch.save({
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch
            }, save_path)               
        else:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch
            }, save_path)    
        
    def after_epoch(self, learn):
        epoch = learn.epoch + 1
        if self.save_chk and (epoch%self.step_size==0 or epoch==self.epochs):
            chkpth = self.folder_name +'/'+self.prefix+str(epoch)+'.chk'
            self.save_checkpoint(learn.model, learn.opt, chkpth, epoch)        
        if epoch==self.epochs:
            mpth = self.folder_name +'/'+self.prefix+str(epoch)+'.pth'
            if isinstance(learn.model, torch.nn.parallel.DistributedDataParallel):
                torch.save(learn.model.module, mpth)
            else:
                torch.save(learn.model, mpth)

        
        

class DeviceCB(Callback):
        
    def before_batch(self, learn): learn.batch = to_device(learn.batch, device=learn.device_id)

class TrainCB(Callback):
    def __init__(self, n_inp=1): self.n_inp = n_inp
    # def predict(self, learn): learn.preds = learn.model(*learn.batch[:self.n_inp])
    # def get_loss(self, learn): learn.loss = learn.loss_func(learn.preds, *learn.batch[self.n_inp:])
    def predict(self, learn): learn.preds = learn.model(learn.batch)
    def get_loss(self, learn): learn.loss = learn.loss_func(learn.preds) if learn.training else learn.val_loss_func(learn.preds, learn.batch.y)
    def backward(self, learn): learn.loss.backward()
    def step(self, learn): learn.opt.step()
    def zero_grad(self, learn): learn.opt.zero_grad()

# class TrainCB(Callback):
#   def __init__(self, n_inp=1): self.n_inp = n_inp
#   def predict(self, learn): learn.preds = learn.model(learn.batch[0])
#   def get_loss(self, learn):
#     # print("Predictions shape:", learn.preds.shape)
#     # print("Targets shape:", learn.batch[1].shape)
#     learn.loss = learn.loss_func(learn.preds, learn.batch[1])

#   def backward(self, learn): learn.loss.backward()
#   def step(self, learn): learn.opt.step()
#   def zero_grad(self, learn): learn.opt.zero_grad()

class ProgressCB(Callback):
    order = MetricsCB.order+1
    def __init__(self, plot=False): self.plot = plot
    def before_fit(self, learn):
        learn.epochs = self.mbar = master_bar(learn.epochs)
        self.first = True
        if hasattr(learn, 'metrics'): learn.metrics._log = self._log
        self.losses = []
        self.val_losses = []
    
    def _log(self, d):
        if self.first:
            self.mbar.write(list(d), table=True)
            self.first = False
        self.mbar.write(list(d.values()), table=True)
    
    def before_epoch(self, learn): learn.dl = progress_bar(learn.dl, leave=False, parent=self.mbar)
        
    def after_batch(self, learn):
        learn.dl.comment = f'{learn.loss:.3f}'
        if self.plot and hasattr(learn, 'metrics') and learn.training:
            self.losses.append(learn.loss.item())
            if self.val_losses: self.mbar.update_graph([[fc.L.range(self.losses), self.losses],[fc.L.range(learn.epoch).map(lambda x: (x+1)*len(learn.dls.train)), self.val_losses]])
    
    def after_epoch(self, learn):
        if not learn.training:
            if self.plot and hasattr(learn, 'metrics'):
                self.val_losses.append(learn.metrics.all_metrics['loss'].compute())
                self.mbar.update_graph([[fc.L.range(self.losses), self.losses],[fc.L.range(learn.epoch+1).map(lambda x: (x+1)*len(learn.dls.train)), self.val_losses]])



class BaseSchedCB(Callback):
    def __init__(self, sched): self.sched = sched
    def before_fit(self, learn): self.schedo = self.sched(learn.opt)
    def _step(self, learn):
        if learn.training: self.schedo.step()

class BatchSchedCB(BaseSchedCB):
    def after_batch(self, learn): self._step(learn)

class EpochSchedCB(BaseSchedCB):
    def after_epoch(self, learn): self._step(learn)

class RecorderCB(Callback):
    def __init__(self, **d): self.d = d
    def before_fit(self, learn):
        self.recs = {k:[] for k in self.d}
        self.pg = learn.opt.param_groups[0]

    def after_batch(self, learn):
        if not learn.training: return
        for k,v in self.d.items():
            self.recs[k].append(v(self))

    def plot(self):
        for k,v in self.recs.items():
            plt.plot(v, label=k)
            plt.legend()
            plt.show()


class Hook():
    def __init__(self, m, f): self.hook = m.register_forward_hook(partial(f, self))
    def remove(self): self.hook.remove()
    def __del__(self): self.remove()

class Hooks(list):
    def __init__(self, ms, f): super().__init__([Hook(m, f) for m in ms])
    def __enter__(self, *args): return self
    def __exit__ (self, *args): self.remove()
    def __del__(self): self.remove()
    def __delitem__(self, i):
        self[i].remove()
        super().__delitem__(i)
    def remove(self):
        for h in self: h.remove()

class HooksCallback(Callback):
    def __init__(self, hookfunc, mod_filter=fc.noop, on_train=True, on_valid=False, mods=None):
        fc.store_attr()
        super().__init__()

    def before_fit(self, learn):
        if self.mods: mods=self.mods
        else: mods = fc.filter_ex(learn.model.modules(), self.mod_filter)
        self.hooks = Hooks(mods, partial(self._hookfunc, learn))

    def _hookfunc(self, learn, *args, **kwargs):
        if (self.on_train and learn.training) or (self.on_valid and not learn.training): self.hookfunc(*args, **kwargs)

    def after_fit(self, learn): self.hooks.remove()
    def __iter__(self): return iter(self.hooks)
    def __len__(self): return len(self.hooks)

def append_stats(hook, mod, inp, outp):
    if not hasattr(hook,'stats'): hook.stats = ([],[],[])
    acts = to_cpu(outp)
    hook.stats[0].append(acts.mean())
    hook.stats[1].append(acts.std())
    hook.stats[2].append(acts.abs().histc(40,0,10))

def get_hist(h): return torch.stack(h.stats[2]).t().float().log1p()

def get_min(h):
    h1 = torch.stack(h.stats[2]).t().float()
    return h1[0]/h1.sum(0)

class ActivationStats(HooksCallback):
    def __init__(self, mod_filter=fc.noop): super().__init__(append_stats, mod_filter)

    def color_dim(self, figsize=(11,5)):
        fig,axes = get_grid(len(self), figsize=figsize)
        for ax,h in zip(axes.flat, self):
            show_image(get_hist(h), ax, origin='lower')

    def dead_chart(self, figsize=(11,5)):
        fig,axes = get_grid(len(self), figsize=figsize)
        for ax,h in zip(axes.flatten(), self):
            ax.plot(get_min(h))
            ax.set_ylim(0,1)

    def plot_stats(self, figsize=(10,4)):
        fig,axs = plt.subplots(1,2, figsize=figsize)
        for h in self:
            for i in 0,1: axs[i].plot(h.stats[i])
        axs[0].set_title('Means')
        axs[1].set_title('Stdevs')
        plt.legend(fc.L.range(self))
