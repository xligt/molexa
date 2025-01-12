import os, sys, logging, argparse
import torch, math
from torch.optim import lr_scheduler
from functools import partial
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.data import DistributedSampler
from torcheval.metrics import MeanSquaredError

sys.path.append(os.path.join(os.getcwd(), '..'))
from utils.DataLoaders import Get_Dataset, Create_DataLoaders
from utils.DataLoaders_Split import Get_Dataset_Split, Create_DataLoaders_Split
from utils.Learner import Learner
from utils.Callbacks import TrainCB, DeviceCB, MetricsCB, BatchSchedCB, SaveCB
from utils.utils_ML import Adam, set_seed, LayerNorm, MeanAbsoluteError, EDM_MSELoss, MSELoss, EDM_MSELoss_EvalLoss
from utils.misc import GetModel, GetInds



def Train(rank, world_size, save_dir):
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.set_printoptions(precision=5, linewidth=140, sci_mode=False) 
    logging.disable(logging.WARNING)

    batch_size = 16
   
    tls_train, tls_valid = Get_Dataset(path='../download/dataset/dataset_2_7/', rank=rank)
    sampler = DistributedSampler(tls_train, num_replicas=4, rank=rank, shuffle=True, drop_last=True)
    dls = Create_DataLoaders(tls_train, tls_valid, batch_size=batch_size, sampler=sampler)
    
    loss_eval_c = 0.01#1#0.1#0.005#0.05#0.01
    if rank==0:
        print('batch size:', batch_size)
        print('loss_eval_c:', loss_eval_c)
    err_bin_center = torch.arange(0, 10, 0.05) #0.05 #    err_bin_center = torch.arange(0, 10, 0.05) #0.05
    denoise = True 
    sample = True
    freeze_preD_0 = False
    partially_freeze_eval = False
    partially_freeze_denoise = False

    model, diffu_params = GetModel(device=rank, err_bin_center=err_bin_center, denoise=denoise, sample=sample)
    
    
    torch.manual_seed(rank) #make sure this seed setting is after Create_DataLoaders which has torch.manual_seed in it through RandomSplitter, it will make torch.randn and torch.randn_like in diffusion work independently of ranks.

   
    ddp_model = DDP(model.to(rank), device_ids=[rank])
    
    epochs = 100#100
    
    lr = 0.001 #learning rate
    beta1 = 0.9 #memory fraction of past gradients
    beta2 = 0.99 #memomry fraction of past gradients std
    
    #warmups = int(218/4*6)
    step_size = int(218*2000+1) #how often to change the learning rate, in # of steps (batches)
    gamma = 0.8
    
    sv_step_size = 2 #how often to save the model (in # of epochs)
    sv_prefix = 'model_'
       
    
    loss_func = EDM_MSELoss_EvalLoss(diffu_params=diffu_params, loss_eval_c = loss_eval_c, err_bin_center=err_bin_center, device=rank)
    val_loss_func = MSELoss()
    MCB = MetricsCB(mae=MeanAbsoluteError(), mse_eval=MeanSquaredError())

    #sched = partial(lr_scheduler.StepLR, step_size=step_size, gamma=gamma)
    #sched = partial(lr_scheduler.LambdaLR, lr_lambda=lambda epoch: gamma**((epoch-warmups)//step_size)*4 if epoch>warmups else 4*float(epoch)/float(warmups))
    cbs = [TrainCB(), DeviceCB(), MCB]
    xtra = []#[BatchSchedCB(sched)]

    if rank==0:
        SCB = SaveCB(epochs=epochs, step_size=sv_step_size, save_dir=save_dir, prefix=sv_prefix)
        xtra += [SCB]
    
    
    learn = Learner(ddp_model, dls, loss_func=loss_func, val_loss_func=val_loss_func, cbs=cbs+xtra, opt_func=torch.optim.Adam, lr=lr, beta1=beta1, beta2=beta2, parallel=True, device_id=rank, rank=rank, sampler=sampler) 
    
    learn.fit(epochs)
    
    dist.destroy_process_group()


if __name__ =="__main__":
    n_gpus = torch.cuda.device_count()
    print('n_gpus:', n_gpus)
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus    
    train_type, save_dir0, save_ind1, save_ind2 = GetInds()
    save_dir = save_dir0+save_ind1+'/'+save_ind2
    print('train_type:', train_type)
    mp.spawn(Train,
             args=(world_size, save_dir),
             nprocs=world_size,
             join=True)





