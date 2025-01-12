import os, sys, logging, argparse
import torch, math
from torch.optim import lr_scheduler
from functools import partial
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.data import DistributedSampler

sys.path.append(os.path.join(os.getcwd(), '..'))
from utils.DataLoaders import Get_Dataset, Create_DataLoaders
from utils.Learner import Learner
from utils.Callbacks import TrainCB, DeviceCB, MetricsCB, BatchSchedCB, SaveCB
from utils.utils_ML import Adam, set_seed, LayerNorm, MeanAbsoluteError, EDM_MSELoss, MSELoss
from utils.misc import GetModel, GetInds, FindModel    

def Train(rank, world_size, save_dir, load_path):
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.set_printoptions(precision=5, linewidth=140, sci_mode=False)
    logging.disable(logging.WARNING)

    tls_train, tls_valid = Get_Dataset(path='../download/dataset/dataset_2_7/', rank=rank)
    sampler = DistributedSampler(tls_train, num_replicas=4, rank=rank, shuffle=True, drop_last=True)
    dls = Create_DataLoaders(tls_train, tls_valid, batch_size=32, sampler=sampler)

    model, diffu_params = GetModel(device=rank)

    torch.manual_seed(rank) #make sure this seed setting is after Create_DataLoaders which has torch.manual_seed in it    
   
    ddp_model = DDP(model.to(rank), device_ids=[rank])
    
    epochs = 100
    
    lr = 0.001#0.001 #learning rate
    beta1 = 0.9 #memory fraction of past gradients
    beta2 = 0.99 #memomry fraction of past gradients std
    
    step_size = int(1964/4)*10+1 #218*50+1 #how often to change the learning rate, in # of steps (batches), int(num_batches_per_epoch/n_gpus)*num_epochs_to_change_rate+1
    #print('scheduler step_size:', step_size)
    gamma = 0.9 #0.75 #learning rate decaying rate
    
    sv_step_size = 50 #how often to save the model (in # of epochs)
    sv_prefix = 'model_'    
    
    loss_func = EDM_MSELoss(diffu_params=diffu_params) 
    val_loss_func = MSELoss()
    MCB = MetricsCB(mae=MeanAbsoluteError())
    sched = partial(lr_scheduler.StepLR, step_size=step_size, gamma=gamma)
    cbs = [TrainCB(), DeviceCB(), MCB]
    xtra = [BatchSchedCB(sched)]

    if rank==0:
        SCB = SaveCB(epochs=epochs, step_size=sv_step_size, save_dir=save_dir, prefix=sv_prefix)
        xtra += [SCB]
    
    learn = Learner(ddp_model, dls, loss_func=loss_func, val_loss_func=val_loss_func, cbs=cbs+xtra, load_path=load_path, opt_func=torch.optim.Adam, lr=lr, beta1=beta1, beta2=beta2, parallel=True, device_id=rank, rank=rank) 
    learn.fit(epochs)
    
    dist.destroy_process_group()


if __name__ =="__main__":
    n_gpus = torch.cuda.device_count()
    print('n_gpus:', n_gpus)
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus    
    
    train_type, save_dir0, save_ind1, save_ind2 = GetInds()
    save_dir = save_dir0+save_ind1+'/'+save_ind2
    load_path = FindModel(save_dir0+save_ind1+'/'+str(int(save_ind2)-1))
    print('train_type:', train_type, 'load_path:', load_path)
    
    mp.spawn(Train,
             args=(world_size, save_dir, load_path),
             nprocs=world_size,
             join=True)




