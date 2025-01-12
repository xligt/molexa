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
from utils.Learner import Learner
from utils.Callbacks import TrainCB, DeviceCB, MetricsCB, BatchSchedCB, SaveCB, OptResetCB
from utils.utils_ML import Adam, set_seed, LayerNorm, MeanAbsoluteError, EDM_MSELoss, MSELoss, EDM_MSELoss_EvalLoss
from utils.misc import GetModel, GetInds, FindModel


    

def Train(rank, world_size, save_dir, load_path):
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.set_printoptions(precision=5, linewidth=140, sci_mode=False)
    logging.disable(logging.WARNING)
    
    batch_size = 16

    tls_train, tls_valid = Get_Dataset(path='../download/dataset/dataset_2_7/', rank=rank)
    sampler = DistributedSampler(tls_train, num_replicas=4, rank=rank, shuffle=True, drop_last=True)
    dls = Create_DataLoaders(tls_train, tls_valid, batch_size=batch_size, sampler=sampler)

    loss_eval_c = 0.01
    if rank==0:
        print('batch size:', batch_size)
        print('loss_eval_c:', loss_eval_c)
    err_bin_center = torch.arange(0, 10, 0.05)
    denoise = True 
    sample = True
    freeze_preD_0 = False
    partially_freeze_eval = False
    partially_freeze_denoise = False
    exclude_loading = None #'Evaluator.'

    model, diffu_params = GetModel(device=rank, err_bin_center=err_bin_center, denoise=denoise, sample=sample)

    torch.manual_seed(rank) #make sure this seed setting is after Create_DataLoaders which has torch.manual_seed in it   
    
    load_opt = False
    if rank==0: print('load_opt:', load_opt)

##
    if partially_freeze_denoise:
        if rank==0: print('Selectively freeze blocks in Denoiser!')
        for blk in [model.Denoiser.model.EMB_sigma, model.Denoiser.model.Linear_emb, model.Denoiser.model.Linear_x, model.Denoiser.model.Prj_preAtt, model.Denoiser.model.UFO]:
            for param in blk.parameters():
                param.requires_grad = False
        
        for iblk, blk in enumerate(model.Evaluator.Att_Blocks):
            #if iblk >= 1: break
            for param in blk.parameters():
                param.requires_grad = False
##


##
    if partially_freeze_eval:
        if rank==0: print('Selectively freeze blocks in Evaluator!')
        for blk in [model.Evaluator.Linear_x, model.Evaluator.Prj_preAtt, model.Evaluator.UFO, model.Evaluator.Prj_out.lin1]:
            for param in blk.parameters():
                param.requires_grad = False
        
        for iblk, blk in enumerate(model.Evaluator.Att_Blocks):
            #if iblk >= 1: break
            for param in blk.parameters():
                param.requires_grad = False
##


    freeze = True
    freeze_Denoiser = False if (partially_freeze_denoise or denoise) else True
    freeze_preD = True if (freeze_preD_0 or partially_freeze_denoise or not denoise) else False
    freeze_Eval = False if sample else True
    if freeze:
        model = model.to(rank)
        if freeze_Denoiser:
            if rank==0: print('Freeze Denoiser!')
            for blk in [model.Denoiser]:
                for param in blk.parameters():
                    param.requires_grad = False
        if freeze_preD:
            if rank==0: print('Freeze blocks before Denoiser!')
            for blk in [model.EMB_z, model.EMB_q, model.Linear_pos, model.Prj_edge, model.UFO]:
                for param in blk.parameters():
                    param.requires_grad = False
    
            for blk in model.Att_Blocks:
                for param in blk.parameters():
                    param.requires_grad = False
        if freeze_Eval:
            if rank==0: print('Freeze Evaluator!')
            for blk in [model.Evaluator]:
                for param in blk.parameters():
                    param.requires_grad = False

        ddp_model = DDP(model, device_ids=[rank])

    else:
        if rank==0: print('Without freezing!')
        ddp_model = DDP(model.to(rank), device_ids=[rank])
    
    epochs = 100#100
    
    lr = 0.001#0.001#0.001 #learning rate
    beta1 = 0.9 #memory fraction of past gradients
    beta2 = 0.99 #memomry fraction of past gradients std
   
    step_size = 1 # in units of epochs
    gamma = 0.9 #0.75 #learning rate decaying rate
    
    sv_step_size = 2#20 #how often to save the model (in # of epochs)
    sv_prefix = 'model_'    

    
    loss_func = EDM_MSELoss_EvalLoss(diffu_params=diffu_params, loss_eval_c = loss_eval_c, err_bin_center=err_bin_center, device=rank) 
    val_loss_func = MSELoss()
    MCB = MetricsCB(mae=MeanAbsoluteError(), mse_eval=MeanSquaredError())
    
    cbs = [TrainCB(), DeviceCB(), MCB]
    xtra = []#[OptResetCB(step_size=step_size, gamma=gamma)]

    if rank==0:
        SCB = SaveCB(epochs=epochs, step_size=sv_step_size, save_dir=save_dir, prefix=sv_prefix)
        xtra += [SCB]   
    
    learn = Learner(ddp_model, dls, loss_func=loss_func, val_loss_func=val_loss_func, cbs=cbs+xtra, load_path=load_path, opt_func=torch.optim.Adam, lr=lr, beta1=beta1, beta2=beta2, parallel=True, device_id=rank, rank=rank, sampler=sampler, freeze=freeze, load_opt=load_opt, exclude_loading=exclude_loading) 
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





